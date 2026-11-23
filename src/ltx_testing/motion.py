"""Turn a reference clip of someone turning into a reusable depth "motion template" (Union-Control IC-LoRA).

    uv run modal run src/ltx_testing/motion.py --back-frame 76

Reads the first video in inputs/motion/ (any size; black pillar/letterbox bars are cropped away), retimes it
to NUM_FRAMES at FPS, estimates per-frame depth (Depth-Anything-V2-Small, Apache-2.0), normalizes depth over
the whole clip (no flicker), keeps only the person (largest near region touching the centre) and fits it into
the MOTION_WIDTH x HEIGHT canvas. Built once, reused for every try-on: the photos set person and outfit, the
template sets the motion.

--back-frame: output frame (0-120) where the reference shows the person's back. The back photo is guided at
that frame. Check results/motion/template_preview.png (numbered frames) and rebuild if it's off.
"""

from __future__ import annotations

import os

import modal

from ltx_testing.common import (
    FPS,
    HEIGHT,
    MOTION_TEMPLATE,
    MOTION_TEMPLATE_META,
    MOTION_WIDTH,
    NUM_FRAMES,
    WEIGHTS_DIR,
    WEIGHTS_VOLUME,
)

WIDTH = MOTION_WIDTH

app = modal.App("ltx25-motion-template")
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "transformers==4.57.1", "av", "numpy", "pillow", "scipy")
    .add_local_dir("inputs", "/inputs")
    .add_local_python_source("ltx_testing")
)

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _content_bbox(frames):
    import numpy as np

    m = np.max(np.stack(frames[::10]), axis=0).max(axis=2)
    ys, xs = np.where(m > 20)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


@app.function(image=image, gpu="A10G", volumes={WEIGHTS_DIR: weights_vol}, timeout=1800)
def build_template(back_frame: int) -> bytes:
    import json

    import av
    import numpy as np
    import torch
    from PIL import Image
    from transformers import pipeline

    refs = sorted(f for f in os.listdir("/inputs/motion") if f.lower().endswith(VIDEO_EXTS))         if os.path.isdir("/inputs/motion") else []
    if not refs:
        raise SystemExit("Put a reference clip (person turning 360 degrees) in inputs/motion/ first.")
    ref = f"/inputs/motion/{refs[0]}"
    with av.open(ref) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    x0, x1, y0, y1 = _content_bbox(frames)
    n_src = len(frames)
    # Retime the whole reference onto NUM_FRAMES (it is ~4.5 s at 30 fps; ours is 5.04 s at 24 fps).
    picks = [round(i * (n_src - 1) / (NUM_FRAMES - 1)) for i in range(NUM_FRAMES)]
    crops = [Image.fromarray(frames[i][y0:y1, x0:x1]) for i in picks]
    print(f"reference: {n_src} frames, content box x[{x0}:{x1}] y[{y0}:{y1}] -> {NUM_FRAMES} frames")

    depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=0, torch_dtype=torch.float16)
    maps = []
    for k in range(0, len(crops), 16):
        for out in depth(crops[k : k + 16]):
            maps.append(out["predicted_depth"].float().cpu().numpy().squeeze())
    stack = np.stack([np.asarray(Image.fromarray(m).resize(crops[0].size, Image.BICUBIC)) for m in maps])
    lo, hi = np.percentile(stack, 1), np.percentile(stack, 99.5)
    stack = np.clip((stack - lo) / (hi - lo), 0, 1)  # near = bright, normalized over the whole clip
    # Keep only the person (largest connected near region touching the centre columns) and set everything else
    # to the static wall/floor depth, so props in the reference don't leak in and padding can't smear them.
    from scipy import ndimage

    h, w = stack.shape[1:]
    centre = slice(w // 2 - w // 8, w // 2 + w // 8)
    # Static background: per-row depth = median over time of each frame's 10th-percentile row value.
    wall = np.median(np.percentile(stack, 10, axis=2), axis=0)[:, None]
    edge = 30  # props tend to sit at the frame edges; the (centred) person never reaches them
    for t in range(len(stack)):
        m = stack[t]
        m[:, :edge] = wall
        m[:, -edge:] = wall
        labels, n = ndimage.label(m > wall + 0.10)
        if n:
            sizes = ndimage.sum(np.ones_like(m), labels, index=range(1, n + 1))
            centre_ids = set(np.unique(labels[:, centre])) - {0}
            keep = max(centre_ids, key=lambda i: sizes[i - 1]) if centre_ids else int(np.argmax(sizes)) + 1
            person = ndimage.binary_dilation(labels == keep, iterations=3)
        else:
            person = np.zeros_like(m, dtype=bool)
        stack[t] = np.where(person, m, np.broadcast_to(wall, m.shape))

    new_w = round(stack.shape[2] * HEIGHT / stack.shape[1])
    out_frames = []
    for m in stack:
        g = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((new_w, HEIGHT), Image.BICUBIC))
        pad = WIDTH - new_w
        g = np.pad(g, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge") if pad > 0 else g[:, -pad // 2 : -pad // 2 + WIDTH]
        out_frames.append(np.repeat(g[:, :, None], 3, axis=2))

    os.makedirs(os.path.dirname(MOTION_TEMPLATE), exist_ok=True)
    with av.open(MOTION_TEMPLATE, "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        s.options = {"crf": "10"}
        for f in out_frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    with open(MOTION_TEMPLATE_META, "w") as f:
        json.dump({"source": refs[0], "source_frames": n_src, "frames": NUM_FRAMES, "fps": FPS,
                   "width": WIDTH, "height": HEIGHT, "back_frame": back_frame}, f, indent=2)
    weights_vol.commit()
    with open(MOTION_TEMPLATE, "rb") as f:
        return f.read()


def _preview(video: str, dst: str, step: int = 8) -> None:
    """Numbered frame strip of the template, to check where the back is."""
    import av
    from PIL import Image, ImageDraw

    with av.open(video) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = list(range(0, len(frames), step))
    w, h = 96, 160
    sheet = Image.new("RGB", (w * len(idx), h + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        sheet.paste(Image.fromarray(frames[i]).resize((w, h)), (k * w, 18))
        draw.text((k * w + 4, 3), str(i), fill="black")
    sheet.save(dst)


@app.local_entrypoint()
def main(back_frame: int = (NUM_FRAMES - 1) // 2):
    if not 0 < back_frame < NUM_FRAMES - 1:
        raise SystemExit(f"--back-frame must be between 1 and {NUM_FRAMES - 2}")
    data = build_template.remote(back_frame)
    os.makedirs("results/motion", exist_ok=True)
    local = "results/motion/template_depth.mp4"
    with open(local, "wb") as f:
        f.write(data)
    _preview(local, "results/motion/template_preview.png")
    print(f"Template saved to the weights volume and {local}; back photo will guide frame {back_frame}.")
    print("Check results/motion/template_preview.png: if the back view is centred on another frame number,")
    print("rebuild with --back-frame <that number>.")
