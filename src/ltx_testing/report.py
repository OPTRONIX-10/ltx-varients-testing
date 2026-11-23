"""Summarize benchmark results into results/REPORT.md (+ summary.json and a contact sheet per video).

    uv run modal volume get ltx25-bench-results native results/ --force
    uv run modal volume get ltx25-bench-results comfy results/ --force
    uv run modal volume get ltx25-bench-results motion results/ --force   # if you ran the motion pipeline
    uv run python -m ltx_testing.report results
"""

from __future__ import annotations

import functools
import glob
import json
import os
import statistics as st
import sys

import numpy as np
from PIL import Image

from ltx_testing.common import BACK_FRAME, FRONT_END_FRAME, discover_subjects, prepare_still

SHEET_FRAMES = [0, 15, 30, 45, 60, 75, 90, 105, 120]


@functools.lru_cache(maxsize=8)
def read_frames(path: str) -> np.ndarray:
    """Decoded frames at half resolution (352x640): enough for SSIM/PSNR and contact sheets."""
    import av

    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24", width=352, height=640) for f in c.decode(video=0)])


def _gray(img: np.ndarray, size=(176, 320)) -> np.ndarray:
    return np.asarray(Image.fromarray(img).convert("L").resize(size, Image.BILINEAR), dtype=np.float64)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(_gray(a), _gray(b), data_range=255))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def contact_sheet(frames: np.ndarray, dst: str) -> None:
    idx = [i for i in SHEET_FRAMES if i < len(frames)]
    sheet = Image.new("RGB", (176 * len(idx), 320), "white")
    for k, i in enumerate(idx):
        sheet.paste(Image.fromarray(frames[i]).resize((176, 320)), (176 * k, 0))
    sheet.save(dst)


def _stills(root: str) -> dict:
    out = {}
    try:
        subjects = discover_subjects("inputs")
    except SystemExit:
        return out
    os.makedirs(f"{root}/_stills", exist_ok=True)
    for s in subjects:
        for side, path in (("front", s.front), ("back", s.back)):
            im = Image.open(prepare_still(path, f"{root}/_stills/{s.name}_{side}.png"))
            out[(s.name, side)] = np.asarray(im.resize((352, 640), Image.LANCZOS))
    return out


def _fmt(v, unit="", nd=1):
    return "—" if v is None else f"{v:.{nd}f}{unit}"


def main(root: str) -> None:
    stills = _stills(root)
    reports = [json.load(open(p)) for p in sorted(glob.glob(f"{root}/*/*/*/metrics.json"))]
    if not reports:
        raise SystemExit(f"No metrics.json under {root}/<backend>/<label>/<precision>/. Download results first.")

    rows = []
    for rep in reports:
        backend = "native" if rep["backend"].startswith("native") else "comfy"
        for run in rep["runs"]:
            row = {"backend": backend, "precision": rep["precision"], "label": rep["gpu_label"], **run}
            if run.get("video") and os.path.exists(f"{root}/{run['video']}"):
                fr = read_frames(f"{root}/{run['video']}")
                s = run["subject"]
                if (s, "front") in stills:
                    row["kf_ssim_front_start"] = round(ssim(fr[0], stills[(s, "front")]), 3)
                    row["kf_ssim_back"] = round(ssim(fr[min(BACK_FRAME, len(fr) - 1)], stills[(s, "back")]), 3)
                    row["kf_ssim_front_end"] = round(ssim(fr[min(FRONT_END_FRAME, len(fr) - 1)], stills[(s, "front")]), 3)
                contact_sheet(fr, f"{root}/{run['video'].removesuffix('.mp4')}_sheet.png")
            rows.append(row)

    # Drift vs the bf16 video of the same backend/label/subject/rep (same seed).
    for row in rows:
        if row["precision"] == "bf16" or not row.get("video"):
            continue
        ref = next((r for r in rows if r["backend"] == row["backend"] and r["label"] == row["label"]
                    and r["precision"] == "bf16" and r["subject"] == row["subject"] and r["rep"] == row["rep"]
                    and r.get("video")), None)
        if ref and os.path.exists(f"{root}/{ref['video']}"):
            a, b = read_frames(f"{root}/{row['video']}"), read_frames(f"{root}/{ref['video']}")
            n = min(len(a), len(b))
            row["vs_bf16_psnr_db"] = round(float(np.mean([psnr(a[i], b[i]) for i in range(0, n, 4)])), 2)
            row["vs_bf16_ssim"] = round(float(np.mean([ssim(a[i], b[i]) for i in range(0, n, 8)])), 3)

    lines = ["# LTX-2.5 benchmark report", ""]
    lines += [
        "Warm = mean of every video after the first in the same process/server. First video includes model load.",
        "Startup = ComfyUI server boot, or native Python imports + pipeline construction (Modal container start "
        "excluded). Drift = PSNR/SSIM vs the bf16 video of the same backend and seed (higher = closer).",
        "",
        "| Backend | Label | GPU | Precision | Startup | First video | Warm / video | Videos / GPU-h | Peak VRAM "
        "| PSNR vs bf16 | SSIM vs bf16 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for rep in reports:
        backend = "native" if rep["backend"].startswith("native") else "comfy"
        runs = rep["runs"]
        times = [r.get("generate_s", r.get("prompt_wall_s")) for r in runs]
        warm = st.mean(times[1:]) if len(times) > 1 else None
        if backend == "native":
            startup = sum(rep.get(k) or 0 for k in ("process_spawn_to_fn_s", "python_imports_s", "pipeline_construct_s"))
            vram = max(r["peak_vram_alloc_gb"] for r in runs)
            tag = f"native ({rep.get('mode', 'official')})"
        else:
            startup, vram, tag = rep["server_boot_s"], max(r["peak_vram_used_gb"] for r in runs), "ComfyUI"
        mine = [r for r in rows if r["backend"] == backend and r["label"] == rep["gpu_label"]
                and r["precision"] == rep["precision"]]
        ps = [r["vs_bf16_psnr_db"] for r in mine if r.get("vs_bf16_psnr_db")]
        ss = [r["vs_bf16_ssim"] for r in mine if r.get("vs_bf16_ssim")]
        entry = {"backend": tag, "label": rep["gpu_label"], "gpu": rep["gpu_name"], "precision": rep["precision"],
                 "startup_s": startup, "first_s": times[0], "warm_s": warm,
                 "videos_per_gpu_hour": 3600 / warm if warm else None, "peak_vram_gb": vram,
                 "psnr": st.mean(ps) if ps else None, "ssim": st.mean(ss) if ss else None}
        summary.append(entry)
    summary.sort(key=lambda e: (e["warm_s"] or 1e9))
    for e in summary:
        lines.append(
            f"| {e['backend']} | {e['label']} | {e['gpu']} | {e['precision']} | {_fmt(e['startup_s'], ' s')} "
            f"| {_fmt(e['first_s'], ' s')} | {_fmt(e['warm_s'], ' s')} | {_fmt(e['videos_per_gpu_hour'], '', 0)} "
            f"| {_fmt(e['peak_vram_gb'], ' GB')} | {_fmt(e['psnr'], ' dB', 1)} | {_fmt(e['ssim'], '', 3)} |"
        )

    native = [r for r in reports if r["backend"].startswith("native")]
    if native:
        lines += ["", "## Native pipeline: warm stage breakdown (s)", "",
                  "Denoise excludes transformer load/LoRA fuse, which is timed separately.", "",
                  "| Label | Precision | Text encode | Transformer load | Denoise | Upsample | VAE decode |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for rep in native:
            warm_runs = rep["runs"][1:] or rep["runs"]
            avg = lambda k: st.mean(r["breakdown_s"].get(k, 0) for r in warm_runs)  # noqa: E731
            load = avg("transformer_load")
            lines.append(
                f"| {rep['gpu_label']} | {rep['precision']} | {avg('text_encode'):.1f} | {load:.1f} "
                f"| {avg('denoise_stage1') + avg('denoise_stage2') - load:.1f} | {avg('latent_upsample'):.1f} "
                f"| {avg('vae_decode'):.1f} |"
            )

    motion = [json.load(open(p)) for p in sorted(glob.glob(f"{root}/motion/metrics_*.json"))]
    if motion:
        lines += ["", "## Motion-template pipeline (ComfyUI)", "",
                  "| Variant | Union strength | Subject | Precision | Time | Peak VRAM | Video |",
                  "|---|---:|---|---|---:|---:|---|"]
        for m in motion:
            for r in m["runs"]:
                lines.append(f"| {r['variant']} | {r['union_strength']} | {r['subject']} | {r['precision']} "
                             f"| {r['prompt_wall_s']:.1f} s | {r['peak_vram_used_gb']:.1f} GB | {r['video']} |")
        lines.append("")
        lines.append("The first run in each motion job includes model loading.")

    with open(f"{root}/summary.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    with open(f"{root}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {root}/REPORT.md, {root}/summary.json and a *_sheet.png contact sheet per video.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
