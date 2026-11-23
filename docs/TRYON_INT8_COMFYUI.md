# LTX-2.5 outfit try-on video: ComfyUI + int8 (production recipe)

A self-contained recipe for running the fastest variant from this repo's benchmark in your own service (written so you can also hand it to a coding agent). Every value here was tested on 2026-11-23 on an A100-80GB.

## What it does

Input: a user's **front** and **back** full-body photo. Output: a **720×1280 (9:16), 24 fps, 121-frame (5.04 s), silent MP4** of that person doing one smooth, realistic 360° turn (front → side → back → other side → front) in the photographed outfit. The movement is copied from a reference clip through a pre-built **depth motion template**. The prompt is generic and identical for every user; the photos supply the person and the outfit.

Performance (A100-80GB, model weights already cached on disk): ComfyUI boot about 12 s, first video about 75 s (includes model load), **about 37 s per video once warm**, 60 GB peak VRAM. An H100 also works and should be faster.

## Files you need

| File | What it is |
|---|---|
| `docs/tryon_int8_motion_api.json` | The exact tested ComfyUI graph in API format (46 nodes). Load it, set the per-request fields, POST it. |
| `motion_template.mp4` | Depth motion template: 768×1280, 24 fps, 121 frames, grayscale (near = bright). Not shipped: build it from your own reference clip with `src/ltx_testing/motion.py` (section 7). Put it in `ComfyUI/input/`. |

## 1. Software

- Python 3.12; apt: `git build-essential ffmpeg libgl1 libglib2.0-0`
- ComfyUI **v0.46.0**: `git clone --depth 1 --branch v0.46.0 https://github.com/Comfy-Org/ComfyUI.git`
- Lightricks ComfyUI-LTXVideo, tested at commit **`ced6bad92b82f90028e1e1fd5a37527302550fc5`**, cloned into `ComfyUI/custom_nodes/ComfyUI-LTXVideo`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`, then `pip install -r ComfyUI/requirements.txt -r ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt`
- The int8 kernels come with ComfyUI's `comfy-kitchen` requirement. Nothing else is needed.

## 2. Model weights (about 41 GB, all gated)

Create a Hugging Face **read** token on an account that has clicked "Agree and access" on **all three** repos. Each one is gated separately.

| ComfyUI folder | File | HF repo | Size |
|---|---|---|---|
| `models/diffusion_models/` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` (path `diffusion_models/…`) | 21.5 GB |
| `models/text_encoders/` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` (`text_encoders/…`) | 15.4 GB |
| `models/vae/` | `ltx-2.5-video-vae-bf16.safetensors` | `Lightricks/LTX-2.5` (`vae/…`) | 1.5 GB |
| `models/vae/` | `ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5` (`vae/…`) | 0.36 GB |
| `models/latent_upscale_models/` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `Lightricks/LTX-2.5` (`latent_upscale_models/…`) | 1.0 GB |
| `models/loras/` | `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` (detailer) | `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` | 0.33 GB |
| `models/loras/` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` (motion control) | `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | 0.65 GB |

Download example: `hf download Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors --local-dir /weights`. Either point ComfyUI at the files with `extra_model_paths.yaml` (keys: `diffusion_models`, `text_encoders`, `vae`, `latent_upscale_models`, `loras`) or place them in the folders above.

The Union-Control LoRA is an **LTX-2.3** file used on the 2.5 model. That is how Lightricks' own LTX-2.5 Union-Control workflow does it, and it is intended.

The audio VAE is required even though there is no audio: the model generates video and audio together. Audio is never decoded.

## 3. Per-request input prep (do this in your code before calling ComfyUI)

For each photo: convert to RGB, resize to **height 1280** keeping the aspect ratio, then **edge-pad** (repeat border pixels) or centre-crop to **width 768**. Save it into `ComfyUI/input/` (or upload with `POST /upload/image`).

```python
import numpy as np
from PIL import Image

def prepare(src, dst, W=768, H=1280):
    im = Image.open(src).convert("RGB")
    new_w = round(im.width * H / im.height)
    a = np.asarray(im.resize((new_w, H), Image.LANCZOS))
    pad = W - new_w
    if pad > 0:
        a = np.pad(a, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode="edge")
    elif pad < 0:
        a = a[:, -pad // 2 : -pad // 2 + W]
    Image.fromarray(a).save(dst)
```

Photos must be full body (head to shoes), standing upright, roughly centred. A plain background works best, and the photo's background is kept in the video.

## 4. Running a job

1. Start the server once and keep it warm: `python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch`. Wait until `GET /system_stats` returns 200.
2. Load `tryon_int8_motion_api.json` and set these per-request fields:
   - `img_front.inputs.image` and `img_back.inputs.image`: the prepared filenames
   - `noise.inputs.noise_seed`: a **random** seed per request, e.g. `random.randint(0, 2**32 - 1)`. The JSON ships with 42 (the tested value), so always overwrite it. Store the seed with the job, so a result can be reproduced or re-rolled with a different seed if a clip comes out poorly.
   - `save.inputs.filename_prefix`: a unique prefix such as `tryon/<job_id>`
   - Leave `pos.inputs.text` (the generic prompt) unchanged.
3. `POST /prompt` with `{"prompt": graph, "client_id": "<id>"}`, then poll `GET /history/<prompt_id>` until `status.completed` is true.
4. The MP4 is the history output entry with `type == "output"` ending in `.mp4`, under `ComfyUI/output/<subfolder>/<filename>`.

## 5. What the graph does (fixed values; don't change them without re-testing)

- **Models:** UNETLoader (int8 file, `weight_dtype=default`); CLIPLoader (int8 Gemma, `type=ltxv`); VAELoader for the video and audio VAEs; LatentUpscaleModelLoader.
- **Conditioning:** prompt plus negative (`blurry, distorted body, extra limbs, morphing clothes, changing outfit, cut, zoom, camera shake`) → LTXVConditioning at `frame_rate=24`.
- **Photos:** LTXVPreprocess with `img_compression=18`. Front photo via LTXVImgToVideoInplace at frame 0 (strength 1.0). Back photo via LTXVAddGuide at the template's **back frame** (0.9; 76 in the shipped JSON, which was right for the reference clip it was tested with). Front photo via LTXVAddGuide at **frame 120** (0.9).
- **Stage 1 (384×640, 121 frames):** EmptyLTXVLatentVideo plus LTXVEmptyLatentAudio. Then LTXICLoRALoaderModelOnly(union LoRA, 1.0) and LTXAddVideoICLoRAGuide with the template frames (LoadVideo → GetVideoComponents) at **strength 0.6**, using the loader's `latent_downscale_factor`. Sampling: CFGGuider (cfg 1), `euler_ancestral`, ManualSigmas `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`. Then LTXVSeparateAVLatent → LTXVCropGuides.
- **Stage 2 (768×1280):** VAEDecode of the stage-1 video as reference; LTXVLatentUpsampler ×2; the photo guides re-applied; LTXICLoRALoaderModelOnly(detailer, **0.5**) plus LTXAddVideoICLoRAGuide(reference, strength 1.0). Audio latent is the stage-1 one. Sigmas `0.909375, 0.725, 0.421875, 0.0`, `euler_ancestral`, cfg 1.
- **Output:** LTXVCropGuides → VAEDecode → **ImageCrop to 720×1280 at x=24** → CreateVideo(24 fps) → SaveVideo.

## 6. Gotchas (each one cost a debugging cycle)

1. **Width must be a multiple of 128.** Union-Control reads its reference at half the stage-1 size, so 704 or 720 fails with an einops `rearrange` error. Generate at 768 and crop to 720.
2. **ComfyUI caches identical prompts.** The same graph and inputs return the old result in about 3 s without regenerating. A new seed or new photos is enough to force a real run.
3. **`/history` outputs also list the LoadVideo input** (the template). Filter on `type == "output"`.
4. **Keep the server warm.** The first prompt after boot adds about 40 s of model loading.
5. **Union strength 0.6, not 1.0.** At 1.0 the template's floor patch leaks into the background near the feet.
6. **The back-guide frame is tied to the template.** After building your template, set `g_back_s1.inputs.frame_idx` and `g_back_s2.inputs.frame_idx` in the JSON to the frame where it shows the back. `motion.py` writes that value to `template.json`, and its preview strip shows frame numbers.
7. **Use int8, not bf16, on an 80 GB GPU.** bf16 peaks at 77 GB; int8 peaks at 60 GB, is faster (about 36 s vs 40 s) and stays closest to bf16 quality.
8. **Prompt detail doesn't help.** Per-outfit descriptions showed no gain over the generic prompt.

## 7. Building the motion template (once per movement)

Start from any clip of a person doing the target move on a plain background:
1. Crop to the person (bounding box of non-black content).
2. Retime to 121 frames: source frame = `round(i * (n_src - 1) / 120)`.
3. Run `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0) per frame and normalize over the whole clip (1st–99.5th percentile → 0..1, near = bright).
4. Keep only the person: set the outer 30 px to background; background per row = the median over time of each frame's 10th-percentile row value; keep the largest connected region above background + 0.10 that touches the centre columns, dilated 3 px; set everything else to background.
5. Resize to height 1280, edge-pad to width 768, and encode H.264 at 24 fps (crf 10).

Reference implementation: `src/ltx_testing/motion.py` (`uv run modal run src/ltx_testing/motion.py --back-frame N`).

## Reference code in this repo

- `src/ltx_testing/comfy_bench.py`: `build_graph()` generates the JSON; `run_motion()` shows server boot, prompt submission, history polling and output copy.
- `src/ltx_testing/common.py`: `GENERIC_PROMPT`, `prepare_still()`, and the frame and resolution constants.
- `src/ltx_testing/weights.py`: the Hugging Face download list.
