# LTX-2.5 precision & backend benchmark on Modal

Benchmark [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) video generation on [Modal](https://modal.com) GPUs:

- **Precision:** bf16 vs fp8 vs int8 (transformer and text encoder; video VAE stays bf16)
- **Backend:** native [`ltx-pipelines`](https://github.com/Lightricks/LTX-2) (no server) vs a headless [ComfyUI](https://github.com/Comfy-Org/ComfyUI) server

The workload is an outfit try-on: a **front** and a **back** full-body photo of a person go in, and a **5-second, 9:16, 720p clip** of that person turning to show the outfit comes out. Every variant runs the same job with the same seed on the same GPU model, and you get a short Markdown report with speed, startup, memory and quality-drift metrics.

An optional **motion-template pipeline** copies a realistic 360° turn from any reference clip you provide (ComfyUI + int8 + Lightricks' Union-Control IC-LoRA). That's the fastest setup we found, and it works with any front and back photo pair.

## What gets compared

| | Native (`ltx-pipelines` DFRPipeline) | ComfyUI (v0.46.0 + ComfyUI-LTXVideo) |
|---|---|---|
| Pipeline | Distilled 22B, 8 steps at half res → ×2 latent upsample → 3 steps at full res with the detailer IC-LoRA (Pixel-Spatial-Upscaler) | Same recipe as a ComfyUI graph |
| bf16 | official bf16 weights | official bf16 weights |
| fp8 | official `fp8-cast` (transformer) + fp8 weight-only text encoder | fp8 e4m3 files built from bf16 by `weights.py` (Lightricks publishes no fp8 LTX-2.5) |
| int8 | custom W8A8 policy (`ltx_quant.py`; ltx-pipelines has no int8 mode) | official Lightricks `comfy-int8-convrot` files |
| Modes | `official` (frees the transformer after each video) and `resident` (keeps it loaded, server-style) | one long-lived server per precision |

Output: 704×1280 (benchmark; DFR needs both sides divisible by 64) or 720×1280 (motion pipeline), 121 frames at 24 fps (5.04 s), no audio.

## Reference results (A100-80GB SXM4, two subjects × two runs)

| Setup | Warm / video | First video (incl. load) | Startup | Peak VRAM | Drift vs bf16 (PSNR) |
|---|---:|---:|---:|---:|---:|
| **ComfyUI int8** | **36.4 s** | 59.3 s | 12.4 s | 58.7 GB | 34.4 dB |
| ComfyUI fp8 | 39.0 s | 62.9 s | 11.7 s | 58.7 GB | 32.2 dB |
| ComfyUI bf16 | 39.9 s | 72.9 s | 17.7 s | 77.3 GB | — |
| Native fp8, resident | 56.6 s | 61.4 s | 6.3 s | 42.3 GB | 28.3 dB |
| Native int8, resident | 56.5 s | 70.0 s | 6.3 s | 42.3 GB | 26.1 dB |
| Native bf16, official | 59–72 s* | 72–89 s* | 9–12 s | 71.2 GB | — |
| **ComfyUI int8 + motion template** (720×1280) | **37.3 s** | 75.7 s | ~12 s | 60.4 GB | — |

\* depends on the Modal host: the official native pipeline reloads the transformer for every video.

What we learned:
- **The A100 has no fp8 tensor cores**, so fp8 saves 19–33 GB of VRAM but no time. Try an H100 for fp8 compute.
- **int8 helps ComfyUI by about 9%** (fused int8 kernels). The custom native int8 matched per-layer microbenchmarks but not the full pipeline, which is dominated by attention.
- **ComfyUI beats the official native pipeline** mostly because it keeps models and LoRAs loaded between prompts. DFR also adds generated keyframes and a heavier keyframe-aware decoder (about 19 s).
- **Front/back guides alone give a 180° turn and back, not a 360°**: with only front and back photos, the model returns the way it came. The motion-template pipeline fixes this.
- Your numbers will differ by GPU, host and Modal region. That's why this repo exists.

## Requirements

- A **Modal** account (`pip install modal` comes with `uv sync`) and enough credit for A100-80GB time. See [Modal pricing](https://modal.com/pricing).
- A **Hugging Face** account with a **read** token. Open each of these pages while logged in and click **"Agree and access"**; each one is gated separately:
  - [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (required)
  - [Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) (required: the detailer)
  - [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) (only for the motion pipeline)
- [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything heavy runs on Modal, so your machine needs no GPU.

## Setup

```sh
git clone <this repo> && cd ltx-testing
uv sync
uv run modal setup                                              # log in to Modal
uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx  # your HF read token
```

To use a different secret name, set `LTX_HF_SECRET=<name>` in your shell before running the commands below.

Then add your inputs, following [inputs/README.md](inputs/README.md):

```
inputs/subjects/<name>/front.png   # full body, facing the camera
inputs/subjects/<name>/back.png    # same outfit, from behind
inputs/motion/reference.mp4        # optional: someone turning 360°, for the motion pipeline
```

`inputs/` is git-ignored, so your photos are never committed.

## Usage

All commands run from the repo root. Each step runs on Modal and stores its outputs in two Modal volumes: `ltx25-weights` and `ltx25-bench-results`.

### 1. Download weights (once, about 110 GB plus 37 GB of generated fp8 files)

```sh
uv run modal run src/ltx_testing/weights.py::access    # optional: check your token can read the gated repos
uv run modal run src/ltx_testing/weights.py::main      # download + build the ComfyUI fp8 files
```

Use `--skip-fp8` to skip building the ComfyUI fp8 files. The download runs on a CPU container and is resumable: files already on the volume are skipped.

### 2. (Optional) Smoke-test the native image

```sh
uv run modal run src/ltx_testing/native_bench.py::check
```

This builds the native image (first time takes about 10 min) and checks the GPU, torch and natten, plus the int8 and fp8 kernels against bf16.

### 3. Run the benchmarks

```sh
# Native ltx-pipelines, all precisions sequentially on one GPU, official mode
uv run modal run src/ltx_testing/native_bench.py::main

# Native, server-style (transformer kept on the GPU): any label ending in _resident
uv run modal run src/ltx_testing/native_bench.py::main --label a100_resident

# ComfyUI, all precisions sequentially on one GPU, a fresh server per precision
uv run modal run src/ltx_testing/comfy_bench.py::main
```

Common options for both:

| Option | Default | Meaning |
|---|---|---|
| `--precisions` | `bf16,fp8,int8` | Subset to run, e.g. `int8` |
| `--gpu` | `A100-80GB` | Any Modal GPU with ≥ 80 GB, e.g. `H100`, `H200`, `B200` |
| `--runs-per-subject` | `2` | Videos per subject; the first one in a process counts as cold |
| `--label` | derived from `--gpu` | Results folder name; lets you keep several runs side by side |
| `--require-gpu` | `auto` | Substring the GPU name must contain. `auto` pins A100 runs to SXM4, because Modal's `A100-80GB` can also be a slower PCIe card. Use `""` to accept anything. |

Each suite runs every precision one after another in **one container on one GPU**, so the comparison is fair. Before each precision it reads that precision's weights into the page cache, so disk speed doesn't favour any variant.

### 4. (Optional) Motion-template pipeline: a real 360° from any photos

```sh
uv run modal run src/ltx_testing/motion.py --back-frame 60           # build the template from inputs/motion/
uv run modal run src/ltx_testing/comfy_bench.py::motion              # generate for every subject (int8, strength 0.6)
```

1. `motion.py` turns your reference clip into a depth "motion template" (Depth-Anything-V2-Small; the background and props are removed). It saves `results/motion/template_preview.png` with numbered frames.
2. **Set `--back-frame`** to the frame where the person shows their back, check the preview, and rebuild if it's off. The back photo is guided at that frame.
3. `comfy_bench.py::motion` generates each subject at 768×1280 and centre-crops to 720×1280. Variants are `name:strength` pairs, e.g. `--variants s06:0.6,s10:1.0`. We found 0.6 cleaner than 1.0. Also available: `--precision`, `--seed`, `--gpu`.

For using this in production, see [docs/TRYON_INT8_COMFYUI.md](docs/TRYON_INT8_COMFYUI.md). It covers exact weights, the API-format graph ([docs/tryon_int8_motion_api.json](docs/tryon_int8_motion_api.json)), per-request fields and gotchas.

### 5. Get the report

```sh
uv run modal volume get ltx25-bench-results native results/ --force
uv run modal volume get ltx25-bench-results comfy results/ --force
uv run modal volume get ltx25-bench-results motion results/ --force   # if you ran step 4
uv run python -m ltx_testing.report results
```

This writes `results/REPORT.md`, a summary table sorted by speed plus a native stage breakdown and the motion runs. It also writes `results/summary.json` (all raw numbers) and a `*_sheet.png` contact sheet (9 frames) next to every video. The videos themselves are under `results/<backend>/<label>/<precision>/`.

## Metrics

| Metric | How it's measured |
|---|---|
| Startup | ComfyUI: server process start until `/system_stats` answers. Native: Python imports + pipeline construction. Modal container start is excluded for both. |
| First video | First generation in a fresh process or server, including model load (weights already in page cache). |
| Warm / video | Mean of every later video in the same process or server. ComfyUI repeats add whitespace to the prompt, because ComfyUI otherwise returns cached outputs instantly. |
| Videos / GPU-hour | 3600 / warm. |
| Peak VRAM | Native: PyTorch allocator peak. ComfyUI: `nvidia-smi` peak of the process (includes cached blocks). |
| Stage breakdown (native) | Text encode, transformer load + LoRA fuse, denoise (stage 1 + 2), latent upsample, VAE decode. |
| Drift vs bf16 | PSNR and SSIM of each fp8/int8 video against the bf16 video of the same backend, subject and seed. Higher is closer. |
| Keyframe SSIM | SSIM of frames 0 / 60 / 120 against the input photos: a sanity check that the guides were respected, not a quality score. |

## Cost and time (A100-80GB, two subjects × two runs)

| Step | GPU time (approx.) |
|---|---|
| Weights download + fp8 build | CPU container ~15 min, plus a small GPU job ~5 min |
| Native suite (3 precisions) | 20–30 min |
| ComfyUI suite (3 precisions) | 12–15 min |
| Motion template + 2 subjects | ~2 min (A10G) + ~5 min |

The volumes keep about 150 GB of weights between runs (Modal storage pricing applies). Delete them when you're done: `uv run modal volume delete ltx25-weights` and `uv run modal volume delete ltx25-bench-results`.

## Troubleshooting

- **`401` / `403` / `GatedRepoError` while downloading:** accept the terms on every repo listed under Requirements, using the account that owns the token. Each LoRA repo is gated separately. `weights.py::access` tells you which one is missing.
- **`OPENSSL_Uplink ... no OPENSSL_Applink` on Windows:** the `SSLKEYLOGFILE` environment variable is set (some tools set it globally). Unset it for Modal: `env -u SSLKEYLOGFILE uv run modal ...` in Git Bash, or `Remove-Item Env:SSLKEYLOGFILE` in PowerShell.
- **`[retry n/6] got WrongGpu`:** Modal gave you an A100 PCIe while the suite pinned SXM4. It retries automatically; pass `--require-gpu ""` to accept any variant.
- **Native int8 or fp8 is no faster than bf16 on A100:** that's expected; see "What we learned".
- **Motion pipeline: einops `rearrange` error:** the width must be a multiple of 128 (Union-Control uses a half-size reference). Keep `MOTION_WIDTH = 768`.
- **The motion clip turns back instead of going all the way round, or the back photo appears at the wrong moment:** rebuild the template with the correct `--back-frame`.
- **Out of memory:** use a GPU with ≥ 80 GB. bf16 peaks at 71–77 GB.

## Project layout

```
src/ltx_testing/
  common.py        # job definition: weights, resolution, frames, keyframe plan, prompt, subject discovery
  weights.py       # Modal: download gated weights into a volume, build ComfyUI fp8 files
  native_bench.py  # Modal: ltx-pipelines DFRPipeline benchmark (+ smoke test)
  ltx_quant.py     # int8 W8A8 policy + text-encoder fp8/int8 for ltx-pipelines
  comfy_bench.py   # Modal: ComfyUI benchmark, API graph builder, motion-template runner
  motion.py        # Modal: build the depth motion template from a reference clip
  report.py        # local: REPORT.md + summary.json + contact sheets
inputs/            # your photos and reference clip (git-ignored)
docs/              # production recipe for the ComfyUI int8 motion pipeline
```

Upstream versions are pinned in `common.py`: LTX-2 commit, ComfyUI tag, torch builds. Bump them deliberately and rerun every precision when you do.

## License

The code in this repository is licensed under [Apache-2.0](LICENSE).

The models are **not** covered by that license. LTX-2.5 weights and the LTX IC-LoRAs are released by Lightricks under their own licenses (LTX-2.x / LTX-2 Community License; see each Hugging Face page). Depth-Anything-V2-Small is Apache-2.0. You are responsible for complying with the model licenses and for having the rights to the photos and reference clips you use.

## Acknowledgements

[Lightricks LTX-2](https://github.com/Lightricks/LTX-2) and [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and [Modal](https://modal.com).
