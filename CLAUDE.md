# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rule

Read this file before running any task in this repo, and keep it up to date when the project's structure, commands, or conventions change.

## What this repo is

An open-source benchmark of LTX-2.5 video generation on Modal: precision (bf16 / fp8 / int8) × backend (native `ltx-pipelines` DFRPipeline vs headless ComfyUI), on an outfit try-on workload (front + back photo → 5 s 9:16 turn). It also includes a motion-template pipeline (ComfyUI int8 + Union-Control IC-LoRA) that copies a 360° turn from a reference clip. `README.md` is the user guide; keep it in sync with the CLI options.

## Architecture

Everything heavy runs on Modal; local code only launches jobs and builds the report.

- `common.py`: single source of truth for the job (weights paths on the volume, resolution, frames, keyframe plan, `GENERIC_PROMPT`, `prepare_still`, `discover_subjects`, `with_gpu_retry`). Subjects come from `inputs/subjects/<name>/front.*|back.*[+prompt.txt]`, mounted at `/inputs` in every image.
- `weights.py`: gated HF downloads into volume `ltx25-weights` (Modal secret `huggingface-secret`, overridable via `LTX_HF_SECRET`). The Union-Control LoRA is optional. Also builds ComfyUI fp8 files (no official fp8 exists), quantizing the same layers as Lightricks' int8-convrot build.
- `native_bench.py`: the image installs Lightricks/LTX-2 at a pinned commit via `uv pip install -e packages/ltx-core[natten] -e packages/ltx-pipelines` (the repo has no uv.lock). `run_native_suite` runs all precisions sequentially on one GPU, each in a fresh subprocess (quant patches are process-global). A label ending in `_resident` disables `release_transformer`.
- `ltx_quant.py`: int8 W8A8 `QuantizationPolicy` (modeled on ltx-core's nvfp4 unit policy, companion `weight_scale`, re-quantizes on LoRA fuse) and text-encoder fp8/int8 patching. PromptEncoder reuses one module shell across builds (dispose moves params to meta), so patches must re-quantize any Linear whose weight is floating again.
- `comfy_bench.py`: ComfyUI v0.46.0 + ComfyUI-LTXVideo. `build_graph` emits the API graph (schemas from `/object_info`, see `inspect_nodes`). `run_comfy_suite` = benchmark; `run_motion` = motion-template pipeline (reads `back_frame` from `template.json` on the volume).
- `motion.py`: reference clip in `inputs/motion/` → depth template (Depth-Anything-V2-Small) + `template.json` on the weights volume.
- `report.py`: local; reads `results/<backend>/<label>/<precision>/metrics.json` and `results/motion/metrics_*.json`, writes `results/REPORT.md` + `summary.json` + contact sheets.

## Commands

```sh
uv sync
uv run modal run src/ltx_testing/weights.py::access | ::main [--skip-fp8]
uv run modal run src/ltx_testing/native_bench.py::check            # image + kernel smoke test
uv run modal run src/ltx_testing/native_bench.py::main [--precisions int8 --gpu H100 --label x --runs-per-subject 1]
uv run modal run src/ltx_testing/comfy_bench.py::main  [same options]
uv run modal run src/ltx_testing/comfy_bench.py::inspect_nodes      # dump ComfyUI node schemas (CPU)
uv run modal run src/ltx_testing/motion.py --back-frame 76
uv run modal run src/ltx_testing/comfy_bench.py::motion [--variants s06:0.6]
uv run modal volume get ltx25-bench-results native results/ --force # also comfy, motion
uv run python -m ltx_testing.report results
```

Files with several Modal entrypoints need the `::name` suffix. On Windows machines with `SSLKEYLOGFILE` set, the Modal CLI crashes (`OPENSSL_Applink`): prefix commands with `env -u SSLKEYLOGFILE`.

## Gotchas

- `--require-gpu auto` pins A100 runs to SXM4 (Modal also hands out slower PCIe cards) and retries on `WrongGpu`.
- ComfyUI returns cached outputs for identical prompts; benchmark repeats append whitespace to the prompt.
- `/history` outputs include the LoadVideo input; filter on `type == "output"`.
- Union-Control reads its reference at half the stage-1 size, so the motion width must be a multiple of 128: 768, cropped to 720.
- Front/back guides alone give a 180° turn and back; the motion template fixes it.
- Never commit anything under `inputs/` (except its README) or `results/`; both are git-ignored.
- No test runner or linter is configured.
