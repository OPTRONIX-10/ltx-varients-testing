"""Native (no ComfyUI) benchmark: LTX-2.5 DFRPipeline from Lightricks/LTX-2 on Modal.

DFR = distilled 22B transformer (8-step stage 1 at half res) + Pixel-Spatial-Upscaler detailing
IC-LoRA (stage 2 at full res) + keyframe-aware decode. Front photo pinned at frame 0, back photo
guided at the middle frame, front photo guided at the last frame -> 360-degree turn.

    env -u SSLKEYLOGFILE uv run modal run src/ltx_testing/native_bench.py --precisions bf16,fp8,int8 --gpu A100-80GB
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import modal

from ltx_testing.common import (
    AUDIO_VAE,
    BACK_FRAME,
    DETAILER_LORA,
    FPS,
    FRONT_END_FRAME,
    GUIDE_STRENGTH_BACK,
    GUIDE_STRENGTH_END,
    HEIGHT,
    LTX2_COMMIT,
    LTX2_REPO,
    NUM_FRAMES,
    RESULTS_DIR,
    RESULTS_VOLUME,
    SEED,
    SPATIAL_UPSAMPLER,
    discover_subjects,
    TEXT_ENCODER_BF16,
    TRANSFORMER_BF16,
    VIDEO_VAE,
    WEIGHTS_DIR,
    WEIGHTS_VOLUME,
    WIDTH,
    prepare_still,
)

app = modal.App("ltx25-native-bench")
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "ffmpeg", "libgl1")
    .pip_install("uv>=0.9.8")
    .run_commands(
        f"git clone {LTX2_REPO} /opt/LTX-2 && cd /opt/LTX-2 && git checkout {LTX2_COMMIT}",
        # uv pip honours the packages' [tool.uv.sources]/indexes (torch cu132, natten wheels).
        "cd /opt/LTX-2 && uv pip install --system --python /usr/local/bin/python "
        "-e 'packages/ltx-core[natten]' -e packages/ltx-pipelines",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("inputs", "/inputs")
    .add_local_python_source("ltx_testing")
)


class _Timed:
    """Wrap a pipeline block so its __call__ is timed (with CUDA sync)."""

    def __init__(self, inner, name: str, sink: dict):
        self._inner, self._name, self._sink = inner, name, sink

    def __call__(self, *a, **kw):
        import torch

        torch.cuda.synchronize()
        t0 = time.time()
        out = self._inner(*a, **kw)
        torch.cuda.synchronize()
        self._sink[self._name] = self._sink.get(self._name, 0.0) + time.time() - t0
        return out

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _timed_fn(fn, name: str, sink: dict):
    def wrapper(*a, **kw):
        import torch

        torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        sink[name] = sink.get(name, 0.0) + time.time() - t0
        return out

    return wrapper


def _prewarm(paths: list[str]) -> dict:
    """Read weight files once so every precision starts from the same (page-cached) state."""
    out = {}
    for p in paths:
        t0 = time.time()
        with open(p, "rb") as f:
            while f.read(64 << 20):
                pass
        out[os.path.basename(p)] = round(time.time() - t0, 1)
    return out


class WrongGpu(RuntimeError):
    """Raised when Modal hands us a different variant of the requested GPU."""


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={WEIGHTS_DIR: weights_vol, RESULTS_DIR: results_vol},
    timeout=4 * 3600,
    memory=131072,
    cpu=8,
)
def run_native_suite(precisions: list[str], gpu_label: str, runs_per_subject: int = 2, require_gpu: str = "") -> list[dict]:
    """All precisions sequentially on ONE GPU, each in a fresh Python process (patches are global)."""
    _name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    if require_gpu and require_gpu not in _name:
        raise WrongGpu(_name)

    import sys

    volume_read_s = _prewarm([TRANSFORMER_BF16, TEXT_ENCODER_BF16, VIDEO_VAE, AUDIO_VAE, SPATIAL_UPSAMPLER,
                              DETAILER_LORA])
    print(f"[native] volume -> page cache read times (s): {volume_read_s}")
    reports = []
    for p in precisions:
        t0 = time.time()
        code = (f"from ltx_testing.native_bench import run_native_impl; "
                f"run_native_impl({p!r}, {gpu_label!r}, {runs_per_subject})")
        rc = subprocess.run([sys.executable, "-c", code], env={**os.environ, "BENCH_SPAWN_T": str(t0), "BENCH_RESIDENT": "1" if gpu_label.endswith("_resident") else "0"}).returncode
        wall = time.time() - t0
        path = f"{RESULTS_DIR}/native/{gpu_label}/{p}/metrics.json"
        if rc != 0 or not os.path.exists(path):
            reports.append({"precision": p, "error": f"exit code {rc}"})
            continue
        rep = json.load(open(path))
        rep["process_wall_s"] = round(wall, 2)
        rep["volume_read_s"] = volume_read_s
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        results_vol.commit()
        reports.append(rep)
    return reports


def run_native_impl(precision: str, gpu_label: str, runs_per_subject: int = 2) -> dict:
    t_fn_start = time.time()
    t_spawn = float(os.environ.get("BENCH_SPAWN_T", t_fn_start))
    import torch

    t0 = time.time()
    import ltx_pipelines.dfr_pipeline as dfr_mod
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_pipelines.dfr_pipeline import DFRPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput

    from ltx_testing import ltx_quant

    t_imports = time.time() - t0

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[native] precision={precision} gpu={gpu_name} torch={torch.__version__}")
    try:
        import natten  # noqa: F401

        natten_ok = True
    except Exception:
        natten_ok = False

    if precision == "bf16":
        quant = None
    elif precision == "fp8":
        from ltx_pipelines.utils.quantization_factory import QuantizationKind

        quant = QuantizationKind.FP8_CAST.to_policy(TRANSFORMER_BF16)
    elif precision == "int8":
        quant = ltx_quant.build_int8_policy()
    else:
        raise ValueError(precision)
    ltx_quant.patch_text_encoder(precision)

    stage_times: dict[str, float] = {}
    from ltx_pipelines.utils.blocks.diffusion_stage import DiffusionStage

    # Time transformer (re)builds separately from denoising: the official pipeline loads the 22B
    # transformer inside each call and frees it at the end (release_transformer).
    DiffusionStage._build_transformer = _timed_fn(DiffusionStage._build_transformer, "transformer_load", stage_times)
    resident = os.environ.get("BENCH_RESIDENT") == "1"
    if resident:
        # Server-style: keep the transformer allocation on the GPU between calls.
        DiffusionStage.release_transformer = lambda self: None
    for name in ("denoise_stage1", "denoise_stage2"):
        setattr(dfr_mod, name, _timed_fn(getattr(dfr_mod, name), name, stage_times))

    t0 = time.time()
    pipe = DFRPipeline(
        model_paths=ModelPaths.from_split(
            transformer_path=TRANSFORMER_BF16,
            text_encoder_path=TEXT_ENCODER_BF16,
            video_vae_path=VIDEO_VAE,
            audio_vae_path=AUDIO_VAE,
        ),
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        loras=(),
        detailing_lora=[LoraPathStrengthAndSDOps(DETAILER_LORA, 0.5, LTXV_LORA_COMFY_RENAMING_MAP)],
        quantization=quant,
    )
    t_construct = time.time() - t0
    pipe.prompt_encoder = _Timed(pipe.prompt_encoder, "text_encode", stage_times)
    pipe.video_decoder = _Timed(pipe.video_decoder, "vae_decode", stage_times)
    pipe.upsampler = _Timed(pipe.upsampler, "latent_upsample", stage_times)
    pipe.audio_decoder = _Timed(pipe.audio_decoder, "audio_decode_discarded", stage_times)

    os.makedirs("/tmp/stills", exist_ok=True)
    out_dir = f"{RESULTS_DIR}/native/{gpu_label}/{precision}"
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for subject in discover_subjects():
        front = prepare_still(subject.front, f"/tmp/stills/{subject.name}_front.png")
        back = prepare_still(subject.back, f"/tmp/stills/{subject.name}_back.png")
        images = [
            ImageConditioningInput.replace(front, 0, 1.0),
            ImageConditioningInput.guide(back, BACK_FRAME, GUIDE_STRENGTH_BACK),
            ImageConditioningInput.guide(front, FRONT_END_FRAME, GUIDE_STRENGTH_END),
        ]
        for rep in range(runs_per_subject):
            stage_times.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                result = pipe(
                    prompt=subject.prompt,
                    seed=SEED,
                    height=HEIGHT,
                    width=WIDTH,
                    frame_rate=float(FPS),
                    images=images,
                    num_frames=NUM_FRAMES,
                    tiling_config=AUTO_TILING,
                )
                torch.cuda.synchronize()
                t1 = time.time()
                # The decoder may return a lazy iterator: decode happens while iterating.
                chunks = [result.video] if isinstance(result.video, torch.Tensor) else list(result.video)
                torch.cuda.synchronize()
                stage_times["vae_decode"] = stage_times.get("vae_decode", 0.0) + time.time() - t1
            t_generate = time.time() - t0
            peak_alloc = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            path = f"{out_dir}/{subject.name}_run{rep}.mp4"
            t0 = time.time()
            encode_video(
                video=iter(chunks), fps=FPS, audio=None, output_path=path, video_chunks_number=len(chunks)
            )
            n_out = int(result.num_frames)
            del chunks, result
            t_encode = time.time() - t0
            breakdown = {k: round(v, 2) for k, v in stage_times.items()}
            other = t_generate - sum(stage_times.values())
            breakdown["other_load_and_misc"] = round(other, 2)
            run = {
                "subject": subject.name,
                "rep": rep,
                "generate_s": round(t_generate, 2),
                "mp4_encode_s": round(t_encode, 2),
                "breakdown_s": breakdown,
                "peak_vram_alloc_gb": round(peak_alloc, 2),
                "peak_vram_reserved_gb": round(peak_reserved, 2),
                "num_frames": n_out,
                "video": path.removeprefix(RESULTS_DIR + "/"),
            }
            print(f"[native] {json.dumps(run)}")
            runs.append(run)
            pass  # committed by run_native_suite

    report = {
        "backend": "native-ltx-pipelines-DFR",
        "mode": "resident" if resident else "official",
        "precision": precision,
        "gpu_label": gpu_label,
        "gpu_name": gpu_name,
        "torch": str(torch.__version__),
        "natten": natten_ok,
        "transformer": {"bf16": "bf16", "fp8": "fp8-cast (official, weight-only on A100)",
                        "int8": "int8 W8A8 dynamic (custom policy)"}[precision],
        "text_encoder": {"bf16": "bf16", "fp8": "fp8 weight-only per-row",
                         "int8": "int8 W8A8 dynamic"}[precision],
        "video_vae": "bf16 (diffusion decoder)",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "frames": NUM_FRAMES,
        "fps": FPS,
        "process_spawn_to_fn_s": round(t_fn_start - t_spawn, 2),
        "python_imports_s": round(t_imports, 2),
        "pipeline_construct_s": round(t_construct, 2),
        "runs": runs,
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    # committed by run_native_suite
    return report


@app.function(image=image, gpu="A100-80GB", timeout=900)
def smoke() -> dict:
    """Check the image (imports, natten, GPU) and the int8/fp8 Linear kernels against bf16."""
    import subprocess

    import torch

    import ltx_pipelines.dfr_pipeline  # noqa: F401
    from ltx_testing import ltx_quant

    info = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__)}
    info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip()
    try:
        import natten

        info["natten"] = natten.__version__
    except Exception as e:  # noqa: BLE001
        info["natten"] = f"missing: {e}"
    torch.manual_seed(0)
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    for kind in ("int8", "fp8"):
        lin = torch.nn.Linear(4096, 16384, device="cuda", dtype=torch.bfloat16)
        ref = lin(x)
        ltx_quant._swap_materialized(lin, kind)
        out = lin(x)
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        for _ in range(3):
            lin(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            lin(x)
        torch.cuda.synchronize()
        info[f"{kind}_rel_err"] = round(rel, 5)
        info[f"{kind}_ms"] = round((time.time() - t0) / 20 * 1000, 3)
    lin = torch.nn.Linear(4096, 16384, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        lin(x)
    torch.cuda.synchronize()
    info["bf16_ms"] = round((time.time() - t0) / 20 * 1000, 3)
    info["int_mm_ok"] = ltx_quant.Int8Linear._int_mm_ok
    return info


@app.local_entrypoint()
def main(precisions: str = "bf16,fp8,int8", gpu: str = "A100-80GB", runs_per_subject: int = 2,
         require_gpu: str = "auto", label: str = ""):
    from ltx_testing.common import with_gpu_retry

    t_local = time.time()
    label = label or gpu.lower().replace("-", "")
    if require_gpu == "auto":  # Modal's "A100-80GB" can be SXM4 or PCIe; pin SXM4 for comparable timings
        require_gpu = "SXM4" if gpu.upper().startswith("A100") else ""
    reports = with_gpu_retry(run_native_suite.with_options(gpu=gpu), precisions.split(","), label,
                              runs_per_subject, require_gpu)
    for report in reports:
        print(json.dumps(report, indent=2))
    print(f"[native] wall clock incl. container scheduling: {time.time() - t_local:.0f}s")


@app.local_entrypoint()
def check():
    print(json.dumps(smoke.remote(), indent=2))
