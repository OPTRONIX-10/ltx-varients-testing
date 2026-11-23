"""ComfyUI benchmark: the same LTX-2.5 try-on job through a headless ComfyUI server on Modal.

Measures server boot, first (cold, includes model load) and second (warm) prompt latency.
Graph mirrors DFR as closely as ComfyUI allows: distilled transformer, stage 1 at half res
(8 sigmas, euler ancestral), latent x2 upsample, stage 2 with the Pixel-Spatial-Upscaler
(detailer) IC-LoRA at full res (3 sigmas), tiled VAE decode, no audio decode.

    env -u SSLKEYLOGFILE uv run modal run src/ltx_testing/comfy_bench.py --precisions bf16,fp8,int8
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import modal

from ltx_testing.common import (
    BACK_FRAME,
    COMFY_LTX_NODES_REPO,
    COMFY_REPO,
    COMFY_TAG,
    DETAILER_FILE,
    MOTION_TEMPLATE,
    UNION_FILE,
    FPS,
    FRONT_END_FRAME,
    GUIDE_STRENGTH_BACK,
    GUIDE_STRENGTH_END,
    HEIGHT,
    NUM_FRAMES,
    RESULTS_DIR,
    RESULTS_VOLUME,
    SEED,
    discover_subjects,
    WEIGHTS_DIR,
    WEIGHTS_VOLUME,
    WIDTH,
)

app = modal.App("ltx25-comfy-bench")
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

COMFY_DIR = "/opt/ComfyUI"
EXTRA_PATHS = f"""
ltx:
    base_path: {WEIGHTS_DIR}
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    vae: vae
    latent_upscale_models: latent_upscale_models
    loras: loras
"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("uv>=0.9.8")
    .run_commands(
        f"git clone --depth 1 --branch {COMFY_TAG} {COMFY_REPO} {COMFY_DIR}",
        f"git clone --depth 1 {COMFY_LTX_NODES_REPO} {COMFY_DIR}/custom_nodes/ComfyUI-LTXVideo",
        "uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
        f"uv pip install --system -r {COMFY_DIR}/requirements.txt "
        f"-r {COMFY_DIR}/custom_nodes/ComfyUI-LTXVideo/requirements.txt requests",
    )
    .run_commands(f"cat > {COMFY_DIR}/extra_model_paths.yaml <<'EOF'\n{EXTRA_PATHS}\nEOF")
    .add_local_dir("inputs", "/inputs")
    .add_local_python_source("ltx_testing")
)

UNET = {
    "bf16": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "fp8": "ltx-2.5-22b-distilled-transformer-comfy-fp8-e4m3fn.safetensors",
    "int8": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
}
TEXT_ENCODER = {
    "bf16": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "fp8": "gemma4-12b-with-proj-ltx-2.5-comfy-fp8-e4m3fn.safetensors",
    "int8": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
}
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
NEGATIVE = "blurry, distorted body, extra limbs, morphing clothes, changing outfit, cut, zoom, camera shake"


def _start_server(extra_args: list[str]) -> subprocess.Popen:
    cmd = ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", *extra_args]
    return subprocess.Popen(cmd, cwd=COMFY_DIR, stdout=open("/tmp/comfy.log", "w"), stderr=subprocess.STDOUT)


def _wait_ready(proc: subprocess.Popen, timeout: float = 900) -> None:
    import requests

    t_end = time.time() + timeout
    while time.time() < t_end:
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI exited:\n" + open("/tmp/comfy.log").read()[-5000:])
        try:
            if requests.get("http://127.0.0.1:8188/system_stats", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError("ComfyUI did not start")


def build_graph(precision: str, prompt: str, front: str, back: str, prefix: str,
                motion: str | None = None, motion_strength: float = 1.0, back_frame: int = BACK_FRAME,
                seed: int = SEED, width: int = WIDTH, crop_width: int | None = None) -> dict:
    """API-format ComfyUI graph. Node ids are strings; links are [node_id, output_index].

    ``motion``: filename (in ComfyUI/input) of a depth motion template. When set, stage 1 runs with the
    Union-Control IC-LoRA following that depth video, so the clip copies the template's movement.
    """
    g: dict[str, dict] = {}

    def node(nid: str, cls: str, **inputs) -> list:
        g[nid] = {"class_type": cls, "inputs": inputs}
        return [nid, 0]

    def out(nid: str, idx: int) -> list:
        return [nid, idx]

    half_w, half_h = width // 2, HEIGHT // 2

    model = node("unet", "UNETLoader", unet_name=UNET[precision], weight_dtype="default")
    clip = node("clip", "CLIPLoader", clip_name=TEXT_ENCODER[precision], type="ltxv", device="default")
    vae = node("vae", "VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")
    audio_vae = node("avae", "VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors")
    upscaler = node("ups", "LatentUpscaleModelLoader",
                    model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

    front_img = node("pre_front", "LTXVPreprocess", image=node("img_front", "LoadImage", image=front),
                     img_compression=18)
    back_img = node("pre_back", "LTXVPreprocess", image=node("img_back", "LoadImage", image=back),
                    img_compression=18)

    pos = node("pos", "CLIPTextEncode", text=prompt, clip=clip)
    neg = node("neg", "CLIPTextEncode", text=NEGATIVE, clip=clip)
    node("cond", "LTXVConditioning", positive=pos, negative=neg, frame_rate=float(FPS))
    pos, neg = out("cond", 0), out("cond", 1)

    def add_guides(tag: str, latent, pos, neg):
        latent = node(f"i2v_{tag}", "LTXVImgToVideoInplace", vae=vae, image=front_img, latent=latent,
                      strength=1.0, bypass=False)
        node(f"g_back_{tag}", "LTXVAddGuide", positive=pos, negative=neg, vae=vae, latent=latent,
             image=back_img, frame_idx=back_frame, strength=GUIDE_STRENGTH_BACK)
        node(f"g_end_{tag}", "LTXVAddGuide", positive=out(f"g_back_{tag}", 0), negative=out(f"g_back_{tag}", 1),
             vae=vae, latent=out(f"g_back_{tag}", 2), image=front_img, frame_idx=FRONT_END_FRAME,
             strength=GUIDE_STRENGTH_END)
        return out(f"g_end_{tag}", 0), out(f"g_end_{tag}", 1), out(f"g_end_{tag}", 2)

    noise = node("noise", "RandomNoise", noise_seed=seed)
    sampler = node("sampler", "KSamplerSelect", sampler_name="euler_ancestral")

    # ---- stage 1: half resolution, 8 distilled sigmas
    lat1 = node("empty", "EmptyLTXVLatentVideo", width=half_w, height=half_h, length=NUM_FRAMES, batch_size=1)
    p1, n1, lat1 = add_guides("s1", lat1, pos, neg)
    model1 = model
    if motion:
        node("union", "LTXICLoRALoaderModelOnly", model=model, lora_name=UNION_FILE, strength_model=1.0)
        model1 = out("union", 0)
        depth = out(node("motion_vid", "GetVideoComponents",
                         video=node("motion_file", "LoadVideo", file=motion))[0], 0)
        node("motion_guide", "LTXAddVideoICLoRAGuide", positive=p1, negative=n1, vae=vae, latent=lat1,
             image=depth, frame_idx=0, strength=motion_strength, latent_downscale_factor=out("union", 1),
             crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
        p1, n1, lat1 = out("motion_guide", 0), out("motion_guide", 1), out("motion_guide", 2)
    alat = node("empty_audio", "LTXVEmptyLatentAudio", frames_number=NUM_FRAMES, frame_rate=FPS, batch_size=1,
                audio_vae=audio_vae)
    av1 = node("concat1", "LTXVConcatAVLatent", video_latent=lat1, audio_latent=alat)
    guider1 = node("guider1", "CFGGuider", model=model1, positive=p1, negative=n1, cfg=1.0)
    s1 = node("sample1", "SamplerCustomAdvanced", noise=noise, guider=guider1, sampler=sampler,
              sigmas=node("sig1", "ManualSigmas", sigmas=STAGE1_SIGMAS), latent_image=av1)
    node("sep1", "LTXVSeparateAVLatent", av_latent=s1)
    node("crop1", "LTXVCropGuides", positive=p1, negative=n1, latent=out("sep1", 0))
    v1 = out("crop1", 2)

    # ---- stage 2: x2 latent upsample, re-apply guides, detailer IC-LoRA with stage-1 video as reference
    ref = node("dec_ref", "VAEDecode", samples=v1, vae=vae)
    lat2 = node("upsample", "LTXVLatentUpsampler", samples=v1, upscale_model=upscaler, vae=vae)
    node("iclora", "LTXICLoRALoaderModelOnly", model=model, lora_name=DETAILER_FILE, strength_model=0.5)
    p2, n2, lat2 = add_guides("s2", lat2, out("crop1", 0), out("crop1", 1))
    node("icguide", "LTXAddVideoICLoRAGuide", positive=p2, negative=n2, vae=vae, latent=lat2, image=ref,
         frame_idx=0, strength=1.0, latent_downscale_factor=out("iclora", 1), crop="disabled",
         use_tiled_encode=False, tile_size=256, tile_overlap=64)
    av2 = node("concat2", "LTXVConcatAVLatent", video_latent=out("icguide", 2), audio_latent=out("sep1", 1))
    guider2 = node("guider2", "CFGGuider", model=out("iclora", 0), positive=out("icguide", 0),
                   negative=out("icguide", 1), cfg=1.0)
    s2 = node("sample2", "SamplerCustomAdvanced", noise=noise, guider=guider2, sampler=sampler,
              sigmas=node("sig2", "ManualSigmas", sigmas=STAGE2_SIGMAS), latent_image=av2)
    node("sep2", "LTXVSeparateAVLatent", av_latent=s2)
    node("crop2", "LTXVCropGuides", positive=out("icguide", 0), negative=out("icguide", 1),
         latent=out("sep2", 0))
    frames = node("decode", "VAEDecode", samples=out("crop2", 2), vae=vae)
    if crop_width and crop_width < width:
        frames = node("crop_out", "ImageCrop", image=frames, width=crop_width, height=HEIGHT,
                      x=(width - crop_width) // 2, y=0)
    video = node("video", "CreateVideo", images=frames, fps=float(FPS))
    node("save", "SaveVideo", video=video, filename_prefix=prefix, format="auto", codec="auto")
    return g


class _GpuMemPoller(threading.Thread):
    """Polls nvidia-smi for peak used memory (ComfyUI runs in another process)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                self.peak_mib = max(self.peak_mib, int(r.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            self._stop.wait(0.5)

    def reset(self):
        self.peak_mib = 0

    def stop(self):
        self._stop.set()


def _run_prompt(graph: dict) -> dict:
    import requests

    r = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": graph, "client_id": "bench"})
    if not r.ok:
        raise RuntimeError(f"/prompt rejected: {r.status_code} {r.text[:4000]}")
    pid = r.json()["prompt_id"]
    while True:
        h = requests.get(f"http://127.0.0.1:8188/history/{pid}").json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", False):
                msgs = json.dumps(status.get("messages", []))[-4000:]
                raise RuntimeError(f"prompt failed: {msgs}\n" + open("/tmp/comfy.log").read()[-4000:])
            return entry
        time.sleep(0.25)


def _comfy_one(precision: str, gpu_label: str, runs_per_subject: int, poller: "_GpuMemPoller") -> dict:
    import shutil

    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    poller.reset()
    t0 = time.time()
    proc = _start_server([])
    _wait_ready(proc)
    boot_s = time.time() - t0
    print(f"[comfy] server ready in {boot_s:.1f}s")

    out_dir = f"{RESULTS_DIR}/comfy/{gpu_label}/{precision}"
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for s in discover_subjects():
        for rep in range(runs_per_subject):
            prefix = f"bench/{precision}_{s.name}_run{rep}"
            # Trailing spaces bust ComfyUI's output cache so repeats really regenerate (models stay loaded).
            graph = build_graph(precision, s.prompt + " " * rep, f"{s.name}_front.png", f"{s.name}_back.png", prefix)
            poller.reset()
            t0 = time.time()
            entry = _run_prompt(graph)
            wall = time.time() - t0
            msgs = {m[0]: m[1] for m in entry["status"].get("messages", [])}
            exec_s = None
            if "execution_start" in msgs and "execution_success" in msgs:
                exec_s = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000
            files = [v for o in entry.get("outputs", {}).values() for k in ("images", "videos", "gifs")
                     for v in o.get(k, [])]
            dst = None
            for f in files:
                src = os.path.join(COMFY_DIR, "output", f.get("subfolder", ""), f["filename"])
                if src.endswith(".mp4"):
                    dst = f"{out_dir}/{s.name}_run{rep}.mp4"
                    shutil.copy(src, dst)
            run = {
                "subject": s.name,
                "rep": rep,
                "prompt_wall_s": round(wall, 2),
                "comfy_exec_s": round(exec_s, 2) if exec_s else None,
                "peak_vram_used_gb": round(poller.peak_mib / 1024, 2),
                "video": dst.removeprefix(RESULTS_DIR + "/") if dst else None,
            }
            print(f"[comfy] {json.dumps(run)}")
            runs.append(run)
            results_vol.commit()
    proc.terminate()
    proc.wait(timeout=120)
    log_tail = open("/tmp/comfy.log").read()[-6000:]
    with open(f"{out_dir}/comfy.log", "w") as f:
        f.write(open("/tmp/comfy.log").read())
    report = {
        "backend": "comfyui",
        "precision": precision,
        "gpu_label": gpu_label,
        "gpu_name": gpu_name,
        "comfy_version": COMFY_TAG,
        "transformer_file": UNET[precision],
        "text_encoder_file": TEXT_ENCODER[precision],
        "video_vae": "bf16",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "frames": NUM_FRAMES,
        "fps": FPS,
        "server_boot_s": round(boot_s, 2),
        "boot_peak_note": "server boot only; models load lazily on the first prompt",
        "runs": runs,
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    print(log_tail[-2500:])
    return report


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
def run_comfy_suite(precisions: list[str], gpu_label: str, runs_per_subject: int = 2, require_gpu: str = "") -> list[dict]:
    """All precisions sequentially on ONE GPU; a fresh ComfyUI server per precision."""
    _name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    if require_gpu and require_gpu not in _name:
        raise WrongGpu(_name)

    from ltx_testing.common import prepare_still

    for s in discover_subjects():
        prepare_still(s.front, f"{COMFY_DIR}/input/{s.name}_front.png")
        prepare_still(s.back, f"{COMFY_DIR}/input/{s.name}_back.png")
    poller = _GpuMemPoller()
    poller.start()
    reports = []
    for p in precisions:
        # Pre-read this precision's files so disk/volume speed doesn't favour any variant.
        files = [f"{WEIGHTS_DIR}/diffusion_models/{UNET[p]}", f"{WEIGHTS_DIR}/text_encoders/{TEXT_ENCODER[p]}"]
        read_s = {}
        for fpath in files:
            t0 = time.time()
            with open(fpath, "rb") as fh:
                while fh.read(64 << 20):
                    pass
            read_s[os.path.basename(fpath)] = round(time.time() - t0, 1)
        try:
            rep = _comfy_one(p, gpu_label, runs_per_subject, poller)
            rep["volume_read_s"] = read_s
            with open(f"{RESULTS_DIR}/comfy/{gpu_label}/{p}/metrics.json", "w") as fh:
                json.dump(rep, fh, indent=2)
            results_vol.commit()
            reports.append(rep)
        except Exception as e:  # noqa: BLE001  keep going with the other precisions
            print(f"[comfy] {p} failed: {e}")
            reports.append({"precision": p, "error": str(e)[-3000:]})
    poller.stop()
    return reports


@app.local_entrypoint()
def main(precisions: str = "bf16,fp8,int8", gpu: str = "A100-80GB", runs_per_subject: int = 2, label: str = "",
         require_gpu: str = "auto"):
    from ltx_testing.common import with_gpu_retry

    label = label or gpu.lower().replace("-", "")
    if require_gpu == "auto":  # Modal's "A100-80GB" can be SXM4 or PCIe; pin SXM4 for comparable timings
        require_gpu = "SXM4" if gpu.upper().startswith("A100") else ""
    fn = run_comfy_suite.with_options(gpu=gpu)
    for report in with_gpu_retry(fn, precisions.split(","), label, runs_per_subject, require_gpu):
        print(json.dumps(report, indent=2))


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={WEIGHTS_DIR: weights_vol, RESULTS_DIR: results_vol},
    timeout=2 * 3600,
    memory=131072,
    cpu=8,
)
def run_motion(variants: list[tuple[str, float]], precision: str = "int8", seed: int = SEED) -> list[dict]:
    """Motion-template try-on: each variant = (name, union strength). One server, models stay loaded."""
    import shutil

    from ltx_testing.common import (MOTION_BACK_FRAME, MOTION_TEMPLATE_META, MOTION_WIDTH, OUTPUT_WIDTH,
                                    prepare_still)

    if not os.path.exists(MOTION_TEMPLATE):
        raise SystemExit("No motion template on the weights volume. Run: modal run src/ltx_testing/motion.py")
    back_frame = MOTION_BACK_FRAME
    if os.path.exists(MOTION_TEMPLATE_META):
        back_frame = json.load(open(MOTION_TEMPLATE_META))["back_frame"]
    subjects = discover_subjects()
    for s in subjects:
        prepare_still(s.front, f"{COMFY_DIR}/input/{s.name}_front.png", MOTION_WIDTH)
        prepare_still(s.back, f"{COMFY_DIR}/input/{s.name}_back.png", MOTION_WIDTH)
    shutil.copy(MOTION_TEMPLATE, f"{COMFY_DIR}/input/motion_template.mp4")
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    poller = _GpuMemPoller()
    poller.start()
    t0 = time.time()
    proc = _start_server([])
    _wait_ready(proc)
    boot_s = time.time() - t0
    runs = []
    for name, strength in variants:
        out_dir = f"{RESULTS_DIR}/motion/{name}"
        os.makedirs(out_dir, exist_ok=True)
        for s in subjects:
            prefix = f"motion/{name}_{s.name}"
            g = build_graph(precision, s.prompt, f"{s.name}_front.png", f"{s.name}_back.png", prefix,
                            motion="motion_template.mp4" if strength > 0 else None, motion_strength=strength,
                            back_frame=back_frame, seed=seed, width=MOTION_WIDTH,
                            crop_width=OUTPUT_WIDTH)
            poller.reset()
            t1 = time.time()
            entry = _run_prompt(g)
            wall = time.time() - t1
            dst = None
            for o in entry.get("outputs", {}).values():
                for f in o.get("images", []) + o.get("videos", []):
                    src = os.path.join(COMFY_DIR, "output", f.get("subfolder", ""), f["filename"])
                    if src.endswith(".mp4") and f.get("type", "output") == "output":
                        dst = f"{out_dir}/{s.name}.mp4"
                        shutil.copy(src, dst)
            run = {"variant": name, "union_strength": strength, "back_frame": back_frame, "subject": s.name, "precision": precision,
                   "prompt_wall_s": round(wall, 2), "peak_vram_used_gb": round(poller.peak_mib / 1024, 2),
                   "video": dst.removeprefix(RESULTS_DIR + "/") if dst else None}
            print(f"[motion] {json.dumps(run)}")
            runs.append(run)
            results_vol.commit()
    proc.terminate()
    poller.stop()
    report = {"gpu_name": gpu_name, "server_boot_s": round(boot_s, 2), "runs": runs}
    with open(f"{RESULTS_DIR}/motion/metrics_{'_'.join(v[0] for v in variants)}.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return [report]


@app.local_entrypoint()
def motion(variants: str = "union0.6:0.6", precision: str = "int8", seed: int = SEED, gpu: str = "A100-80GB"):
    vs = [(v.split(":")[0], float(v.split(":")[1])) for v in variants.split(",")]
    for r in run_motion.with_options(gpu=gpu).remote(vs, precision, seed):
        print(json.dumps(r, indent=2))


@app.function(image=image, timeout=1800, cpu=4, memory=16384)
def object_info(nodes: list[str]) -> dict:
    """Boot ComfyUI on CPU and return the input schema of the given nodes (to build the graph)."""
    import requests

    proc = _start_server(["--cpu"])
    _wait_ready(proc)
    info = requests.get("http://127.0.0.1:8188/object_info").json()
    proc.terminate()
    missing = [n for n in nodes if n not in info]
    found = {n: {"input": info[n]["input"], "output": info[n].get("output"),
                 "output_name": info[n].get("output_name")} for n in nodes if n in info}
    ltx_like = sorted(k for k in info if "LTX" in k or "ICLoRA" in k)
    return {"missing": missing, "nodes": found, "ltx_nodes": ltx_like}


@app.local_entrypoint()
def inspect_nodes(names: str = ""):
    default = (
        "UNETLoader,CLIPLoader,VAELoader,LatentUpscaleModelLoader,LoadImage,LTXVPreprocess,CLIPTextEncode,"
        "LTXVConditioning,EmptyLTXVLatentVideo,LTXVImgToVideoInplace,LTXVAddGuide,LTXVEmptyLatentAudio,"
        "LTXVConcatAVLatent,LTXVSeparateAVLatent,LTXVCropGuides,LTXVLatentUpsampler,RandomNoise,CFGGuider,"
        "KSamplerSelect,ManualSigmas,SamplerCustomAdvanced,LTXICLoRALoaderModelOnly,LTXAddVideoICLoRAGuide,"
        "VAEDecodeTiled,LTXVTiledVAEDecode,VAEDecode,CreateVideo,SaveVideo"
    )
    res = object_info.remote((names or default).split(","))
    os.makedirs("logs", exist_ok=True)
    with open("logs/comfy_object_info.json", "w") as f:
        json.dump(res, f, indent=1)
    print("missing:", res["missing"])
    print("ltx nodes:", res["ltx_nodes"])
