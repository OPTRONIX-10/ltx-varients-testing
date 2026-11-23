"""Download LTX-2.5 weights into a Modal volume and build the fp8 ComfyUI files.

    uv run modal run src/ltx_testing/weights.py::access   # check the HF token can read every gated repo
    uv run modal run src/ltx_testing/weights.py::main     # download (+ build fp8 files; --skip-fp8 to skip)

Lightricks ships bf16, comfy-int8-convrot and nvfp4 builds of LTX-2.5, but no fp8 one. For the
ComfyUI fp8 variant we write fp8 (e4m3fn, per-tensor scale) files in Comfy's quant format,
quantizing exactly the layers that Lightricks quantized in their int8 build.
"""

from __future__ import annotations

import json
import os
import time

import modal

from ltx_testing.common import (
    DETAILER_FILE,
    DETAILER_REPO,
    HF_SECRET,
    LTX_FILES,
    LTX_REPO,
    TEXT_ENCODER_BF16,
    TEXT_ENCODER_FP8_COMFY,
    TEXT_ENCODER_INT8_COMFY,
    TRANSFORMER_BF16,
    TRANSFORMER_FP8_COMFY,
    TRANSFORMER_INT8_COMFY,
    UNION_FILE,
    UNION_REPO,
    WEIGHTS_DIR,
    WEIGHTS_VOLUME,
)

app = modal.App("ltx25-weights")
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.36", "safetensors", "torch==2.9.1", "numpy")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("ltx_testing")
)


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_vol},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    timeout=3 * 3600,
    cpu=8,
    memory=16384,
)
def download() -> dict:
    from huggingface_hub import hf_hub_download

    sizes = {}
    jobs = [(LTX_REPO, f, WEIGHTS_DIR, True) for f in LTX_FILES]
    jobs.append((DETAILER_REPO, DETAILER_FILE, f"{WEIGHTS_DIR}/loras", True))
    jobs.append((UNION_REPO, UNION_FILE, f"{WEIGHTS_DIR}/loras", False))  # only for the motion pipeline
    for repo, fname, local_dir, required in jobs:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            print(f"skip (exists) {dest}")
        else:
            t0 = time.time()
            try:
                hf_hub_download(repo, fname, local_dir=local_dir)
            except Exception as e:  # noqa: BLE001
                if required:
                    raise SystemExit(f"Could not download {repo}/{fname}: {type(e).__name__}. Accept the "
                                     f"terms at https://huggingface.co/{repo} with the token's account.") from e
                print(f"SKIPPED optional {repo} ({type(e).__name__}); accept https://huggingface.co/{repo} "
                      "and rerun if you want the motion pipeline.")
                continue
            print(f"downloaded {fname} in {time.time() - t0:.0f}s")
            weights_vol.commit()
        sizes[fname] = os.path.getsize(dest)
    return sizes


def _quantized_modules(int8_path: str) -> set[str]:
    """Module prefixes that carry a `.comfy_quant` entry in the official int8 build."""
    from safetensors import safe_open

    with safe_open(int8_path, "pt") as f:
        return {k.removesuffix(".comfy_quant") for k in f.keys() if k.endswith(".comfy_quant")}


def _convert_fp8(src: str, int8_ref: str, dst: str) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    modules = _quantized_modules(int8_ref)
    # Sample the int8 config so we can report what Lightricks used.
    with safe_open(int8_ref, "pt") as f:
        sample = next(iter(sorted(modules)))
        int8_cfg = bytes(f.get_tensor(f"{sample}.comfy_quant").tolist()).decode()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    quant_cfg = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
    out: dict[str, torch.Tensor] = {}
    n_q = 0
    with safe_open(src, "pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            t = f.get_tensor(key)
            mod = key.removesuffix(".weight")
            if key.endswith(".weight") and mod in modules and t.ndim == 2:
                tf = t.to("cuda" if torch.cuda.is_available() else "cpu", torch.float32)
                scale = (tf.abs().amax() / fp8_max).clamp(min=1e-12)
                out[key] = (tf / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).cpu()
                out[f"{mod}.weight_scale"] = scale.reshape(()).cpu()
                out[f"{mod}.comfy_quant"] = quant_cfg.clone()
                n_q += 1
            else:
                out[key] = t
    save_file(out, dst, metadata=metadata)
    return {"dst": dst, "quantized_layers": n_q, "int8_reference_layers": len(modules), "int8_cfg": int8_cfg}


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_vol},
    timeout=2 * 3600,
    gpu="A10G",  # only used to speed up amax/cast
    cpu=8,
    memory=131072,
)
def make_comfy_fp8(which: str) -> dict:
    src, ref, dst = {
        "transformer": (TRANSFORMER_BF16, TRANSFORMER_INT8_COMFY, TRANSFORMER_FP8_COMFY),
        "text_encoder": (TEXT_ENCODER_BF16, TEXT_ENCODER_INT8_COMFY, TEXT_ENCODER_FP8_COMFY),
    }[which]
    if os.path.exists(dst):
        return {"dst": dst, "skipped": True, "size": os.path.getsize(dst)}
    t0 = time.time()
    info = _convert_fp8(src, ref, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    weights_vol.commit()
    info["seconds"] = round(time.time() - t0, 1)
    info["size"] = os.path.getsize(dst)
    return info


@app.local_entrypoint()
def main(skip_fp8: bool = False):
    print(json.dumps(download.remote(), indent=2))
    if not skip_fp8:
        for r in make_comfy_fp8.map(["transformer", "text_encoder"]):
            print(json.dumps(r, indent=2))


@app.function(image=image, secrets=[modal.Secret.from_name(HF_SECRET)], timeout=300)
def check_access(repos: list[str]) -> dict:
    from huggingface_hub import HfApi

    api, out = HfApi(), {}
    for repo in repos:
        try:
            files = [f for f in api.list_repo_files(repo) if f.endswith(".safetensors")]
            api.get_paths_info(repo, files[:1])
            from huggingface_hub import hf_hub_url, get_hf_file_metadata

            get_hf_file_metadata(hf_hub_url(repo, files[0]))
            out[repo] = f"ok ({len(files)} weight files)"
        except Exception as e:  # noqa: BLE001
            out[repo] = f"DENIED: {type(e).__name__}"
    return out


@app.local_entrypoint()
def access(repos: str = f"{LTX_REPO},{DETAILER_REPO},{UNION_REPO}"):
    print(json.dumps(check_access.remote(repos.split(",")), indent=2))


@app.local_entrypoint()
def fp8_only():
    for r in make_comfy_fp8.map(["transformer", "text_encoder"]):
        print(json.dumps(r, indent=2))
