"""Quantization add-ons for the native ltx-pipelines benchmark.

ltx-pipelines (LTX-2 repo) ships fp8 transformer quantization (`fp8-cast`) but no int8, and no
text-encoder quantization at all. This module adds:

* ``build_int8_policy()``: a ``QuantizationPolicy`` that converts the same transformer Linears the
  official fp8-cast policy touches to int8 (per-output-channel scales) at load time, runs them as
  W8A8 with dynamic per-token activation scales on int8 tensor cores (``torch._int_mm``), and
  re-quantizes on LoRA fusion so the DFR detailing IC-LoRA still switches in place. Modeled on
  ``ltx_core.quantization.nvfp4.unit``.
* ``patch_text_encoder(kind)``: swaps the Gemma text encoder's Linears to fp8 (weight-only,
  per-row scale) or int8 (W8A8) right after it is built.

Imported only inside the Modal container (needs ltx-core).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

_INT8_MAX = 127.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

# Same Linear subset as ltx_core.quantization.fp8_cast so fp8 and int8 quantize identical layers.
_KEY_PREFIX = "transformer_blocks."
_LINEAR_SUFFIXES = (".to_q", ".to_k", ".to_v", ".to_out.0", "ff.net.0.proj", "ff.net.2")


def _is_target(module_name: str) -> bool:
    return _KEY_PREFIX in module_name and module_name.endswith(_LINEAR_SUFFIXES)


# ---------------------------------------------------------------- quantize helpers
def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / _INT8_MAX).clamp_(min=1e-12)
    q = torch.round(wf / scale[:, None]).clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale


def quantize_fp8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / _FP8_MAX).clamp_(min=1e-12)
    q = (wf / scale[:, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


# ---------------------------------------------------------------- modules
class Int8Linear(torch.nn.Linear):
    """int8 weight [N, K] + fp32 per-row ``weight_scale`` [N]; W8A8 via torch._int_mm.

    Used through ``__class__`` reassignment like ltx-core's Fp8CastLinear. Inputs with <= 16 rows
    (cuBLAS int8 limit) or a failing int8 GEMM fall back to dequantized bf16.
    """

    weight_scale: torch.nn.Parameter
    _int_mm_ok: bool = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        shape = input.shape
        x = input.reshape(-1, shape[-1])
        out = None
        if Int8Linear._int_mm_ok and x.is_cuda and x.shape[0] > 16:
            try:
                xq, s_x = _quant_act(x)
                acc = torch._int_mm(xq, self.weight.t())
                out = _dequant_out(acc, s_x, self.weight_scale, self.bias, input.dtype)
            except RuntimeError as e:  # unsupported shape/layout on this GPU
                print(f"[ltx_quant] torch._int_mm unavailable ({e}); falling back to dequantized bf16")
                Int8Linear._int_mm_ok = False
        if out is None:
            w = self.weight.to(input.dtype) * self.weight_scale.to(input.dtype)[:, None]
            out = F.linear(x, w, self.bias)
        return out.reshape(*shape[:-1], self.out_features)


def _quant_act_eager(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s_x = (x.abs().amax(dim=-1, keepdim=True).float() / _INT8_MAX).clamp(min=1e-12)
    xq = torch.round(x.float() / s_x).clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return xq, s_x


def _dequant_out_eager(acc, s_x, w_scale, bias, dtype):
    out = acc.float() * s_x * w_scale.float()[None, :]
    if bias is not None:
        out = out + bias.float()
    return out.to(dtype)


# Fused elementwise kernels around the int8 GEMM (eager versions are memory-bound and slow).
# Many distinct (dtype, bias / no-bias, shape) combos -> raise the recompile limit (default 8).
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.recompile_limit = 256
_quant_act = torch.compile(_quant_act_eager, dynamic=True)
_dequant_out = torch.compile(_dequant_out_eager, dynamic=True)


class Fp8WeightOnlyLinear(torch.nn.Linear):
    """fp8 e4m3 weight + fp32 per-row scale, upcast to the input dtype at use (works on A100)."""

    weight_scale: torch.nn.Parameter

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        w = self.weight.to(input.dtype) * self.weight_scale.to(input.dtype)[:, None]
        return F.linear(input, w, self.bias)


def _retype_meta(layer: torch.nn.Linear, cls: type, dtype: torch.dtype) -> None:
    """Retype a (meta) Linear so its params match the converted state-dict entries."""
    layer.__class__ = cls
    dev = layer.weight.device
    layer.weight = torch.nn.Parameter(
        torch.empty(layer.weight.shape, dtype=dtype, device=dev), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((layer.out_features,), dtype=torch.float32, device=dev), requires_grad=False
    )


def _swap_materialized(layer: torch.nn.Linear, kind: str) -> None:
    """Quantize an already-loaded Linear in place."""
    w = layer.weight.data
    q, s = quantize_int8_rowwise(w) if kind == "int8" else quantize_fp8_rowwise(w)
    layer.__class__ = Int8Linear if kind == "int8" else Fp8WeightOnlyLinear
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)


# ---------------------------------------------------------------- transformer int8 policy
def build_int8_policy():
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import KeyValueOperation, SDOps
    from ltx_core.loader.units import FuseRule
    from ltx_core.quantization.policy import QuantizationPolicy

    float_in = {torch.bfloat16, torch.float16, torch.float32}

    def selected(key: str, value: torch.Tensor) -> bool:
        return (
            key.endswith(".weight")
            and _is_target(key.removesuffix(".weight"))
            and value.ndim == 2
            and value.dtype in float_in
        )

    def convert(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = inputs[key]
        if not selected(key, value):
            return {key: value}
        src = value if value.is_cuda else value.to("cuda")
        q, s = quantize_int8_rowwise(src)
        prefix = key.removesuffix(".weight")
        return {key: q, f"{prefix}.weight_scale": s}

    def convert_abstract(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value = inputs[key]
        if not selected(key, value):
            return {key: value}
        prefix = key.removesuffix(".weight")
        return {
            key: torch.empty(value.shape, dtype=torch.int8, device="meta"),
            f"{prefix}.weight_scale": torch.empty((value.shape[0],), dtype=torch.float32, device="meta"),
        }

    stage = (
        SDOps("int8-w8a8-online-convert")
        .with_matching()
        .with_kv_operation(
            KeyValueOperation(convert, convert_abstract, preserve_output_device=True),
            key_prefix=_KEY_PREFIX,
            key_suffix=".weight",
        )
    )

    def swap(model: torch.nn.Module) -> torch.nn.Module:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and _is_target(name):
                _retype_meta(m, Int8Linear, torch.int8)
        return model

    swap_op = ModuleOps(
        "int8-w8a8-swap",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=swap,
    )

    def patch(key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]):
        if weight.dtype != torch.int8:
            return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}
        prefix = key.removesuffix(".weight")
        scale_key = f"{prefix}.weight_scale"
        dev = torch.device("cuda") if torch.cuda.is_available() else weight.device
        deq = weight.to(dev, torch.float32) * clean[scale_key].to(dev, torch.float32)[:, None]
        q, s = quantize_int8_rowwise(deq + delta.to(dev, torch.float32))
        return {key: q.to(weight.device), scale_key: s.to(weight.device)}

    fuse = FuseRule("int8-w8a8", torch.bfloat16, patch, (".weight_scale",))
    return QuantizationPolicy(sd_ops=(stage,), module_ops=(swap_op,), fuse_rule=fuse)


# ---------------------------------------------------------------- text encoder
def patch_text_encoder(kind: str) -> None:
    """Quantize every Linear of the Gemma text encoder right after PromptEncoder builds it."""
    if kind == "bf16":
        return
    from ltx_pipelines.utils.blocks import prompt_encoder as pe

    original = pe.PromptEncoder._build_text_encoder

    float_dtypes = (torch.bfloat16, torch.float16, torch.float32)

    def build_quantized(self):
        # The builder reuses one module shell across builds (dispose() moves params to meta), so a
        # layer swapped on a previous build comes back with freshly loaded float weights: re-quantize
        # every Linear whose weight is floating again, whatever its current class.
        model = original(self)
        n = 0
        with torch.no_grad():
            for name, m in model.named_modules():
                if (
                    isinstance(m, torch.nn.Linear)
                    and m.weight.dtype in float_dtypes
                    and "lm_head" not in name
                    and m.in_features % 16 == 0
                ):
                    _swap_materialized(m, kind)
                    n += 1
        torch.cuda.empty_cache()
        print(f"[ltx_quant] text encoder: {n} Linear layers -> {kind}")
        return model

    pe.PromptEncoder._build_text_encoder = build_quantized
