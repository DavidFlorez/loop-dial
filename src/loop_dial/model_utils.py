"""Model loading, loop-dial configuration, and physical decoder-layer hook utilities."""

from __future__ import annotations

import copy
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loop_dial.metrics import (
    QWEN35_MODEL_ID,
    SENSENOVA_LOOP_MODEL_ID,
)


def expected_execution_plan(
    n_passes: int,
    *,
    start: int = 4,
    end: int = 28,
    num_layers: int = 32,
) -> Tuple[int, ...]:
    """Return the expected physical decoder-layer execution tuple for `n_passes`."""
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    if not (0 <= start < end <= num_layers):
        raise ValueError(f"Invalid loop bounds [{start}, {end}) for num_layers={num_layers}")
    return tuple(range(0, start)) + tuple(range(start, end)) * int(n_passes) + tuple(range(end, num_layers))


def resolve_language_model(model: Any) -> Any:
    """Locate the inner text decoder (`language_model`) holding `.layers` for SenseNova-Loop or Qwen3.5."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise AttributeError(f"Could not locate decoder `.layers` on model of type {type(model).__name__}")


def get_active_execution_plan(model: Any) -> Tuple[int, ...]:
    """Return the active layer execution plan tuple (`_loop_plan` for SenseNova-Loop, `0..L-1` for Qwen3.5)."""
    lm = resolve_language_model(model)
    plan = getattr(lm, "_loop_plan", None)
    if plan is not None:
        return tuple(int(x) for x in plan)
    return tuple(range(len(lm.layers)))


def set_loop_passes_in_place(model: Any, n_passes: int) -> Tuple[int, ...]:
    """Update `_loop_plan`, `_loop_cache_config`, and `config.loop_settings` on an already-loaded Loop model.

    Useful both as a fallback if `AutoConfig` ever fails to propagate `loop_settings` and for instant
    zero-reload loop-dial switching in interactive GPU sessions (since weights are shared across passes).
    """
    n = int(n_passes)
    if n < 1:
        raise ValueError(f"n_passes must be >= 1, got {n}")
    lm = resolve_language_model(model)
    loop_cfg = dict(getattr(model.config, "loop_settings", None) or {"start": 4, "end": 28})
    start = int(loop_cfg.get("start", 4))
    end = int(loop_cfg.get("end", 28))
    num_layers = len(lm.layers)
    plan = expected_execution_plan(n, start=start, end=end, num_layers=num_layers)

    loop_cfg["eval_count"] = n
    loop_cfg["counts"] = [n]
    model.config.loop_settings = loop_cfg

    lm._loop_plan = plan
    cache_cfg = copy.deepcopy(lm.config)
    cache_cfg.num_hidden_layers = len(plan)
    cache_cfg.layer_types = [lm.config.layer_types[i] for i in plan]
    lm._loop_cache_config = cache_cfg
    return plan


def set_custom_execution_plan_in_place(model: Any, custom_plan: Sequence[int]) -> Tuple[int, ...]:
    """Set a custom physical layer execution tuple (e.g., partial-loop probe entering at 4 and leaving at 27).

    Note: Because two-stream mHC expands hidden states at physical layer 4 (`enter`) and averages them back
    to 1 stream at physical layer 27 (`leave`), any valid sub-loop inside `4..27` must enter at 4 and exit at 27
    without jumping between 1-stream (`0..3, 28..31`) and 2-stream (`4..27`) representations mid-stream.
    """
    plan = tuple(int(i) for i in custom_plan)
    lm = resolve_language_model(model)
    lm._loop_plan = plan
    cache_cfg = copy.deepcopy(lm.config)
    cache_cfg.num_hidden_layers = len(plan)
    cache_cfg.layer_types = [lm.config.layer_types[i] for i in plan]
    lm._loop_cache_config = cache_cfg
    return plan


def load_sensenova_loop_model(
    n_passes: int = 2,
    *,
    model_id: str = SENSENOVA_LOOP_MODEL_ID,
    device: str = "cuda",
    drop_visual: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load `sensenova/SenseNova-Loop-9B-Preview` in `bfloat16` (preserving FP32 mHC routers) at `n_passes`.

    Follows the primary `AutoConfig` override path first, verifies `len(plan) == 8 + 24 * n_passes`,
    and logs whether `AutoConfig` propagated `loop_settings` directly or needed post-load patching.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    n = int(n_passes)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.loop_settings["eval_count"] = n
    cfg.loop_settings["counts"] = [n]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load in bfloat16 while letting HF keep `_keep_in_fp32_modules = ["mhc_attn", "mhc_ffn"]` in float32.
    # IMPORTANT: Move to CUDA via `.to(device)` (device-only) or `device_map`, NEVER `.to(dtype=bfloat16)`!
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if drop_visual and hasattr(model, "model") and hasattr(model.model, "visual"):
        del model.model.visual
        gc.collect()

    model = model.to(device).eval()

    expected_plan = expected_execution_plan(n)
    actual_plan = get_active_execution_plan(model)
    autoconfig_propagated = actual_plan == expected_plan
    if not autoconfig_propagated:
        actual_plan = set_loop_passes_in_place(model, n)

    lm = resolve_language_model(model)
    mhc_dtypes = sorted(
        {
            str(p.dtype)
            for layer in lm.layers
            for attr in ("mhc_attn", "mhc_ffn")
            if hasattr(layer, attr)
            for p in getattr(layer, attr).parameters()
        }
    )

    gpu_mem_allocated_gb = (
        round(torch.cuda.memory_allocated() / (1024**3), 3)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    gpu_mem_reserved_gb = (
        round(torch.cuda.memory_reserved() / (1024**3), 3)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )

    meta = {
        "model_id": model_id,
        "n_passes": n,
        "loop_settings": dict(model.config.loop_settings),
        "plan_len": len(actual_plan),
        "expected_plan_len": len(expected_plan),
        "plan_matches_expected": actual_plan == expected_plan,
        "autoconfig_propagated_cleanly": autoconfig_propagated,
        "mhc_parameter_dtypes": mhc_dtypes,
        "dropped_visual_tower": drop_visual,
        "gpu_mem_allocated_gb": gpu_mem_allocated_gb,
        "gpu_mem_reserved_gb": gpu_mem_reserved_gb,
    }
    return tokenizer, model, meta


def load_qwen35_model(
    *,
    model_id: str = QWEN35_MODEL_ID,
    device: str = "cuda",
    drop_visual: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load `Qwen/Qwen3.5-9B` in `bfloat16` under Hugging Face `transformers`."""
    import torch
    import transformers as tfm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = None
    errors: List[str] = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
        loader = getattr(tfm, cls_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:
            errors.append(f"{cls_name}: {exc}")

    if model is None:
        raise RuntimeError(f"Unable to load {model_id}: {'; '.join(errors)}")

    if drop_visual and hasattr(model, "model") and hasattr(model.model, "visual"):
        del model.model.visual
        gc.collect()

    model = model.to(device).eval()
    plan = get_active_execution_plan(model)
    gpu_mem_allocated_gb = (
        round(torch.cuda.memory_allocated() / (1024**3), 3)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    meta = {
        "model_id": model_id,
        "n_passes": 1,
        "plan_len": len(plan),
        "dropped_visual_tower": drop_visual,
        "gpu_mem_allocated_gb": gpu_mem_allocated_gb,
    }
    return tokenizer, model, meta


class DecoderLayerCallCounter:
    """Context manager registering PyTorch forward hooks on every physical decoder layer (`0..31`).

    Separately counts prefill calls (`seq_len > 1`) vs single-token decode calls (`seq_len == 1`)
    and per-layer visit counts so we can empirically prove `8 + 24*n` layer executions per token.
    """

    def __init__(self, model: Any) -> None:
        self.lm = resolve_language_model(model)
        self.num_physical_layers = len(self.lm.layers)
        self.handles: List[Any] = []
        self.reset()

    def reset(self) -> None:
        self.total_layer_calls: int = 0
        self.prefill_layer_calls: int = 0
        self.decode_layer_calls: int = 0
        self.per_layer_calls: Dict[int, int] = {i: 0 for i in range(self.num_physical_layers)}
        self.forward_steps: int = 0

    def _make_hook(self, layer_idx: int) -> Callable[..., None]:
        def _hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            self.total_layer_calls += 1
            self.per_layer_calls[layer_idx] += 1
            if inputs and hasattr(inputs[0], "shape") and len(inputs[0].shape) >= 2:
                # Note: inside mHC layers (4..27), `hidden_states` entering MHCDecoderLayer.forward
                # has shape (B, S, D) at layer 4 (`enter` happens inside forward) and (B, S, 2, D) at layers 5..27!
                seq_len = int(inputs[0].shape[1])
                if seq_len > 1:
                    self.prefill_layer_calls += 1
                else:
                    self.decode_layer_calls += 1
            else:
                self.decode_layer_calls += 1

        return _hook

    def __enter__(self) -> "DecoderLayerCallCounter":
        self.reset()
        for idx, layer in enumerate(self.lm.layers):
            self.handles.append(layer.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def unload_model(*objs: Any) -> None:
    """Release GPU memory cleanly between `n=1,2,3,4` loads."""
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
