#!/usr/bin/env python3
"""5-problem partial-middle-layer probe (`results/partial_layer_probe.json`).

Tests what happens when looping only a subset of the 24 middle layers on the 5 fixed sanity problems:
1. Naive config override `loop_settings={"start": 4, "end": 16, "eval_count": 2, "counts": [2]}` —
   verifies whether baked-in `enter` (layer 4) / `leave` (layer 27) mHC stream adapters reject arbitrary sub-ranges.
2. Valid 2-stream interior sub-loops via `set_custom_execution_plan_in_place` (`cap=2048`):
   - `early_middle_2x` (`43` layer calls): `0..4`, `5..15` twice, `16..31`
   - `late_middle_2x` (`43` layer calls): `0..15`, `16..26` twice, `27..31`
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from loop_dial.common import RESULTS_DIR, atomic_write_json, load_subset_from_disk
from loop_dial.metrics import SENSENOVA_LOOP_MODEL_ID
from loop_dial.model_utils import (
    load_sensenova_loop_model,
    set_custom_execution_plan_in_place,
    unload_model,
)
from smoke_test import run_5_problem_sanity_batch


def test_naive_subrange_config(device: str = "cuda") -> Dict[str, Any]:
    """Check what happens if `loop_settings` sets `end=16` instead of `28`."""
    out: Dict[str, Any] = {
        "tested_loop_settings": {"start": 4, "end": 16, "eval_count": 2, "counts": [2]},
    }
    model = None
    try:
        cfg = AutoConfig.from_pretrained(SENSENOVA_LOOP_MODEL_ID, trust_remote_code=True)
        cfg.loop_settings = {"start": 4, "end": 16, "eval_count": 2, "counts": [2]}
        model = AutoModelForCausalLM.from_pretrained(
            SENSENOVA_LOOP_MODEL_ID,
            config=cfg,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            del model.model.visual
        model = model.to(device).eval()
        inp = torch.tensor([[1, 2, 3]], device=device)
        with torch.no_grad():
            _ = model(input_ids=inp, use_cache=False)
        out["succeeded"] = True
    except Exception as exc:
        out["succeeded"] = False
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)[:600]
    finally:
        if model is not None:
            del model
            unload_model()
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problems_5 = load_subset_from_disk(n=5)

    print("1. Testing naive loop_settings start=4, end=16 override...", flush=True)
    naive_res = test_naive_subrange_config(device=device)
    print("   Naive subrange result:", naive_res, flush=True)

    print("2. Loading SenseNova-Loop-9B-Preview once for 2-stream interior partial-loop probes...", flush=True)
    tok, model, _ = load_sensenova_loop_model(n_passes=2, device=device)

    variants = {
        "early_middle_2x": tuple(range(0, 5)) + tuple(range(5, 16)) * 2 + tuple(range(16, 32)),
        "late_middle_2x": tuple(range(0, 16)) + tuple(range(16, 27)) * 2 + tuple(range(27, 32)),
    }
    variant_summaries: List[Dict[str, Any]] = []
    variant_records: List[Dict[str, Any]] = []

    for name, plan in variants.items():
        active = set_custom_execution_plan_in_place(model, plan)
        print(f"   Running {name} (len(plan)={len(active)}) on 5 problems (cap=2048)...", flush=True)
        recs, summ = run_5_problem_sanity_batch(
            model,
            tok,
            problems_5,
            condition=name,
            n_passes=1,
            max_new_tokens=2048,
            device=device,
        )
        summ["plan_len"] = len(active)
        summ["plan_description"] = (
            "Layers 5..15 run twice; 16..26 run once (43 layer calls/tok)"
            if name == "early_middle_2x"
            else "Layers 5..15 run once; 16..26 run twice (43 layer calls/tok)"
        )
        print(
            f"   -> {name}: strict={summ['n_correct_strict']}/5 | hit_cap={summ['n_hit_cap']} | "
            f"coherent={summ['n_coherent']}/5 | mean_toks={summ['mean_tokens']}",
            flush=True,
        )
        variant_summaries.append(summ)
        for r in recs:
            variant_records.append(
                {
                    "condition": name,
                    "unique_id": r["unique_id"],
                    "gold_answer": r["gold_answer"],
                    "pred_strict": r["pred_strict"],
                    "correct_strict": r["correct_strict"],
                    "pred_any": r["pred_any"],
                    "correct_any": r["correct_any"],
                    "n_tokens": r["n_tokens"],
                    "hit_cap": r["hit_cap"],
                    "loop_onset_token": r["loop_onset_token"],
                    "coherence": r["coherence"],
                    "head_200": r["text"][:200],
                    "tail_200": r["text"][-200:],
                }
            )

    del model, tok
    unload_model()

    out_path = RESULTS_DIR / "partial_layer_probe.json"
    atomic_write_json(
        out_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "naive_config_subrange_check": naive_res,
            "interior_partial_loop_summaries": variant_summaries,
            "interior_partial_loop_records": variant_records,
        },
    )
    print(f"Saved partial layer probe -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
