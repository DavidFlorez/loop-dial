#!/usr/bin/env python3
"""Recompute or summarize nested token-budget curves (`2k / 4k / 8k / 16k [/ 32k]`) from saved JSONL runs.

Each JSONL trace saved by `scripts/run_sweep.py` already stores exact subword-tokenized prefix evaluations
under `record["budgets"]`. If a trace comes from an external run without `record["budgets"]`, this script
also supports re-slicing prefixes using the official `Qwen/Qwen3.5-9B` / `SenseNova-Loop-9B-Preview` tokenizer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from loop_dial.common import RESULTS_DIR, atomic_write_text, read_jsonl, write_jsonl_gz
from loop_dial.metrics import (
    QWEN35_MODEL_ID,
    compute_loop_onset_from_token_ids,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
)
from scripts.stats import CONDITION_ORDER, build_budget_curve_df, load_condition_records


def backfill_budgets_if_missing(
    records: List[Dict[str, Any]],
    *,
    budgets: List[int],
    tokenizer_id: str = QWEN35_MODEL_ID,
) -> List[Dict[str, Any]]:
    needs_backfill = any("budgets" not in r or not r["budgets"] for r in records)
    if not needs_backfill:
        return records

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    updated: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("budgets"):
            updated.append(rec)
            continue
        text = str(rec.get("text", ""))
        gold = str(rec.get("gold_answer", ""))
        hit_eos_full = bool(rec.get("hit_eos", not rec.get("hit_cap", False)))
        token_ids = tok.encode(text, add_special_tokens=False)
        b_map: Dict[str, Any] = {}
        for b in sorted(budgets):
            if len(token_ids) < b or (len(token_ids) == b and hit_eos_full):
                b_map[str(b)] = {
                    "budget": b,
                    "n_tokens": len(token_ids),
                    "hit_cap": False,
                    "pred_strict": rec.get("pred_strict"),
                    "correct_strict": bool(rec.get("correct_strict")),
                    "pred_forced": rec.get("pred_forced"),
                    "correct_forced": bool(rec.get("correct_forced")),
                    "loop_onset_token": compute_loop_onset_from_token_ids(token_ids),
                }
            else:
                pref_ids = token_ids[:b]
                pref_txt = tok.decode(pref_ids, skip_special_tokens=False)
                ps = extract_last_boxed_strict(pref_txt, hit_cap=True)
                cs = grade_math_answer(ps, gold)
                pa = ps if ps is not None else extract_last_boxed(pref_txt)
                ca = cs if ps is not None else grade_math_answer(pa, gold)
                b_map[str(b)] = {
                    "budget": b,
                    "n_tokens": b,
                    "hit_cap": True,
                    "pred_strict": ps,
                    "correct_strict": cs,
                    "pred_forced": pa,
                    "correct_forced": ca,
                    "loop_onset_token": compute_loop_onset_from_token_ids(pref_ids),
                }
        new_rec = dict(rec)
        new_rec["budgets"] = b_map
        updated.append(new_rec)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive nested token-budget table from saved JSONL runs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--file-suffix", type=str, default="")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    cond_records = load_condition_records(args.results_dir, conditions=CONDITION_ORDER, file_suffix=args.file_suffix)
    if not cond_records:
        raise SystemExit(f"No condition JSONL files found in {args.results_dir}")

    for cond in list(cond_records.keys()):
        cond_records[cond] = backfill_budgets_if_missing(cond_records[cond], budgets=args.budgets)

    df = build_budget_curve_df(cond_records)
    out_csv = args.out_csv or (args.results_dir / f"budget_curve{args.file_suffix}.csv")
    atomic_write_text(out_csv, df.to_csv(index=False))
    print(df.to_string(index=False))
    print(f"\nSaved budget curve CSV to {out_csv}")


if __name__ == "__main__":
    main()
