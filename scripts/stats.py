#!/usr/bin/env python3
"""Phase C Summary, Nested Budget Curve, Paired McNemar Tests, and Gained/Lost Analysis.

Reads saved per-condition `.jsonl` files under `results/` and writes:
- `results/summary.csv`: Per-condition accuracy (strict & forced), 95% Wilson CIs, cap %, loop %,
  median/mean tokens, GPU seconds, median seconds/problem, and correct answers per GPU-hour.
- `results/budget_curve.csv`: Nested token-budget curve across `2k / 4k / 8k / 16k [/ 32k]`.
- `results/paired_stats.csv`: Exact two-sided McNemar p-values + discordant gained/lost counts
  (`n=1,3,4` vs `n=2`, and `n=2` vs `Qwen/Qwen3.5-9B`, plus all pairs).
- `results/gained_lost_examples.json`: Concrete problem-level examples gained/lost across key comparisons.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from loop_dial.common import RESULTS_DIR, atomic_write_json, atomic_write_text, read_jsonl
from loop_dial.metrics import exact_mcnemar_test, latex_to_plain_text, wilson_ci

CONDITION_ORDER = ["loop_n1", "loop_n2", "loop_n3", "loop_n4", "qwen35_9b"]
CONDITION_LABELS = {
    "loop_n1": "Loop 1 pass (32 layers)",
    "loop_n2": "Loop 2 passes (56 layers, default)",
    "loop_n3": "Loop 3 passes (80 layers)",
    "loop_n4": "Loop 4 passes (104 layers)",
    "qwen35_9b": "Qwen3.5-9B (32 layers, no loop)",
}


def load_condition_records(
    results_dir: Path,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    file_suffix: str = "",
    limit_ids: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load deduplicated per-problem JSONL records ordered by `subset_200` problem order."""
    allow_set = set(limit_ids) if limit_ids is not None else None
    id_rank = {uid: i for i, uid in enumerate(limit_ids)} if limit_ids is not None else {}
    loaded: Dict[str, List[Dict[str, Any]]] = {}

    for cond in conditions:
        p = results_dir / f"{cond}{file_suffix}.jsonl"
        if not p.is_file():
            continue
        raw = read_jsonl(p)
        by_uid: Dict[str, Dict[str, Any]] = {}
        for rec in raw:
            uid = str(rec["unique_id"])
            if allow_set is not None and uid not in allow_set:
                continue
            by_uid[uid] = rec
        if not by_uid:
            continue
        ordered = sorted(
            by_uid.values(),
            key=lambda r: id_rank.get(str(r["unique_id"]), int(r.get("idx", 0))),
        )
        loaded[cond] = ordered
    return loaded


def build_summary_df(cond_records: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cond in CONDITION_ORDER:
        if cond not in cond_records:
            continue
        recs = cond_records[cond]
        n = len(recs)
        if n == 0:
            continue

        n_strict = sum(1 for r in recs if r.get("correct_strict"))
        n_forced = sum(1 for r in recs if r.get("correct_forced"))
        lo_s, hi_s = wilson_ci(n_strict, n)
        lo_f, hi_f = wilson_ci(n_forced, n)

        n_cap = sum(1 for r in recs if r.get("hit_cap"))
        n_loop = sum(1 for r in recs if r.get("loop_onset_token") is not None)
        toks = [int(r.get("n_tokens", 0)) for r in recs]
        gpu_secs = [float(r.get("gpu_seconds_share", 0.0)) for r in recs]
        row_finish_secs = [float(r.get("row_finish_wall_s", 0.0)) for r in recs]
        rep8 = [float(r.get("rep_8gram_ratio", 0.0)) for r in recs]

        total_gpu_s = sum(gpu_secs)
        total_toks = sum(toks)
        tok_per_s = total_toks / max(total_gpu_s, 1e-6)
        correct_strict_per_hr = (n_strict / (total_gpu_s / 3600.0)) if total_gpu_s > 0 else 0.0
        correct_forced_per_hr = (n_forced / (total_gpu_s / 3600.0)) if total_gpu_s > 0 else 0.0

        n_passes = int(recs[0].get("n_passes", 1))
        layer_execs = (8 + 24 * n_passes) if cond.startswith("loop_n") else 32

        rows.append(
            {
                "condition": cond,
                "label": CONDITION_LABELS.get(cond, cond),
                "model_id": recs[0].get("model_id", ""),
                "backend": recs[0].get("backend", ""),
                "n_passes": n_passes,
                "layer_executions_per_token": layer_execs,
                "n_problems": n,
                "max_new_tokens": int(recs[0].get("max_new_tokens", 0)),
                "correct_strict": n_strict,
                "accuracy_strict_pct": round((n_strict / n) * 100.0, 2),
                "ci95_strict_low_pct": round(lo_s * 100.0, 2),
                "ci95_strict_high_pct": round(hi_s * 100.0, 2),
                "correct_forced": n_forced,
                "accuracy_forced_pct": round((n_forced / n) * 100.0, 2),
                "ci95_forced_low_pct": round(lo_f * 100.0, 2),
                "ci95_forced_high_pct": round(hi_f * 100.0, 2),
                "hit_cap_count": n_cap,
                "hit_cap_pct": round((n_cap / n) * 100.0, 2),
                "loop_count": n_loop,
                "loop_pct": round((n_loop / n) * 100.0, 2),
                "mean_tokens": round(statistics.mean(toks), 1),
                "median_tokens": round(float(statistics.median(toks)), 1),
                "mean_rep_8gram_pct": round(statistics.mean(rep8) * 100.0, 2),
                "total_gpu_seconds": round(total_gpu_s, 2),
                "median_gpu_seconds_per_problem": round(float(statistics.median(gpu_secs)), 3),
                "mean_gpu_seconds_per_problem": round(statistics.mean(gpu_secs), 3),
                "median_finish_latency_s": round(float(statistics.median(row_finish_secs)), 3),
                "aggregate_tok_per_s": round(tok_per_s, 2),
                "correct_strict_per_gpu_hour": round(correct_strict_per_hr, 2),
                "correct_forced_per_gpu_hour": round(correct_forced_per_hr, 2),
            }
        )
    return pd.DataFrame(rows)


def build_budget_curve_df(cond_records: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cond in CONDITION_ORDER:
        if cond not in cond_records:
            continue
        recs = cond_records[cond]
        if not recs:
            continue
        budget_keys = sorted(
            {int(k) for r in recs for k in (r.get("budgets") or {}).keys()}
        )
        for b in budget_keys:
            b_str = str(b)
            sub = [r["budgets"][b_str] for r in recs if b_str in (r.get("budgets") or {})]
            if not sub:
                continue
            n = len(sub)
            n_strict = sum(1 for s in sub if s.get("correct_strict"))
            n_forced = sum(1 for s in sub if s.get("correct_forced"))
            n_cap = sum(1 for s in sub if s.get("hit_cap"))
            n_loop = sum(1 for s in sub if s.get("loop_onset_token") is not None)
            lo_s, hi_s = wilson_ci(n_strict, n)
            lo_f, hi_f = wilson_ci(n_forced, n)
            mean_toks = statistics.mean(int(s.get("n_tokens", 0)) for s in sub)
            rows.append(
                {
                    "condition": cond,
                    "label": CONDITION_LABELS.get(cond, cond),
                    "budget": b,
                    "n_problems": n,
                    "correct_strict": n_strict,
                    "accuracy_strict_pct": round((n_strict / n) * 100.0, 2),
                    "ci95_strict_low_pct": round(lo_s * 100.0, 2),
                    "ci95_strict_high_pct": round(hi_s * 100.0, 2),
                    "correct_forced": n_forced,
                    "accuracy_forced_pct": round((n_forced / n) * 100.0, 2),
                    "ci95_forced_low_pct": round(lo_f * 100.0, 2),
                    "ci95_forced_high_pct": round(hi_f * 100.0, 2),
                    "hit_cap_pct": round((n_cap / n) * 100.0, 2),
                    "loop_pct": round((n_loop / n) * 100.0, 2),
                    "mean_tokens": round(mean_toks, 1),
                }
            )
    return pd.DataFrame(rows)


def build_paired_stats_and_examples(
    cond_records: Dict[str, List[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute exact McNemar p-values + discordant gained/lost tables for headline & all pairs."""
    by_cond_uid: Dict[str, Dict[str, Dict[str, Any]]] = {
        c: {str(r["unique_id"]): r for r in recs} for c, recs in cond_records.items()
    }

    headline_pairs: List[Tuple[str, str, str]] = [
        ("loop_n2", "loop_n1", "Turning dial down: 1 pass vs 2 passes (default)"),
        ("loop_n2", "loop_n3", "Turning dial up: 3 passes vs 2 passes (default)"),
        ("loop_n2", "loop_n4", "Turning dial up: 4 passes vs 2 passes (default)"),
        ("qwen35_9b", "loop_n2", "Default Loop (2 passes) vs plain Qwen3.5-9B"),
        ("qwen35_9b", "loop_n1", "1-pass Loop vs plain Qwen3.5-9B"),
        ("qwen35_9b", "loop_n3", "3-pass Loop vs plain Qwen3.5-9B"),
        ("qwen35_9b", "loop_n4", "4-pass Loop vs plain Qwen3.5-9B"),
    ]

    rows: List[Dict[str, Any]] = []
    examples_out: Dict[str, Any] = {}

    for ref_cond, cmp_cond, description in headline_pairs:
        if ref_cond not in by_cond_uid or cmp_cond not in by_cond_uid:
            continue
        ref_map = by_cond_uid[ref_cond]
        cmp_map = by_cond_uid[cmp_cond]
        common_uids = [uid for uid in ref_map.keys() if uid in cmp_map]
        if not common_uids:
            continue

        for metric in ("strict", "forced"):
            field = "correct_strict" if metric == "strict" else "correct_forced"
            pred_field = "pred_strict" if metric == "strict" else "pred_forced"

            vec_ref = [bool(ref_map[u].get(field, False)) for u in common_uids]
            vec_cmp = [bool(cmp_map[u].get(field, False)) for u in common_uids]
            mc = exact_mcnemar_test(vec_ref, vec_cmp)
            lo_ref, hi_ref = wilson_ci(sum(vec_ref), len(common_uids))
            lo_cmp, hi_cmp = wilson_ci(sum(vec_cmp), len(common_uids))

            rows.append(
                {
                    "comparison": f"{cmp_cond}_vs_{ref_cond}",
                    "description": description,
                    "metric": metric,
                    "reference_condition": ref_cond,
                    "comparison_condition": cmp_cond,
                    "n_paired": mc["n_paired"],
                    "ref_correct": sum(vec_ref),
                    "ref_accuracy_pct": round(mc["acc_a"] * 100.0, 2),
                    "ref_ci95_low_pct": round(lo_ref * 100.0, 2),
                    "ref_ci95_high_pct": round(hi_ref * 100.0, 2),
                    "cmp_correct": sum(vec_cmp),
                    "cmp_accuracy_pct": round(mc["acc_b"] * 100.0, 2),
                    "cmp_ci95_low_pct": round(lo_cmp * 100.0, 2),
                    "cmp_ci95_high_pct": round(hi_cmp * 100.0, 2),
                    "diff_cmp_minus_ref_pp": mc["diff_b_minus_a_pp"],
                    "both_correct": mc["n11_both_correct"],
                    "gained_by_cmp": mc["gained_by_b"],
                    "lost_by_cmp": mc["lost_by_b"],
                    "both_wrong": mc["n00_both_wrong"],
                    "discordant": mc["discordant"],
                    "mcnemar_p_value": float(f"{mc['p_value']:.6g}"),
                    "significant_0_05": bool(mc["p_value"] < 0.05),
                }
            )

            if metric == "strict":
                gained_list = []
                lost_list = []
                for uid in common_uids:
                    r_ref = ref_map[uid]
                    r_cmp = cmp_map[uid]
                    ok_ref = bool(r_ref.get(field, False))
                    ok_cmp = bool(r_cmp.get(field, False))
                    if (not ok_ref) and ok_cmp:
                        gained_list.append(
                            {
                                "unique_id": uid,
                                "subject": r_ref.get("subject", ""),
                                "level": r_ref.get("level", ""),
                                "gold_plain": latex_to_plain_text(r_ref.get("gold_answer")),
                                "ref_pred_plain": latex_to_plain_text(r_ref.get(pred_field)),
                                "cmp_pred_plain": latex_to_plain_text(r_cmp.get(pred_field)),
                                "ref_tokens": r_ref.get("n_tokens"),
                                "cmp_tokens": r_cmp.get("n_tokens"),
                                "ref_hit_cap": r_ref.get("hit_cap"),
                                "cmp_hit_cap": r_cmp.get("hit_cap"),
                            }
                        )
                    elif ok_ref and (not ok_cmp):
                        lost_list.append(
                            {
                                "unique_id": uid,
                                "subject": r_ref.get("subject", ""),
                                "level": r_ref.get("level", ""),
                                "gold_plain": latex_to_plain_text(r_ref.get("gold_answer")),
                                "ref_pred_plain": latex_to_plain_text(r_ref.get(pred_field)),
                                "cmp_pred_plain": latex_to_plain_text(r_cmp.get(pred_field)),
                                "ref_tokens": r_ref.get("n_tokens"),
                                "cmp_tokens": r_cmp.get("n_tokens"),
                                "ref_hit_cap": r_ref.get("hit_cap"),
                                "cmp_hit_cap": r_cmp.get("hit_cap"),
                            }
                        )
                examples_out[f"{cmp_cond}_vs_{ref_cond}"] = {
                    "description": description,
                    "n_paired": len(common_uids),
                    "gained_count": len(gained_list),
                    "lost_count": len(lost_list),
                    "gained_examples": gained_list[:15],
                    "lost_examples": lost_list[:15],
                }

    return pd.DataFrame(rows), examples_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute summary.csv, budget_curve.csv, and paired McNemar stats.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--file-suffix", type=str, default="")
    parser.add_argument("--out-prefix", type=str, default="", help="Optional prefix for output CSV/JSON files")
    args = parser.parse_args()

    cond_records = load_condition_records(args.results_dir, file_suffix=args.file_suffix)
    if not cond_records:
        raise SystemExit(f"No condition JSONL files found in {args.results_dir} (suffix={args.file_suffix!r}).")

    summary_df = build_summary_df(cond_records)
    budget_df = build_budget_curve_df(cond_records)
    paired_df, examples = build_paired_stats_and_examples(cond_records)

    pref = args.out_prefix
    summary_path = args.results_dir / f"{pref}summary.csv"
    budget_path = args.results_dir / f"{pref}budget_curve.csv"
    paired_path = args.results_dir / f"{pref}paired_stats.csv"
    examples_path = args.results_dir / f"{pref}gained_lost_examples.json"

    atomic_write_text(summary_path, summary_df.to_csv(index=False))
    atomic_write_text(budget_path, budget_df.to_csv(index=False))
    atomic_write_text(paired_path, paired_df.to_csv(index=False))
    atomic_write_json(examples_path, examples)

    print("=== Summary ===")
    print(summary_df.to_string(index=False))
    if not paired_df.empty:
        print("\n=== Paired McNemar Stats ===")
        print(paired_df.to_string(index=False))


if __name__ == "__main__":
    main()
