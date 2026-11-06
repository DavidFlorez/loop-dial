#!/usr/bin/env python3
"""Deterministically prepare `data/subset_200.json` from `HuggingFaceH4/MATH-500` (`seed=0`).

Uses the exact same nested sampling algorithm as `DavidFlorez/watch-the-loop-break`:
- First 50 problems (`subset_200[:50]`) are sorted `np.random.default_rng(0).choice(500, size=50, replace=False)`.
- Remaining 150 problems (`subset_200[50:200]`) are `np.random.default_rng(0).permutation(remaining_pool)[:150]`.
Thus `S_5 ⊂ S_50 ⊂ S_200` is strictly nested!
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from datasets import load_dataset

from loop_dial.common import DATA_DIR, atomic_write_json


def select_math500_subset(n: int = 200, seed: int = 0) -> List[Dict[str, Any]]:
    """Deterministically draw `n` problems from `HuggingFaceH4/MATH-500` (`test` split) with `S_50 ⊂ S_200`."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "idx": int(i),
            "unique_id": str(r.get("unique_id", f"math500_{i}")),
            "problem": str(r["problem"]),
            "answer": str(r["answer"]),
            "subject": str(r.get("subject", "")),
            "level": int(r["level"]) if str(r.get("level", "")).isdigit() else str(r.get("level", "")),
        }
        for i, r in enumerate(ds)
    ]
    total = len(rows)
    if n <= 50:
        indices = sorted(np.random.default_rng(seed).choice(total, size=n, replace=False).tolist())
        return [rows[i] for i in indices]

    base_idx = sorted(np.random.default_rng(seed).choice(total, size=50, replace=False).tolist())
    base_set = set(base_idx)
    rem_pool = [i for i in range(total) if i not in base_set]
    rem_order = np.random.default_rng(seed).permutation(rem_pool).tolist()
    selected = base_idx + rem_order[: max(0, min(n, total) - 50)]
    return [rows[i] for i in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/subset_200.json from HuggingFaceH4/MATH-500.")
    parser.add_argument("--n", type=int, default=200, help="Subset size (default: 200)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "subset_200.json",
        help="Output JSON path (default: data/subset_200.json)",
    )
    args = parser.parse_args()

    problems = select_math500_subset(n=args.n, seed=args.seed)
    uids_50 = [p["unique_id"] for p in problems[:50]]
    uids_all = [p["unique_id"] for p in problems]
    sha256_50 = hashlib.sha256(json.dumps(uids_50).encode("utf-8")).hexdigest()
    sha256_all = hashlib.sha256(json.dumps(uids_all).encode("utf-8")).hexdigest()

    payload = {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "seed": args.seed,
        "n_total": len(problems),
        "nested_cohorts": {
            "smoke_5": "problems[:5]",
            "pilot_50": "problems[:50]",
            "main_200": "problems[:200]",
        },
        "sha256_uids_50": sha256_50,
        "sha256_uids_200": sha256_all,
        "unique_ids_50": uids_50,
        "unique_ids_200": uids_all,
        "problems": problems,
    }
    atomic_write_json(args.out, payload)
    print(f"Saved {len(problems)} MATH-500 problems to {args.out}")
    print(f"  First 5 smoke IDs: {uids_all[:5]}")
    print(f"  SHA256 (N=50 IDs):  {sha256_50[:16]}...")
    print(f"  SHA256 (N=200 IDs): {sha256_all[:16]}...")


if __name__ == "__main__":
    main()
