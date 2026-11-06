#!/usr/bin/env python3
"""Build and verify the frozen `data/aime_2024_2025.json` olympiad benchmark (`N = 60`).

Combines:
- `HuggingFaceH4/aime_2024` (`train` split, 30 problems)
- `MathArena/aime_2025` (`train` split, 30 problems)

Stores gold answers as plain decimal strings (`"204"`, `"70"`) for `grade_math_answer` compatibility,
plus `answer_aime3` zero-padded 3-digit strings (`"204"`, `"070"`) for display and `idx` (`0..59`) for
downstream sweep/stats/replay compatibility.
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

from loop_dial.common import DATA_DIR, atomic_write_json

AIME_2024_DATASET = "HuggingFaceH4/aime_2024"
AIME_2025_DATASET = "MathArena/aime_2025"
AIME_JSON_PATH = DATA_DIR / "aime_2024_2025.json"


def _compute_ids_sha256(unique_ids: List[str]) -> str:
    joined = "\n".join(sorted(unique_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def build_aime_2024_2025() -> Dict[str, Any]:
    from datasets import load_dataset

    ds24 = load_dataset(AIME_2024_DATASET, split="train")
    ds25 = load_dataset(AIME_2025_DATASET, split="train")
    if len(ds24) != 30:
        raise RuntimeError(f"Expected 30 rows in {AIME_2024_DATASET}, got {len(ds24)}")
    if len(ds25) != 30:
        raise RuntimeError(f"Expected 30 rows in {AIME_2025_DATASET}, got {len(ds25)}")

    problems: List[Dict[str, Any]] = []
    for row in ds24:
        ans_str = str(int(str(row["answer"]).strip()))
        ans_int = int(ans_str)
        if not (0 <= ans_int <= 999):
            raise ValueError(f"Unexpected AIME 2024 answer out of 0..999: {row['answer']!r}")
        problems.append(
            {
                "idx": len(problems),
                "unique_id": f"aime_2024/{row['id']}",
                "year": 2024,
                "source": AIME_2024_DATASET,
                "problem": str(row["problem"]),
                "answer": ans_str,
                "answer_aime3": f"{ans_int:03d}",
                "level": None,
                "subject": "AIME",
                "url": row.get("url"),
            }
        )

    for row in ds25:
        ans_str = str(int(str(row["answer"]).strip()))
        ans_int = int(ans_str)
        if not (0 <= ans_int <= 999):
            raise ValueError(f"Unexpected AIME 2025 answer out of 0..999: {row['answer']!r}")
        problems.append(
            {
                "idx": len(problems),
                "unique_id": f"aime_2025/{row['problem_idx']}",
                "year": 2025,
                "source": AIME_2025_DATASET,
                "problem": str(row["problem"]),
                "answer": ans_str,
                "answer_aime3": f"{ans_int:03d}",
                "level": None,
                "subject": "AIME",
                "problem_type": row.get("problem_type"),
            }
        )

    assert len(problems) == 60, f"Expected 60 combined AIME problems, got {len(problems)}"
    unique_ids = [p["unique_id"] for p in problems]
    assert len(set(unique_ids)) == 60, "Duplicate unique_id detected in AIME 2024+2025!"

    ans_24 = [int(p["answer"]) for p in problems if p["year"] == 2024]
    ans_25 = [int(p["answer"]) for p in problems if p["year"] == 2025]

    return {
        "name": "aime_2024_2025",
        "datasets": [
            {"id": AIME_2024_DATASET, "split": "train", "n": 30, "year": 2024},
            {"id": AIME_2025_DATASET, "split": "train", "n": 30, "year": 2025, "license": "cc-by-nc-sa-4.0"},
        ],
        "n_total": len(problems),
        "answer_ranges": {
            "2024": [min(ans_24), max(ans_24)],
            "2025": [min(ans_25), max(ans_25)],
        },
        "sorted_unique_ids_sha256": _compute_ids_sha256(unique_ids),
        "unique_ids": unique_ids,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen data/aime_2024_2025.json (N=60).")
    parser.add_argument("--out", type=Path, default=AIME_JSON_PATH, help="Output JSON path")
    parser.add_argument("--force", action="store_true", help="Rebuild from Hugging Face even if file exists")
    args = parser.parse_args()

    if args.out.is_file() and not args.force:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        probs = existing.get("problems", [])
        uids = [p["unique_id"] for p in probs]
        if len(probs) == 60 and existing.get("sorted_unique_ids_sha256") == _compute_ids_sha256(uids):
            print(
                f"Verified existing {args.out} (N={len(probs)}, sha256={existing['sorted_unique_ids_sha256'][:16]}...)"
            )
            return

    payload = build_aime_2024_2025()
    atomic_write_json(args.out, payload)
    print(f"Wrote {args.out} (N={payload['n_total']})")
    print(f"  2024 answer range: {payload['answer_ranges']['2024']}")
    print(f"  2025 answer range: {payload['answer_ranges']['2025']}")
    print(f"  sorted_unique_ids_sha256: {payload['sorted_unique_ids_sha256']}")
    print(f"  first 3 IDs: {payload['unique_ids'][:3]}")
    print(f"  last 3 IDs : {payload['unique_ids'][-3:]}")


if __name__ == "__main__":
    main()
