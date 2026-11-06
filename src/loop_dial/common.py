"""Shared atomic file I/O, subset loading, and JSONL helpers for `loop-dial`."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ASSETS_DIR = REPO_ROOT / "assets"


def atomic_write_text(path: Union[str, Path], content: str) -> None:
    """Atomically write UTF-8 text via a temporary sibling file + `os.replace`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_write_json(path: Union[str, Path], payload: Any) -> None:
    """Atomically write pretty-printed UTF-8 JSON."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    """Append JSON records to a `.jsonl` file and fsync so crashes never lose finished problems."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read all JSON objects from a `.jsonl` or `.jsonl.gz` file."""
    target = Path(path)
    if not target.is_file():
        return []
    opener = gzip.open if target.name.endswith(".gz") else open
    records: List[Dict[str, Any]] = []
    with opener(target, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_jsonl_by_id(path: Union[str, Path], key: str = "unique_id") -> Dict[str, Dict[str, Any]]:
    """Read JSONL records keyed by `unique_id` (latest entry wins on duplicate keys)."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        uid = rec.get(key)
        if isinstance(uid, str):
            out[uid] = rec
    return out


def write_jsonl_gz(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
    """Atomically write records to a `.jsonl.gz` archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(tmp_name, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def load_subset_from_disk(
    n: Optional[int] = None,
    subset_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load the deterministic MATH-500 subset from `data/subset_200.json`."""
    path = Path(subset_path) if subset_path is not None else (DATA_DIR / "subset_200.json")
    if not path.is_file():
        raise FileNotFoundError(f"Subset file not found at {path}. Run scripts/prepare_subset.py first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = payload["problems"] if isinstance(payload, dict) and "problems" in payload else payload
    if n is not None:
        return list(problems[: int(n)])
    return list(problems)
