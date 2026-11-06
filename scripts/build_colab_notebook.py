#!/usr/bin/env python3
"""Generate `notebooks/loop_dial_colab.ipynb` with clean executable cells and real logged findings.

Reads `results/smoke.json`, `results/summary.csv`, `results/paired_stats.csv`, and (when present)
`results/aime/summary.csv` + `results/aime/paired_stats.csv` so every summary table and output
snapshot in the notebook strictly reflects logged runs in `results/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
AIME_RESULTS_DIR = RESULTS_DIR / "aime"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "loop_dial_colab.ipynb"


def _md_cell(cell_id: str, text: str) -> Dict[str, Any]:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }


def _code_cell(cell_id: str, code: str, stdout_text: str = "") -> Dict[str, Any]:
    outputs: List[Dict[str, Any]] = []
    exec_count = None
    if stdout_text:
        exec_count = 1
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_text.splitlines(keepends=True),
            }
        )
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "id": cell_id,
        "metadata": {},
        "outputs": outputs,
        "source": code.splitlines(keepends=True),
    }


def _format_logged_summary_markdown() -> str:
    summary_csv = RESULTS_DIR / "summary.csv"
    paired_csv = RESULTS_DIR / "paired_stats.csv"
    if not summary_csv.is_file():
        return (
            "### Logged `MATH-500` Experiment Results\n\n"
            "Run `python scripts/stats.py` after `scripts/run_sweep.py` to populate `results/summary.csv` "
            "and `results/paired_stats.csv`."
        )

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    md_lines = [
        "### Logged `MATH-500` `N = 200` Results Snapshot (from `results/summary.csv`)",
        "",
        "| Dial setting | Layer steps / token | Solved cleanly (`≤16k` tokens) | With cap forced answer | Hit 16k cap | Median tokens | Strict answers / GPU-hr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['layer_executions_per_token']} | "
            f"**{r['accuracy_strict_pct']}%** ({r['correct_strict']}/{r['n_problems']}, CI [{r['ci95_strict_low_pct']}%, {r['ci95_strict_high_pct']}%]) | "
            f"**{r['accuracy_forced_pct']}%** ({r['correct_forced']}/{r['n_problems']}) | "
            f"{r['hit_cap_pct']}% | "
            f"{float(r['median_tokens']):,.0f} | {float(r['correct_strict_per_gpu_hour']):,.1f} |"
        )

    if paired_csv.is_file():
        p_rows = list(csv.DictReader(paired_csv.open("r", encoding="utf-8")))
        if p_rows:
            md_lines.extend(
                [
                    "",
                    "#### Paired Exact McNemar Tests (`MATH-500`, `N = 200` Identical Problems, Two-Sided)",
                    "",
                    "| Comparison | Metric | Accuracy change (pp) | Gained | Lost | Exact McNemar p-value |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: |",
                ]
            )
            for pr in p_rows[:8]:
                pval = float(pr["mcnemar_p_value"])
                p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
                md_lines.append(
                    f"| {pr['description']} | {pr['metric']} | {float(pr['diff_cmp_minus_ref_pp']):+.1f} pp | "
                    f"+{pr['gained_by_cmp']} | -{pr['lost_by_cmp']} | `{p_str}` |"
                )
    return "\n".join(md_lines)


def _format_logged_aime_summary_markdown() -> str:
    summary_csv = AIME_RESULTS_DIR / "summary.csv"
    paired_csv = AIME_RESULTS_DIR / "paired_stats.csv"
    if not summary_csv.is_file():
        return (
            "### Logged `AIME 2024 + 2025` Results (`N = 60`)\n\n"
            "Run `python scripts/stats.py --results-dir results/aime` after the AIME sweep to populate "
            "`results/aime/summary.csv` and `results/aime/paired_stats.csv`."
        )

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    md_lines = [
        "### Logged `AIME 2024 + 2025` `N = 60` Results Snapshot (from `results/aime/summary.csv`)",
        "",
        "| Dial setting | Layer steps / token | Solved cleanly (`≤32k` tokens) | With cap forced answer | Hit 32k cap | Median tokens | Strict answers / GPU-hr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['layer_executions_per_token']} | "
            f"**{r['accuracy_strict_pct']}%** ({r['correct_strict']}/{r['n_problems']}, CI [{r['ci95_strict_low_pct']}%, {r['ci95_strict_high_pct']}%]) | "
            f"**{r['accuracy_forced_pct']}%** ({r['correct_forced']}/{r['n_problems']}) | "
            f"{r['hit_cap_pct']}% | "
            f"{float(r['median_tokens']):,.0f} | {float(r['correct_strict_per_gpu_hour']):,.1f} |"
        )

    if paired_csv.is_file():
        p_rows = list(csv.DictReader(paired_csv.open("r", encoding="utf-8")))
        if p_rows:
            md_lines.extend(
                [
                    "",
                    "#### Paired Exact McNemar Tests (`AIME 2024 + 2025`, `N = 60` Identical Olympiad Problems, Two-Sided)",
                    "",
                    "| Comparison | Metric | Accuracy change (pp) | Gained | Lost | Exact McNemar p-value |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: |",
                ]
            )
            for pr in p_rows:
                pval = float(pr["mcnemar_p_value"])
                p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
                md_lines.append(
                    f"| {pr['description']} | {pr['metric']} | {float(pr['diff_cmp_minus_ref_pp']):+.1f} pp | "
                    f"+{pr['gained_by_cmp']} | -{pr['lost_by_cmp']} | `{p_str}` |"
                )
    return "\n".join(md_lines)


def _format_smoke_stdout() -> str:
    smoke_json = RESULTS_DIR / "smoke.json"
    if not smoke_json.is_file():
        return ""
    payload = json.loads(smoke_json.read_text(encoding="utf-8"))
    out = [
        f"GPU: {payload.get('gpu_name')} ({payload.get('gpu_total_gb')} GB)",
        f"Versions: {payload.get('versions')}",
        "",
        "64-token forward-hook decoder-layer verification:",
    ]
    for row in payload.get("scaling_64_tokens", []):
        out.append(
            f"  {row['condition']:10s} | plan_len={row['plan_len']:3d} | "
            f"calls/tok={row['decoder_layer_calls_per_generated_token']:5.1f} | "
            f"{row['ms_per_token']:7.2f} ms/tok ({row['tokens_per_s']:6.2f} tok/s) | "
            f"peak_mem={row['peak_gpu_mem_gb']:.2f} GB"
        )
    return "\n".join(out) + "\n"


def build_notebook() -> Dict[str, Any]:
    smoke_out = _format_smoke_stdout()
    summary_md = _format_logged_summary_markdown()
    aime_summary_md = _format_logged_aime_summary_markdown()

    cells: List[Dict[str, Any]] = [
        _md_cell(
            "intro",
            """# Turn the Loop Dial: SenseNova-Loop-9B-Preview (1–4 Passes) vs Plain Qwen3.5-9B

**One plain-English question:** `sensenova/SenseNova-Loop-9B-Preview` takes the 32-layer `Qwen3.5-9B` architecture and runs its **middle 24 layers (`4..27`) twice per token** by default (`4 + 24×2 + 4 = 56` layer executions), stitched together with two-stream Manifold-Constrained Hyper-Connections (`mHC`).

What happens at inference time if you turn that **loop dial** to **1, 2, 3, or 4 passes** (`32, 56, 80, or 104` layer steps per token) without retraining—and is it actually more accurate (or faster per GPU-hour) than just running plain **`Qwen/Qwen3.5-9B`** (`32` layers once)?

### Reproducibility Contract
- **Benchmarks**:
  1. **`MATH-500`** (`HuggingFaceH4/MATH-500`, `test` split), fixed `seed = 0`, nested subset `S_5 ⊂ S_50 ⊂ S_200` (`data/subset_200.json`), `16,384` max new tokens.
  2. **`AIME 2024 + 2025`** (`HuggingFaceH4/aime_2024` + `MathArena/aime_2025`, `N = 60` olympiad items in `data/aime_2024_2025.json`), `32,768` max new tokens.
- **Decoding**: `enable_thinking=True`, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `seed=0`, `1` sample/problem, plus 512-token Pass-2 forced answer when hitting the cap.
- **Serving Backend**: Unified `vllm==0.33.0` + upstream PR `vllm-project/vllm#64832` (`SenseNovaLoopForConditionalGeneration` + `hf_overrides["loop_settings"]`) in an isolated `/content/vllm-venv` on a Colab `>= 80 GB` GPU (`G4` RTX PRO 6000 96 GB or `A100` 80 GB) in `bfloat16` (preserving FP32 `mhc_attn` / `mhc_ffn` routers).""",
        ),
        _md_cell(
            "setup-md",
            """## 1. Clone Repository & Install Pinned Dependencies
We pin `transformers==5.18.0` (matching `SenseNova-Loop-9B-Preview`'s `config.json` for Phase A PyTorch hook checks) alongside `fla-core` for Triton GatedDeltaNet kernels, without overwriting Colab's CUDA-matched `torch`.""",
        ),
        _code_cell(
            "setup-code",
            """import os, subprocess, sys
from pathlib import Path

if not Path("/content/loop-dial").exists() and Path("/content").exists():
    subprocess.run(["git", "clone", "https://github.com/DavidFlorez/loop-dial.git", "/content/loop-dial"], check=True)

REPO_ROOT = Path("/content/loop-dial") if Path("/content/loop-dial").exists() else Path.cwd().resolve().parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
constraint = Path("/tmp/torch_pin.txt")
constraint.write_text(f"torch=={torch.__version__}\\n")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-c", str(constraint),
    "transformers==5.18.0", "accelerate", "safetensors",
    "datasets", "math-verify[antlr4_13_2]", "fla-core", "einops",
    "scipy", "pandas", "matplotlib", "gradio", "-e", "."
], check=True)

subprocess.run(["nvidia-smi"], check=False)
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)""",
        ),
        _md_cell(
            "subset-md",
            """## 2. Verify Fixed `MATH-500` (`N=200`) & `AIME 2024+2025` (`N=60`) Subsets
`data/subset_200.json` and `data/aime_2024_2025.json` freeze the exact problem lists and `unique_id` SHA-256 digests so every condition is evaluated on the exact same items.""",
        ),
        _code_cell(
            "subset-code",
            """import json
from loop_dial.common import DATA_DIR, load_subset_from_disk
from loop_dial.metrics import DEFAULT_SAMPLING

subset_200 = load_subset_from_disk(n=200)
aime_payload = json.loads((DATA_DIR / "aime_2024_2025.json").read_text())
aime_60 = aime_payload["problems"]

print(f"Loaded {len(subset_200)} deterministic MATH-500 problems.")
print("First 5 MATH-500 sanity IDs:", [p["unique_id"] for p in subset_200[:5]])
print(f"Loaded {len(aime_60)} AIME 2024+2025 olympiad problems (SHA256={aime_payload['sorted_unique_ids_sha256'][:16]}...).")
print("Shared sampling config:", DEFAULT_SAMPLING)""",
        ),
        _md_cell(
            "smoke-md",
            """## 3. Phase A — Verify the Loop Dial with PyTorch Forward Hooks (`32 / 56 / 80 / 104` Layer Calls)
`SenseNova-Loop-9B-Preview`'s `execution_plan(config)` requires **both** `cfg.loop_settings["eval_count"] = n` and `cfg.loop_settings["counts"] = [n]`, while keeping `mhc_attn` and `mhc_ffn` in `float32`. Below we run `scripts/smoke_test.py` (or load saved `results/smoke.json`) to verify physical decoder-layer calls per generated token over 64 tokens and inspect the 5-problem sanity check.""",
        ),
        _code_cell(
            "smoke-code",
            """import json
from loop_dial.common import RESULTS_DIR

RUN_PHASE_A_LIVE = False  # Set True on Colab GPU to re-run Phase A from scratch
if RUN_PHASE_A_LIVE:
    subprocess.run([sys.executable, "scripts/smoke_test.py"], check=True)

smoke_data = json.loads((RESULTS_DIR / "smoke.json").read_text())
print(f"GPU: {smoke_data['gpu_name']} ({smoke_data['gpu_total_gb']} GB)")
print(f"Versions: {smoke_data['versions']}\\n")
print("64-token forward-hook decoder-layer verification:")
for row in smoke_data["scaling_64_tokens"]:
    print(
        f"  {row['condition']:10s} | plan_len={row['plan_len']:3d} | "
        f"calls/tok={row['decoder_layer_calls_per_generated_token']:5.1f} | "
        f"{row['ms_per_token']:7.2f} ms/tok ({row['tokens_per_s']:6.2f} tok/s) | "
        f"peak_mem={row['peak_gpu_mem_gb']:.2f} GB"
    )""",
            stdout_text=smoke_out,
        ),
        _md_cell(
            "vllm-setup-md",
            """## 4. Install `vllm==0.33.0` + SenseNova-Loop PR `#64832` (Isolated Venv)
Stock `vllm==0.33.0` does not yet ship `SenseNovaLoopForConditionalGeneration`. `scripts/probe_vllm.py --setup-only` creates an isolated `/content/vllm-venv` (leaving Colab's base `torch` and `transformers==5.18.0` untouched), installs `vllm==0.33.0`, applies upstream [`vllm-project/vllm#64832`](https://github.com/vllm-project/vllm/pull/64832), and verifies `hf_overrides["loop_settings"]` across `n = 1, 2, 3, 4` (`32 / 56 / 80 / 104` layer steps).

`ensure_vllm_python()` is also called automatically whenever you flip `RUN_PHASE_B_LIVE = True` or `RUN_AIME_LIVE = True` below.""",
        ),
        _code_cell(
            "vllm-setup-code",
            """RUN_VLLM_SETUP = False  # Set True to install/verify /content/vllm-venv now (also auto-runs if any RUN_*_LIVE=True)
VLLM_SETUP_VERIFIED = False

def ensure_vllm_python() -> str:
    global VLLM_SETUP_VERIFIED
    if not VLLM_SETUP_VERIFIED:
        subprocess.run([sys.executable, "scripts/probe_vllm.py", "--setup-only"], check=True)
        VLLM_SETUP_VERIFIED = True
    venv_py = Path("/content/vllm-venv/bin/python")
    return str(venv_py) if venv_py.exists() else sys.executable

if RUN_VLLM_SETUP:
    active_vllm_py = ensure_vllm_python()
    print("Ready vLLM Python:", active_vllm_py)
else:
    print("Skipping live vLLM install (offline replay mode). Flip RUN_VLLM_SETUP=True or any RUN_*_LIVE=True on Colab GPU.")""",
        ),
        _md_cell(
            "sweep-md",
            """## 5. Phase B — Main `MATH-500` Sweep (`N = 200`, `16,384` Max New Tokens)
`scripts/run_sweep.py` is fully resumable: it skips any `unique_id` already saved in `results/<condition>.jsonl`, evaluates nested prefix budgets (`2,048 / 4,096 / 8,192 / 16,384` tokens), and runs a 512-token Pass-2 forced answer (`</think>\\nFinal Answer: \\boxed{`) whenever a trace hits the cap without emitting a final boxed answer.""",
        ),
        _code_cell(
            "sweep-code",
            """RUN_PHASE_B_LIVE = False  # Set True on Colab GPU to run or resume the MATH-500 sweep
if RUN_PHASE_B_LIVE:
    vllm_py = ensure_vllm_python()
    subprocess.run([
        vllm_py, "scripts/run_sweep.py",
        "--backend", "vllm",
        "--n-problems", "200",
        "--max-new-tokens", "16384",
        "--conditions", "loop_n2", "loop_n1", "loop_n3", "loop_n4", "qwen35_9b",
    ], check=True)""",
        ),
        _md_cell(
            "stats-md",
            """## 6. Phase C — `MATH-500` Paired McNemar Tests, Wilson 95% CIs, & Charts
Because every model condition is evaluated on the **exact same problems**, we use two-sided **exact McNemar tests** on discordant pairs (`gained` vs `lost` problems) alongside 95% Wilson confidence intervals.""",
        ),
        _code_cell(
            "stats-code",
            """import pandas as pd

subprocess.run([sys.executable, "scripts/stats.py"], check=True)
summary_df = pd.read_csv("results/summary.csv")
paired_df = pd.read_csv("results/paired_stats.csv")

display_cols = [
    "label", "layer_executions_per_token", "n_problems",
    "accuracy_strict_pct", "accuracy_forced_pct",
    "ci95_strict_low_pct", "ci95_strict_high_pct",
    "hit_cap_pct", "loop_pct", "median_tokens", "correct_strict_per_gpu_hour"
]
print(summary_df[display_cols].to_string(index=False))
print("\\nHeadline Paired Exact McNemar Comparisons (strict metric):")
headline = paired_df[paired_df["metric"] == "strict"]
print(headline[["comparison", "diff_cmp_minus_ref_pp", "gained_by_cmp", "lost_by_cmp", "mcnemar_p_value"]].to_string(index=False))""",
        ),
        _md_cell("logged-results-md", summary_md),
        _md_cell(
            "charts-md",
            """### Plain-English `MATH-500` Charts & Real Side-by-Side Trace (`assets/`)
Every chart is programmatically audited to contain zero raw LaTeX tokens (`\\`, `$`, `{`, `}`, `\\boxed`, `</think>`) or internal config jargon.""",
        ),
        _code_cell(
            "charts-code",
            """from IPython.display import Image, display

subprocess.run([sys.executable, "scripts/generate_charts.py"], check=True)
for chart_name in [
    "hero_trace.png",
    "loop_architecture.png",
    "accuracy_vs_passes.png",
    "speed_vs_accuracy.png",
    "budget_curve.png",
    "paired_changes.png",
]:
    chart_path = REPO_ROOT / "assets" / chart_name
    if chart_path.exists():
        print(f"=== assets/{chart_name} ===")
        display(Image(filename=str(chart_path), width=900))""",
        ),
        _md_cell(
            "aime-md",
            """## 7. Phase B2 — Harder Follow-Up: `AIME 2024 + 2025` Olympiad Sweep (`N = 60`, `32,768` Tokens)
On `MATH-500`, default `2` passes hits a **`99.5%` (`199/200`) ceiling**, leaving only 1 unsolved problem out of 200 to test whether extra middle-layer passes (`n = 3, 4`) can rescue harder multi-step derivations once `2` passes has real headroom. Here we evaluate all 60 AIME 2024 + 2025 olympiad problems (`HuggingFaceH4/aime_2024` + `MathArena/aime_2025`) on the **identical `vllm==0.33.0` + PR `#64832` stack** at **`32,768` max new tokens** (`max_num_seqs=16`, nested `2k / 4k / 8k / 16k / 32k` budgets, 512-token Pass-2 forced answer). Results live cleanly under `results/aime/` without touching `MATH-500` files.""",
        ),
        _code_cell(
            "aime-live-code",
            """RUN_AIME_LIVE = False  # Set True on Colab GPU (>= 80 GB VRAM) to run or resume the 60-problem 32k AIME sweep
if RUN_AIME_LIVE:
    vllm_py = ensure_vllm_python()
    if not (REPO_ROOT / "data" / "aime_2024_2025.json").exists():
        subprocess.run([sys.executable, "scripts/prepare_aime.py"], check=True)
    subprocess.run([
        vllm_py, "scripts/run_sweep.py",
        "--backend", "vllm",
        "--problems-json", "data/aime_2024_2025.json",
        "--n-problems", "60",
        "--max-new-tokens", "32768",
        "--forced-max-tokens", "512",
        "--max-num-seqs", "16",
        "--out-dir", "results/aime",
        "--seed", "0",
        "--conditions", "loop_n2", "qwen35_9b", "loop_n4", "loop_n3", "loop_n1",
    ], check=True)""",
        ),
        _code_cell(
            "aime-stats-code",
            """if (REPO_ROOT / "results" / "aime" / "loop_n2.jsonl").exists():
    subprocess.run([sys.executable, "scripts/stats.py", "--results-dir", "results/aime"], check=True)
    subprocess.run([
        sys.executable, "scripts/generate_charts.py",
        "--results-dir", "results/aime",
        "--also-compare-math", "results/summary.csv",
    ], check=True)

    aime_summary_df = pd.read_csv("results/aime/summary.csv")
    aime_paired_df = pd.read_csv("results/aime/paired_stats.csv")
    print(aime_summary_df[display_cols].to_string(index=False))
    print("\\nAIME 2024+2025 Paired Exact McNemar Comparisons (strict metric):")
    aime_strict = aime_paired_df[aime_paired_df["metric"] == "strict"]
    print(aime_strict[["comparison", "diff_cmp_minus_ref_pp", "gained_by_cmp", "lost_by_cmp", "mcnemar_p_value"]].to_string(index=False))

    for chart_name in ["cover_5x2.png", "aime_vs_math500.png", "aime_budget_curve.png"]:
        chart_path = REPO_ROOT / "assets" / chart_name
        if chart_path.exists():
            print(f"=== assets/{chart_name} ===")
            display(Image(filename=str(chart_path), width=900))
else:
    print("No AIME JSONL files found in results/aime/ yet. Set RUN_AIME_LIVE = True on a Colab GPU.")""",
        ),
        _md_cell("logged-aime-results-md", aime_summary_md),
        _md_cell(
            "app-md",
            """## 8. Phase D — Launch the Side-by-Side Gradio Explorer (`app.py`)
Replays both **`MATH-500` (`N = 200`)** and **`AIME 2024+2025` (`N = 60`)** offline (zero GPU needed), or pass `--enable-live-gpu` on a Colab GPU runtime to also test custom math questions live.""",
        ),
        _code_cell(
            "app-code",
            """LAUNCH_GRADIO = False  # Set True to open the interactive side-by-side app
if LAUNCH_GRADIO:
    from app import ReplayStore, build_app, HEAD_META
    demo = build_app(store=ReplayStore(), enable_live_gpu=torch.cuda.is_available())
    demo.launch(head=HEAD_META, share=False)""",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "G4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote Colab notebook ({len(nb['cells'])} cells) -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
