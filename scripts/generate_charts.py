#!/usr/bin/env python3
r"""Generate plain-English, zero-raw-LaTeX publication charts in `assets/` from logged `results/` runs.

Charts produced:
1. `assets/loop_architecture.png`  — Visual diagram of `layers 0–3 -> 4–27 (×1..4) -> 28–31` + measured step cost.
2. `assets/accuracy_vs_passes.png` — Share of problems solved vs how many times the middle runs (+ Qwen3.5-9B line).
3. `assets/speed_vs_accuracy.png`  — Answers solved per GPU-hour and median seconds per problem vs accuracy.
4. `assets/budget_curve.png`       — Share of problems solved across nested token budgets (`2k / 4k / 8k / 16k`).
5. `assets/paired_changes.png`     — Paired problems gained vs lost relative to 2-pass default and Qwen3.5-9B.

Every chart is checked by `validate_figure_no_raw_markup(fig)` before saving to guarantee ZERO raw LaTeX
(`\`, `$`, `{`, `}`, `\boxed`, `\frac`) and ZERO code jargon (`eval_count`, `</think>`, `bf16`, `_`).
"""

from __future__ import annotations

import os

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from loop_dial.common import ASSETS_DIR, RESULTS_DIR

DPI = 160
FIG_W_IN = 10.0   # 1600 px exact
FIG_H_IN = 6.25   # 1000 px exact

PALETTE = {
    "loop_n1": "#64748b",    # slate
    "loop_n2": "#2563eb",    # royal blue (trained default)
    "loop_n3": "#7c3aed",    # violet
    "loop_n4": "#db2777",    # magenta-rose
    "qwen35_9b": "#059669",  # emerald green (non-looped sibling)
}

PLAIN_SHORT_LABELS = {
    "loop_n1": "1 pass\n(32 layer steps)",
    "loop_n2": "2 passes (default)\n(56 layer steps)",
    "loop_n3": "3 passes\n(80 layer steps)",
    "loop_n4": "4 passes\n(104 layer steps)",
    "qwen35_9b": "Plain Qwen3.5-9B\n(32 layer steps)",
}

PLAIN_LEGEND_LABELS = {
    "loop_n1": "SenseNova-Loop: 1 pass (32 steps)",
    "loop_n2": "SenseNova-Loop: 2 passes (56 steps, default)",
    "loop_n3": "SenseNova-Loop: 3 passes (80 steps)",
    "loop_n4": "SenseNova-Loop: 4 passes (104 steps)",
    "qwen35_9b": "Plain Qwen3.5-9B (32 steps, no loop)",
}

FORBIDDEN_TOKENS = ("\\", "$", "{", "}", "**", "boxed", "frac", "eval_count", "</think>", "bf16", "bfloat16")


def _apply_base_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )


def validate_figure_no_raw_markup(fig: Any) -> List[str]:
    """Draw `fig` and assert zero raw LaTeX tokens, underscores, or code variable names appear."""
    rendered: List[str] = []
    orig_draw_text = RendererBase.draw_text

    def _spy(self: Any, gc: Any, x: Any, y: Any, s: str, prop: Any, angle: Any, ismath: Any = False, mtext: Any = None) -> Any:
        rendered.append(str(s))
        return orig_draw_text(self, gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)

    RendererBase.draw_text = _spy  # type: ignore[method-assign]
    try:
        fig.canvas.draw()
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text():
                rendered.append(t.get_text())
    finally:
        RendererBase.draw_text = orig_draw_text  # type: ignore[method-assign]

    for s in rendered:
        low = s.lower()
        for tok in FORBIDDEN_TOKENS:
            if tok.lower() in low:
                raise AssertionError(f"Chart text contains forbidden token {tok!r}: {s!r}")
        if "_" in s:
            raise AssertionError(f"Chart text contains raw underscore '_': {s!r}")
    return rendered


def _save_checked(fig: Any, out_path: Path) -> None:
    validate_figure_no_raw_markup(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved validated chart -> {out_path}")


def generate_architecture_diagram(smoke_json_path: Path, out_path: Path) -> None:
    """Render plain-English diagram of `layers 0–3 -> 4–27 (×n) -> 28–31` with measured step scaling."""
    _apply_base_rc()
    smoke_data: Dict[str, Any] = {}
    if smoke_json_path.is_file():
        smoke_data = json.loads(smoke_json_path.read_text(encoding="utf-8"))

    ms_by_cond: Dict[str, float] = {}
    calls_by_cond: Dict[str, float] = {}
    for row in smoke_data.get("scaling_64_tokens", []):
        c = str(row.get("condition", ""))
        ms_by_cond[c] = float(row.get("ms_per_token", 0.0))
        calls_by_cond[c] = float(row.get("decoder_layer_calls_per_generated_token", 0.0))

    vllm_probe_path = smoke_json_path.parent / "vllm_probe.json"
    if vllm_probe_path.is_file():
        vp = json.loads(vllm_probe_path.read_text(encoding="utf-8"))
        q_ms = vp.get("stock_vllm_0_33_0_qwen35_9b", {}).get("ms_per_token")
        if q_ms:
            ms_by_cond["qwen35_9b"] = float(q_ms)
        per_pass = vp.get("pr_64832_overlay", {}).get("per_pass_verification", {})
        for k_str, p_info in per_pass.items():
            if isinstance(p_info, dict) and p_info.get("ok") and p_info.get("ms_per_token"):
                ms_by_cond[f"loop_n{k_str}"] = float(p_info["ms_per_token"])

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62.5)
    ax.axis("off")

    # Title & subtitle
    ax.text(
        5,
        57.5,
        "How the SenseNova-Loop \"Loop Dial\" Works",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        5,
        54.0,
        "Instead of adding new weights, the model re-runs its middle 24 layers multiple times per word/token",
        fontsize=11.2,
        color="#475569",
    )

    # Top flow diagram (y = 32..48)
    def draw_box(x: float, y: float, w: float, h: float, title: str, sub1: str, sub2: str, bg: str, border: str) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.4",
            facecolor=bg,
            edgecolor=border,
            linewidth=2.0,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.72, title, fontsize=11.5, fontweight="bold", ha="center", va="center", color="#0f172a")
        ax.text(x + w / 2, y + h * 0.45, sub1, fontsize=9.6, ha="center", va="center", color="#334155")
        ax.text(x + w / 2, y + h * 0.20, sub2, fontsize=9.0, ha="center", va="center", color="#64748b")

    # Input token pill
    draw_box(4.5, 34, 13, 12, "Input Token", "Enters model", "1 stream", "#f8fafc", "#94a3b8")
    # Entry layers 0-3
    draw_box(21.0, 34, 17, 12, "Entry Layers 0–3", "4 layers · run once", "Single stream", "#eff6ff", "#60a5fa")
    # Middle looped block 4-27
    draw_box(
        41.8,
        33,
        29.4,
        14,
        "Middle Layers 4–27 (×Dial)",
        "24 shared layers · 2 parallel streams",
        "Separate cache per pass (1×, 2×, 3×, 4×)",
        "#dbeafe",
        "#2563eb",
    )
    # Exit layers 28-31
    draw_box(75.0, 34, 20.0, 12, "Exit Layers 28–31", "4 layers · run once", "Merged → next token", "#eff6ff", "#60a5fa")

    # Forward arrows
    for x0, x1 in ((18.1, 20.4), (38.6, 41.1), (71.9, 74.3)):
        arr = FancyArrowPatch((x0, 40), (x1, 40), arrowstyle="-|>,head_length=6,head_width=4", linewidth=2.0, color="#475569")
        ax.add_patch(arr)

    # Loopback arc over Middle Layers 4-27
    loop_arc = FancyArrowPatch(
        (64.5, 47.4),
        (48.5, 47.4),
        connectionstyle="arc3,rad=0.30",
        arrowstyle="-|>,head_length=7,head_width=4.5",
        linewidth=2.3,
        color="#1d4ed8",
    )
    ax.add_patch(loop_arc)
    ax.text(
        56.5,
        51.6,
        "Loop Dial repeats layers 4–27 (default = 2 passes)",
        fontsize=9.8,
        fontweight="bold",
        ha="center",
        va="center",
        color="#1d4ed8",
    )

    # Bottom table/bars comparing layer executions per token (y = 6..27)
    ax.text(
        4.5,
        26.8,
        "Verified Layer Executions & Single-Stream GPU Latency per Token",
        fontsize=12.0,
        fontweight="bold",
        color="#0f172a",
    )

    dial_specs = [
        ("qwen35_9b", "Plain Qwen3.5-9B (no loop)", 32, PALETTE["qwen35_9b"]),
        ("loop_n1", "Loop Dial = 1 pass", 32, PALETTE["loop_n1"]),
        ("loop_n2", "Loop Dial = 2 passes (default)", 56, PALETTE["loop_n2"]),
        ("loop_n3", "Loop Dial = 3 passes", 80, PALETTE["loop_n3"]),
        ("loop_n4", "Loop Dial = 4 passes", 104, PALETTE["loop_n4"]),
    ]

    base_y = 22.2
    row_step = 4.1
    bar_x = 33.5
    max_bar_w = 33.0
    for idx, (cond_key, label_txt, expected_steps, col) in enumerate(dial_specs):
        y = base_y - idx * row_step
        measured_steps = int(round(calls_by_cond.get(cond_key, expected_steps)))
        bar_w = (measured_steps / 104.0) * max_bar_w
        ax.text(4.5, y, label_txt, fontsize=10.2, fontweight="bold" if cond_key == "loop_n2" else "normal", va="center", color="#1e293b")
        rect = FancyBboxPatch(
            (bar_x, y - 1.15),
            bar_w,
            2.3,
            boxstyle="round,pad=0.05,rounding_size=0.5",
            facecolor=col,
            edgecolor="none",
            alpha=0.90,
        )
        ax.add_patch(rect)
        ms_val = ms_by_cond.get(cond_key, 0.0)
        suffix = f"{measured_steps} layer steps/token"
        if ms_val > 0:
            suffix += f"  ·  {ms_val:.1f} ms/token"
        ax.text(bar_x + bar_w + 1.2, y, suffix, fontsize=9.6, va="center", color="#334155")

    gpu_label = smoke_data.get("gpu_name", "NVIDIA RTX PRO 6000")
    ax.text(
        4.5,
        2.0,
        f"All 4 Loop settings use the exact same 9.66B parameter checkpoint · Hardware: {gpu_label}",
        fontsize=8.8,
        color="#64748b",
    )
    _save_checked(fig, out_path)


def generate_accuracy_vs_passes_chart(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    out_path: Path,
    *,
    gpu_footer: str,
) -> None:
    """Plot MATH-500 accuracy across 1, 2, 3, 4 passes with Qwen3.5-9B horizontal reference line."""
    _apply_base_rc()
    loop_df = summary_df[summary_df["condition"].isin(["loop_n1", "loop_n2", "loop_n3", "loop_n4"])].copy()
    loop_df = loop_df.sort_values("n_passes").reset_index(drop=True)
    qwen_df = summary_df[summary_df["condition"] == "qwen35_9b"]

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.18)

    xs = np.arange(len(loop_df))
    strict_vals = loop_df["accuracy_strict_pct"].to_numpy(dtype=float)
    forced_vals = loop_df["accuracy_forced_pct"].to_numpy(dtype=float)
    err_low = strict_vals - loop_df["ci95_strict_low_pct"].to_numpy(dtype=float)
    err_high = loop_df["ci95_strict_high_pct"].to_numpy(dtype=float) - strict_vals

    bar_w = 0.36
    bars_s = ax.bar(
        xs - bar_w / 2,
        strict_vals,
        width=bar_w,
        color=[PALETTE[c] for c in loop_df["condition"]],
        edgecolor="none",
        zorder=3,
        label="Finished within token limit",
    )
    bars_f = ax.bar(
        xs + bar_w / 2,
        forced_vals,
        width=bar_w,
        color=[PALETTE[c] for c in loop_df["condition"]],
        alpha=0.42,
        edgecolor=[PALETTE[c] for c in loop_df["condition"]],
        linewidth=1.5,
        zorder=3,
        label="Including forced final answer at token cap",
    )
    ax.errorbar(
        xs - bar_w / 2,
        strict_vals,
        yerr=[np.maximum(0, err_low), np.maximum(0, err_high)],
        fmt="none",
        ecolor="#0f172a",
        elinewidth=1.4,
        capsize=4,
        zorder=4,
    )

    if not qwen_df.empty:
        q_strict = float(qwen_df.iloc[0]["accuracy_strict_pct"])
        q_forced = float(qwen_df.iloc[0]["accuracy_forced_pct"])
        q_lo = float(qwen_df.iloc[0]["ci95_strict_low_pct"])
        q_hi = float(qwen_df.iloc[0]["ci95_strict_high_pct"])
        ax.axhspan(q_lo, q_hi, color=PALETTE["qwen35_9b"], alpha=0.10, zorder=1)
        ax.axhline(
            q_strict,
            color=PALETTE["qwen35_9b"],
            linestyle="--",
            linewidth=2.3,
            zorder=4,
            label=f"Plain Qwen3.5-9B (no loop): {q_strict:.1f}% (with cap prompt: {q_forced:.1f}%)",
        )

    ci_high_vals = loop_df["ci95_strict_high_pct"].to_numpy(dtype=float)
    for b_s, b_f, val_s, val_f, ci_hi in zip(bars_s, bars_f, strict_vals, forced_vals, ci_high_vals):
        ax.text(
            b_s.get_x() + b_s.get_width() / 2,
            max(val_s, ci_hi) + 1.5,
            f"{val_s:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
            color="#0f172a",
        )
        if abs(val_f - val_s) >= 0.4:
            ax.text(
                b_f.get_x() + b_f.get_width() / 2,
                val_f + 1.5,
                f"{val_f:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#475569",
            )

    n_probs = int(loop_df["n_problems"].iloc[0])
    cap_toks = int(loop_df["max_new_tokens"].iloc[0])
    ax.set_xticks(xs)
    ax.set_xticklabels([PLAIN_SHORT_LABELS[c] for c in loop_df["condition"]], fontsize=10.5)
    ax.set_ylabel("Share of math problems solved (%)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("How many times the middle 24 layers run per token", fontsize=11.5, fontweight="bold", labelpad=8)

    y_top = max(109.0, float(summary_df[["accuracy_strict_pct", "accuracy_forced_pct", "ci95_strict_high_pct"]].max().max()) + 9.0)
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", color="#e2e8f0", linestyle="-", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    fig.text(
        0.06,
        0.92,
        "Turning the Loop Dial: Math Accuracy Across 1, 2, 3, and 4 Passes",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.06,
        0.87,
        f"Evaluated on {n_probs} fixed MATH-500 problems (up to {cap_toks:,} thinking tokens per problem, 95% confidence bars)",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(0.06, 0.03, gpu_footer, fontsize=8.8, color="#64748b")
    _save_checked(fig, out_path)


def generate_speed_vs_accuracy_chart(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    gpu_footer: str,
) -> None:
    """Two-panel plain-English chart: (Left) Correct answers per GPU-hour, (Right) Median seconds vs Accuracy."""
    _apply_base_rc()
    order = [c for c in ["qwen35_9b", "loop_n1", "loop_n2", "loop_n3", "loop_n4"] if c in set(summary_df["condition"])]
    df = summary_df.set_index("condition").loc[order].reset_index()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.subplots_adjust(left=0.175, right=0.965, top=0.79, bottom=0.18, wspace=0.28)

    # Left panel: Correct answers per GPU-hour
    ys = np.arange(len(df))[::-1]
    ans_per_hr = df["correct_strict_per_gpu_hour"].to_numpy(dtype=float)
    colors = [PALETTE[c] for c in df["condition"]]
    short_names = [
        "Plain Qwen3.5-9B" if c == "qwen35_9b" else f"Loop: {int(r['n_passes'])} {'pass' if int(r['n_passes']) == 1 else 'passes'}"
        for c, (_, r) in zip(df["condition"], df.iterrows())
    ]

    bars = ax_l.barh(ys, ans_per_hr, height=0.56, color=colors, zorder=3)
    ax_l.set_yticks(ys)
    ax_l.set_yticklabels(short_names, fontsize=10.0, fontweight="bold")
    ax_l.set_xlabel("Correct answers solved per GPU-hour", fontsize=10.5, fontweight="bold")
    ax_l.set_title("Throughput: Answers per Hour", fontsize=12.0, fontweight="bold", pad=10)
    ax_l.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    max_hr = max(float(ans_per_hr.max()), 1.0)
    ax_l.set_xlim(0, max_hr * 1.28)
    for bar, val in zip(bars, ans_per_hr):
        ax_l.text(
            val + max_hr * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}/hr" if val >= 20 else f"{val:.1f}/hr",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="#0f172a",
        )

    # Right panel: Median GPU seconds per problem vs Accuracy
    med_s = df["median_gpu_seconds_per_problem"].to_numpy(dtype=float)
    acc = df["accuracy_strict_pct"].to_numpy(dtype=float)
    custom_offsets = {
        "loop_n2": (0.38, 1.0),
        "loop_n3": (0.38, 0.8),
        "loop_n4": (0.38, 0.8),
        "qwen35_9b": (0.38, -4.2),
        "loop_n1": (-0.45, 2.2),
    }
    for idx, row in df.iterrows():
        c = str(row["condition"])
        x_val = float(row["median_gpu_seconds_per_problem"])
        y_val = float(row["accuracy_strict_pct"])
        marker = "D" if c == "qwen35_9b" else "o"
        ax_r.scatter(
            [x_val],
            [y_val],
            s=145,
            color=PALETTE[c],
            marker=marker,
            edgecolors="#ffffff",
            linewidth=1.5,
            zorder=5,
        )
        lbl = "Plain Qwen3.5-9B" if c == "qwen35_9b" else f"{int(row['n_passes'])} {'pass' if int(row['n_passes']) == 1 else 'passes'}"
        dx, dy = custom_offsets.get(c, (0.35, 1.2))
        ha = "right" if dx < 0 else "left"
        ax_r.text(x_val + dx, y_val + dy, f"{lbl} ({y_val:.1f}%)", fontsize=9.1, fontweight="bold", ha=ha, color=PALETTE[c], zorder=6)

    # Connect loop_n1..4 in order
    loop_sub = df[df["condition"].str.startswith("loop_n")].sort_values("n_passes")
    if len(loop_sub) >= 2:
        ax_r.plot(
            loop_sub["median_gpu_seconds_per_problem"],
            loop_sub["accuracy_strict_pct"],
            color="#94a3b8",
            linestyle=":",
            linewidth=1.6,
            zorder=3,
        )

    ax_r.set_xlabel("Median GPU seconds per problem (lower is faster)", fontsize=10.5, fontweight="bold")
    ax_r.set_ylabel("Share of problems solved (%)", fontsize=10.5, fontweight="bold")
    ax_r.set_title("Speed vs Accuracy Trade-off", fontsize=12.0, fontweight="bold", pad=10)
    ax_r.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    x_hi = max(float(med_s.max()) * 1.48, 1.0)
    ax_r.set_xlim(0, x_hi)
    y_lo = max(-4.0, float(acc.min()) - 6.0)
    y_hi = min(107.0, float(acc.max()) + 8.0)
    ax_r.set_ylim(y_lo, y_hi)

    n_probs = int(df["n_problems"].iloc[0])
    backend_name = "Hugging Face Transformers" if df["backend"].iloc[0] == "hf" else "vLLM continuous batching"
    fig.text(
        0.06,
        0.92,
        "Does Extra Looping Pay for Its Compute Cost?",
        fontsize=15.5,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.06,
        0.87,
        f"Measured on the same {backend_name} backend across {n_probs} MATH-500 problems",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(0.06, 0.03, gpu_footer, fontsize=8.8, color="#64748b")
    _save_checked(fig, out_path)


def generate_budget_curve_chart(
    budget_df: pd.DataFrame,
    out_path: Path,
    *,
    gpu_footer: str,
) -> None:
    """Plot nested token-budget curve (`2k / 4k / 8k / 16k [/ 32k]`) across all conditions."""
    _apply_base_rc()
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.17)

    budgets = sorted(budget_df["budget"].unique().tolist())
    x_pos = np.arange(len(budgets))

    for cond in ["qwen35_9b", "loop_n1", "loop_n2", "loop_n3", "loop_n4"]:
        sub = budget_df[budget_df["condition"] == cond].sort_values("budget")
        if sub.empty:
            continue
        b_to_acc = dict(zip(sub["budget"].astype(int), sub["accuracy_strict_pct"].astype(float)))
        ys = [b_to_acc.get(b, np.nan) for b in budgets]
        ls = "--" if cond == "qwen35_9b" else "-"
        lw = 2.8 if cond in ("loop_n2", "qwen35_9b") else 2.0
        marker = "D" if cond == "qwen35_9b" else "o"
        ax.plot(
            x_pos,
            ys,
            linestyle=ls,
            linewidth=lw,
            marker=marker,
            markersize=7.5,
            color=PALETTE[cond],
            label=PLAIN_LEGEND_LABELS.get(cond, cond),
            zorder=4 if cond in ("loop_n2", "qwen35_9b") else 3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b:,} tokens" for b in budgets], fontsize=10.5)
    ax.set_xlabel("Maximum thinking tokens allowed per problem", fontsize=11.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Share of math problems solved (%)", fontsize=11.5, fontweight="bold")
    ax.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = min(105.0, float(budget_df["accuracy_strict_pct"].max()) + 10.0)
    ax.set_ylim(-2.0, max(y_max, 50.0))
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.4)

    n_probs = int(budget_df["n_problems"].max())
    fig.text(
        0.06,
        0.92,
        "Accuracy Across Thinking Token Budgets (Nested Prefix Curve)",
        fontsize=15.5,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.06,
        0.87,
        f"Share of {n_probs} MATH-500 problems solved cleanly before hitting each token cap",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(0.06, 0.03, gpu_footer, fontsize=8.8, color="#64748b")
    _save_checked(fig, out_path)


def generate_paired_changes_chart(
    paired_df: pd.DataFrame,
    out_path: Path,
    *,
    gpu_footer: str,
) -> None:
    """Plot paired problems gained vs lost (+ exact McNemar p-values) for key comparisons."""
    _apply_base_rc()
    strict_p = paired_df[paired_df["metric"] == "strict"].copy()
    target_order = [
        ("loop_n1_vs_loop_n2", "1 pass vs 2 passes (default)"),
        ("loop_n3_vs_loop_n2", "3 passes vs 2 passes (default)"),
        ("loop_n4_vs_loop_n2", "4 passes vs 2 passes (default)"),
        ("loop_n2_vs_qwen35_9b", "2-pass Loop vs Plain Qwen3.5-9B"),
    ]
    rows_to_plot = []
    for key, label in target_order:
        m = strict_p[strict_p["comparison"] == key]
        if not m.empty:
            r = m.iloc[0].to_dict()
            r["clean_label"] = label
            rows_to_plot.append(r)

    if not rows_to_plot:
        return

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.subplots_adjust(left=0.31, right=0.95, top=0.77, bottom=0.17)

    ys = np.arange(len(rows_to_plot))[::-1]
    gained = np.array([int(r["gained_by_cmp"]) for r in rows_to_plot], dtype=float)
    lost = np.array([-int(r["lost_by_cmp"]) for r in rows_to_plot], dtype=float)

    ax.barh(ys, gained, height=0.46, color="#059669", label="Problems newly solved (+gained)", zorder=3)
    ax.barh(ys, lost, height=0.46, color="#dc2626", label="Previously solved problems broken (-lost)", zorder=3)
    ax.axvline(0, color="#0f172a", linewidth=1.2, zorder=4)

    span = max(float(np.max(np.abs(np.concatenate([gained, lost])))), 4.0)
    ax.set_xlim(-span * 1.42, span * 1.48)
    ax.set_ylim(-0.6, len(rows_to_plot) - 0.35)

    for y, r in zip(ys, rows_to_plot):
        g = int(r["gained_by_cmp"])
        l = int(r["lost_by_cmp"])
        pval = float(r["mcnemar_p_value"])
        p_txt = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
        ax.text(g + span * 0.04, y, f"+{g} solved", va="center", ha="left", fontsize=9.6, fontweight="bold", color="#065f46")
        ax.text(-l - span * 0.04, y, f"-{l} lost", va="center", ha="right", fontsize=9.6, fontweight="bold", color="#991b1b")
        ax.text(span * 1.44, y, f"Net {r['diff_cmp_minus_ref_pp']:+.1f} pts ({p_txt})", va="center", ha="right", fontsize=9.1, color="#334155")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["clean_label"] for r in rows_to_plot], fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Number of paired MATH-500 problems gained (+) or lost (-)", fontsize=10.8, fontweight="bold")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        bbox_to_anchor=(0.5, 1.11),
        loc="upper center",
        ncol=2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=9.3,
    )

    n_p = int(rows_to_plot[0]["n_paired"])
    fig.text(
        0.06,
        0.92,
        "Paired Head-to-Head Problem Changes (Exact McNemar Test)",
        fontsize=15.5,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.06,
        0.87,
        f"Which exact problems flip between right and wrong on the same {n_p}-problem cohort",
        fontsize=10.5,
        color="#475569",
    )
    fig.text(0.06, 0.03, gpu_footer, fontsize=8.8, color="#64748b")
    _save_checked(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plain-English validated charts in assets/.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--prefix", type=str, default="", help="Optional CSV prefix inside results/")
    args = parser.parse_args()

    smoke_path = args.results_dir / "smoke.json"
    gpu_name = "Colab NVIDIA RTX PRO 6000 GPU"
    if smoke_path.is_file():
        smoke_obj = json.loads(smoke_path.read_text(encoding="utf-8"))
        gpu_name = smoke_obj.get("gpu_name", gpu_name)

    generate_architecture_diagram(smoke_path, args.assets_dir / "loop_architecture.png")

    summary_csv = args.results_dir / f"{args.prefix}summary.csv"
    budget_csv = args.results_dir / f"{args.prefix}budget_curve.csv"
    paired_csv = args.results_dir / f"{args.prefix}paired_stats.csv"

    if summary_csv.is_file():
        summary_df = pd.read_csv(summary_csv)
        paired_df = pd.read_csv(paired_csv) if paired_csv.is_file() else pd.DataFrame()
        footer = f"Measured on {gpu_name} · Fixed seed = 0 on HuggingFaceH4/MATH-500"

        generate_accuracy_vs_passes_chart(
            summary_df,
            paired_df,
            args.assets_dir / "accuracy_vs_passes.png",
            gpu_footer=footer,
        )
        generate_speed_vs_accuracy_chart(
            summary_df,
            args.assets_dir / "speed_vs_accuracy.png",
            gpu_footer=footer,
        )
        if budget_csv.is_file():
            budget_df = pd.read_csv(budget_csv)
            generate_budget_curve_chart(
                budget_df,
                args.assets_dir / "budget_curve.png",
                gpu_footer=footer,
            )
        if not paired_df.empty:
            generate_paired_changes_chart(
                paired_df,
                args.assets_dir / "paired_changes.png",
                gpu_footer=footer,
            )


if __name__ == "__main__":
    main()
