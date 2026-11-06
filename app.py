#!/usr/bin/env python3
"""Gradio side-by-side explorer for `loop-dial` (offline replay on Mac + optional live GPU mode on Colab).

Features:
- Middle-layer passes slider (`1` to `4`) + `"Plain Qwen3.5-9B"` comparison toggle (plus secondary pass slider).
- Problem selector across the logged `MATH-500` subset with 1-click disagreement filters.
- Side-by-side saved outputs: clean plain-English math answer, `✓ Correct` / `✗ Wrong`, tokens, GPU seconds,
  loop/cap status, and native `<details>`/`<summary>` collapsible step-by-step reasoning.
- Strictly zero raw LaTeX (`\\`, `$`, `{`, `}`) in rendered cards and strictly server-escaped HTML (`html.escape`).
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr

from loop_dial.common import DATA_DIR, RESULTS_DIR, read_jsonl_by_id
from loop_dial.metrics import (
    DEFAULT_SAMPLING,
    QWEN35_MODEL_ID,
    SENSENOVA_LOOP_MODEL_ID,
    build_math_prompt,
    extract_last_boxed,
    extract_last_boxed_strict,
    format_math_prose_plain,
    grade_math_answer,
    latex_to_plain_text,
    split_thinking_and_response,
)

CONDITION_META: Dict[str, Dict[str, Any]] = {
    "loop_n1": {
        "title": "1 middle pass (half default)",
        "subtitle": "SenseNova-Loop-9B · 32 layer steps per token",
        "layers_per_token": 32,
    },
    "loop_n2": {
        "title": "2 middle passes (model default)",
        "subtitle": "SenseNova-Loop-9B · 56 layer steps per token",
        "layers_per_token": 56,
    },
    "loop_n3": {
        "title": "3 middle passes (1.5× default)",
        "subtitle": "SenseNova-Loop-9B · 80 layer steps per token",
        "layers_per_token": 80,
    },
    "loop_n4": {
        "title": "4 middle passes (2× default)",
        "subtitle": "SenseNova-Loop-9B · 104 layer steps per token",
        "layers_per_token": 104,
    },
    "qwen35_9b": {
        "title": "Plain Qwen3.5-9B (no loop)",
        "subtitle": "Standard 32 layers run once per token",
        "layers_per_token": 32,
    },
}

FILTER_CHOICES: List[str] = [
    "All logged problems",
    "Loop passes disagree (1–4)",
    "2 passes beats Plain Qwen3.5-9B",
    "Plain Qwen3.5-9B beats 2 passes",
    "Hit token cap or repetition loop",
]

APP_CSS = """
@layer tokens, layout, cards, disclosure;

@layer tokens {
  :root {
    --ld-bg-light: #f8fafc;
    --ld-bg-dark: #0f172a;
    --ld-surface-light: #ffffff;
    --ld-surface-dark: #1e293b;
    --ld-border-light: #cbd5e1;
    --ld-border-dark: #334155;
    --ld-text-light: #0f172a;
    --ld-text-dark: #f1f5f9;
    --ld-muted-light: #475569;
    --ld-muted-dark: #94a3b8;
    --ld-ok-bg-light: #dcfce7;
    --ld-ok-bg-dark: #064e3b;
    --ld-ok-fg-light: #166534;
    --ld-ok-fg-dark: #bbf7d0;
    --ld-bad-bg-light: #fee2e2;
    --ld-bad-bg-dark: #7f1d1d;
    --ld-bad-fg-light: #991b1b;
    --ld-bad-fg-dark: #fecaca;
    --ld-accent-light: #1d4ed8;
    --ld-accent-dark: #60a5fa;

    --ld-bg: var(--ld-bg-light);
    --ld-surface: var(--ld-surface-light);
    --ld-border: var(--ld-border-light);
    --ld-text: var(--ld-text-light);
    --ld-muted: var(--ld-muted-light);
    --ld-ok-bg: var(--ld-ok-bg-light);
    --ld-ok-fg: var(--ld-ok-fg-light);
    --ld-bad-bg: var(--ld-bad-bg-light);
    --ld-bad-fg: var(--ld-bad-fg-light);
    --ld-accent: var(--ld-accent-light);

    color-scheme: light dark;
    accent-color: var(--ld-accent);
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --ld-bg: var(--ld-bg-dark);
      --ld-surface: var(--ld-surface-dark);
      --ld-border: var(--ld-border-dark);
      --ld-text: var(--ld-text-dark);
      --ld-muted: var(--ld-muted-dark);
      --ld-ok-bg: var(--ld-ok-bg-dark);
      --ld-ok-fg: var(--ld-ok-fg-dark);
      --ld-bad-bg: var(--ld-bad-bg-dark);
      --ld-bad-fg: var(--ld-bad-fg-dark);
      --ld-accent: var(--ld-accent-dark);
    }
  }

  @supports (color: light-dark(white, black)) {
    :root {
      --ld-bg: light-dark(var(--ld-bg-light), var(--ld-bg-dark));
      --ld-surface: light-dark(var(--ld-surface-light), var(--ld-surface-dark));
      --ld-border: light-dark(var(--ld-border-light), var(--ld-border-dark));
      --ld-text: light-dark(var(--ld-text-light), var(--ld-text-dark));
      --ld-muted: light-dark(var(--ld-muted-light), var(--ld-muted-dark));
      --ld-ok-bg: light-dark(var(--ld-ok-bg-light), var(--ld-ok-bg-dark));
      --ld-ok-fg: light-dark(var(--ld-ok-fg-light), var(--ld-ok-fg-dark));
      --ld-bad-bg: light-dark(var(--ld-bad-bg-light), var(--ld-bad-bg-dark));
      --ld-bad-fg: light-dark(var(--ld-bad-fg-light), var(--ld-bad-fg-dark));
      --ld-accent: light-dark(var(--ld-accent-light), var(--ld-accent-dark));
    }
  }
}

@layer cards {
  .ld-banner {
    background: var(--ld-surface);
    border: 1px solid var(--ld-border);
    border-radius: 12px;
    padding: 14px 18px;
    color: var(--ld-text);
  }

  .ld-score-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-block-start: 10px;
  }

  .ld-score-chip {
    background: var(--ld-bg);
    border: 1px solid var(--ld-border);
    border-radius: 8px;
    padding: 6px 11px;
    font-size: 0.9rem;
    color: var(--ld-text);
  }

  .ld-problem-card {
    background: var(--ld-surface);
    border: 1px solid var(--ld-border);
    border-radius: 12px;
    padding: 16px 18px;
    color: var(--ld-text);
    min-width: 0;
    overflow: hidden;
  }

  .ld-problem-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    font-size: 0.88rem;
    color: var(--ld-muted);
    margin-block-end: 8px;
  }

  .ld-problem-text {
    font-size: 1.02rem;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: var(--ld-text);
  }

  .ld-output-card {
    background: var(--ld-surface);
    border: 1px solid var(--ld-border);
    border-radius: 12px;
    padding: 16px 18px;
    color: var(--ld-text);
    min-width: 0;
    overflow: hidden;
  }

  .ld-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-block-end: 12px;
  }

  .ld-card-title {
    font-size: 1.08rem;
    font-weight: 700;
    margin: 0;
    color: var(--ld-text);
  }

  .ld-card-sub {
    font-size: 0.85rem;
    color: var(--ld-muted);
    margin-block-start: 2px;
  }

  .ld-verdict {
    font-weight: 700;
    font-size: 0.92rem;
    padding: 5px 12px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .ld-verdict-ok {
    background: var(--ld-ok-bg);
    color: var(--ld-ok-fg);
  }

  .ld-verdict-bad {
    background: var(--ld-bad-bg);
    color: var(--ld-bad-fg);
  }

  .ld-answer-box {
    background: var(--ld-bg);
    border: 1px solid var(--ld-border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-block-end: 12px;
  }

  .ld-answer-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ld-muted);
  }

  .ld-answer-value {
    font-size: 1.18rem;
    font-weight: 700;
    margin-block-start: 3px;
    color: var(--ld-text);
    word-break: break-word;
    overflow-wrap: anywhere;
  }

  .ld-metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-block-end: 12px;
  }

  .ld-metric-pill {
    font-size: 0.85rem;
    padding: 4px 10px;
    border-radius: 8px;
    background: var(--ld-bg);
    border: 1px solid var(--ld-border);
    color: var(--ld-text);
  }

  .ld-metric-pill:not(:last-child) {
    margin-inline-end: 2px;
  }

  .ld-final-summary {
    font-size: 0.93rem;
    line-height: 1.5;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: var(--ld-text);
    margin-block-end: 12px;
    max-height: 150px;
    overflow-y: auto;
    padding: 8px 10px;
    border-radius: 8px;
    background: var(--ld-bg);
  }
}

@layer disclosure {
  details.ld-details {
    border: 1px solid var(--ld-border);
    border-radius: 10px;
    background: var(--ld-bg);
    padding: 8px 12px;
  }

  details.ld-details > summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--ld-accent);
    padding-block: 4px;
  }

  details.ld-details[open] > summary {
    margin-block-end: 8px;
    border-block-end: 1px solid var(--ld-border);
    padding-block-end: 8px;
  }

  .ld-trace-pre {
    margin: 0;
    max-height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    font-size: 0.84rem;
    line-height: 1.48;
    color: var(--ld-text);
  }
}
"""

HEAD_META = f'<meta name="color-scheme" content="light dark">\n<style>\n{APP_CSS}\n</style>'


def _esc(val: Any) -> str:
    """Strictly HTML-escape any dynamic value before rendering."""
    return html.escape(str(val if val is not None else ""), quote=True)


def clamp_passes(value: Any, default: int = 2) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        n = default
    return max(1, min(4, n))


class ReplayStore:
    """Loads subset problems and all saved condition JSONLs from `results/` (falling back to smoke runs if needed)."""

    def __init__(self, data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.problems: List[Dict[str, Any]] = []
        self.problem_by_id: Dict[str, Dict[str, Any]] = {}
        self.records: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in CONDITION_META}
        self.reload()

    def reload(self) -> None:
        subset_path = self.data_dir / "subset_200.json"
        if subset_path.exists():
            payload = json.loads(subset_path.read_text(encoding="utf-8"))
            self.problems = payload.get("problems", [])
        else:
            self.problems = []
        self.problem_by_id = {p["unique_id"]: p for p in self.problems}

        self.records = {k: {} for k in CONDITION_META}
        # First load smoke_5problems.jsonl as a baseline fallback if main JSONLs aren't complete yet
        smoke_jsonl = self.results_dir / "smoke_5problems.jsonl"
        if smoke_jsonl.exists():
            for line in smoke_jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cond = row.get("condition")
                uid = row.get("unique_id")
                if cond in self.records and isinstance(uid, str):
                    self.records[cond][uid] = self._normalize_row(row, source="smoke_4k")

        # Override with full main-sweep JSONLs (`results/<cond>.jsonl`)
        for cond in CONDITION_META:
            cond_file = self.results_dir / f"{cond}.jsonl"
            if cond_file.exists():
                by_id = read_jsonl_by_id(cond_file)
                for uid, row in by_id.items():
                    self.records[cond][uid] = self._normalize_row(row, source=str(row.get("backend", "sweep")))

    @staticmethod
    def _normalize_row(row: Dict[str, Any], source: str) -> Dict[str, Any]:
        if "correct_forced" in row:
            correct = bool(row.get("correct_forced"))
        elif "correct_final" in row:
            correct = bool(row.get("correct_final"))
        else:
            correct = bool(row.get("correct_strict", False))

        if "pred_forced" in row:
            pred = row.get("pred_forced")
        elif "pred_final" in row:
            pred = row.get("pred_final")
        else:
            pred = row.get("pred_strict")

        used_forced = bool(
            row.get("used_forced_answer")
            or (row.get("pred_strict") is None and row.get("forced_continuation") is not None)
        )
        elapsed = row.get(
            "gpu_seconds_share",
            row.get("elapsed_s", row.get("row_finish_wall_s", row.get("batch_wall_s"))),
        )
        forced_text = row.get("forced_continuation") or row.get("forced_text") or ""
        return {
            "condition": str(row.get("condition", "")),
            "unique_id": str(row.get("unique_id", "")),
            "correct": correct,
            "pred": pred,
            "used_forced_answer": used_forced,
            "n_tokens": int(row.get("n_tokens", 0)),
            "elapsed_s": float(elapsed) if elapsed is not None else None,
            "hit_cap": bool(row.get("hit_cap", False)),
            "loop_onset_token": row.get("loop_onset_token"),
            "text": str(row.get("text", "")),
            "forced_text": str(forced_text),
            "source": source,
        }

    def available_problem_ids(self) -> List[str]:
        logged_ids = set()
        for cond_map in self.records.values():
            logged_ids.update(cond_map.keys())
        ordered = [p["unique_id"] for p in self.problems if p["unique_id"] in logged_ids]
        if ordered:
            return ordered
        return [p["unique_id"] for p in self.problems[:20]]

    def filter_problem_choices(self, filter_mode: str) -> List[Tuple[str, str]]:
        avail = self.available_problem_ids()
        filtered: List[str] = []
        for uid in avail:
            r1 = self.records["loop_n1"].get(uid)
            r2 = self.records["loop_n2"].get(uid)
            r3 = self.records["loop_n3"].get(uid)
            r4 = self.records["loop_n4"].get(uid)
            rq = self.records["qwen35_9b"].get(uid)

            if filter_mode == "Loop passes disagree (1–4)":
                verdicts = {r["correct"] for r in (r1, r2, r3, r4) if r is not None}
                if len(verdicts) < 2:
                    continue
            elif filter_mode == "2 passes beats Plain Qwen3.5-9B":
                if not (r2 and rq and r2["correct"] and not rq["correct"]):
                    continue
            elif filter_mode == "Plain Qwen3.5-9B beats 2 passes":
                if not (r2 and rq and rq["correct"] and not r2["correct"]):
                    continue
            elif filter_mode == "Hit token cap or repetition loop":
                any_issue = any(
                    (r is not None) and (r["hit_cap"] or r["loop_onset_token"] is not None)
                    for r in (r1, r2, r3, r4, rq)
                )
                if not any_issue:
                    continue
            filtered.append(uid)

        if not filtered:
            filtered = avail

        choices: List[Tuple[str, str]] = []
        for uid in filtered:
            prob = self.problem_by_id.get(uid)
            if not prob:
                continue
            plain_preview = format_math_prose_plain(prob["problem"]).replace("\n", " ")
            if len(plain_preview) > 74:
                plain_preview = plain_preview[:71].rstrip() + "..."
            label = f"#{int(prob['idx']) + 1:03d} · Level {prob['level']} {prob['subject']} · {plain_preview}"
            choices.append((label, uid))
        return choices

    def render_scoreboard_html(self) -> str:
        chips: List[str] = []
        for cond, meta in CONDITION_META.items():
            rows = list(self.records[cond].values())
            if not rows:
                continue
            n_tot = len(rows)
            n_ok = sum(1 for r in rows if r["correct"])
            pct = (100.0 * n_ok / n_tot) if n_tot else 0.0
            med_toks = int(round(statistics.median(r["n_tokens"] for r in rows)))
            chips.append(
                f'<div class="ld-score-chip">'
                f'<strong>{_esc(meta["title"])}</strong>: '
                f'{n_ok}/{n_tot} solved ({pct:.1f}%) · median {med_toks:,} tokens'
                f"</div>"
            )
        chips_html = "".join(chips) if chips else (
            '<div class="ld-score-chip">No saved runs found in results/ yet. Run smoke_test.py or run_sweep.py.</div>'
        )
        return (
            '<section class="ld-banner" aria-label="Experiment summary">'
            "<div><strong>Turn the Loop Dial:</strong> SenseNova-Loop-9B runs its middle 24 layers twice per token by "
            "default (56 layer steps). Slide the dial from 1 to 4 passes—or compare against plain Qwen3.5-9B "
            "(32 layer steps once)—on the exact same math problems.</div>"
            f'<div class="ld-score-strip">{chips_html}</div>'
            "</section>"
        )


def render_problem_html(prob: Optional[Dict[str, Any]]) -> str:
    if not prob:
        return '<div class="ld-problem-card">Select a problem from the subset above.</div>'
    clean_problem = format_math_prose_plain(prob.get("problem", ""))
    clean_gold = latex_to_plain_text(prob.get("answer", ""))
    subject = prob.get("subject", "Math")
    level = prob.get("level", "?")
    idx_num = int(prob.get("idx", 0)) + 1
    uid = str(prob.get("unique_id", ""))
    item_num = uid.split("/")[-1].replace(".json", "") if uid else "?"
    return (
        '<article class="ld-problem-card">'
        '<div class="ld-problem-meta">'
        f"<span><strong>Problem #{idx_num:03d}</strong> (MATH-500 item {_esc(item_num)})</span>"
        f"<span>· Subject: <strong>{_esc(subject)}</strong></span>"
        f"<span>· Difficulty: <strong>Level {_esc(level)} of 5</strong></span>"
        f"<span>· Verified Reference Answer: <strong>{_esc(clean_gold)}</strong></span>"
        "</div>"
        f'<div class="ld-problem-text">{_esc(clean_problem)}</div>'
        "</article>"
    )


def render_condition_card_html(
    cond: str,
    record: Optional[Dict[str, Any]],
) -> str:
    meta = CONDITION_META.get(cond, CONDITION_META["loop_n2"])
    title = meta["title"]
    subtitle = meta["subtitle"]
    layers_per_tok = meta["layers_per_token"]

    if record is None:
        return (
            '<section class="ld-output-card">'
            '<div class="ld-card-header">'
            f'<div><h3 class="ld-card-title">{_esc(title)}</h3>'
            f'<div class="ld-card-sub">{_esc(subtitle)}</div></div>'
            "</div>"
            "<p>No saved output logged for this condition and problem yet.</p>"
            "</section>"
        )

    is_ok = bool(record["correct"])
    verdict_cls = "ld-verdict ld-verdict-ok" if is_ok else "ld-verdict ld-verdict-bad"
    verdict_txt = "✓ Correct" if is_ok else "✗ Wrong"

    clean_ans = latex_to_plain_text(record.get("pred"))
    ans_note = " (from cap forced-answer step)" if record.get("used_forced_answer") else ""

    n_tok = int(record.get("n_tokens", 0))
    elapsed = record.get("elapsed_s")
    sec_str = f"{elapsed:.1f} s" if isinstance(elapsed, (int, float)) else "n/a"

    loop_onset = record.get("loop_onset_token")
    if loop_onset is not None:
        loop_str = f"Repeated loop at token {int(loop_onset):,}"
    elif record.get("hit_cap"):
        loop_str = "Hit token cap (no 8-gram loop)"
    else:
        loop_str = "Finished cleanly"

    think_raw, response_raw = split_thinking_and_response(record.get("text", ""))
    clean_think = format_math_prose_plain(think_raw) or "No separate thinking trace recorded."
    clean_response = format_math_prose_plain(response_raw)
    if not clean_response and record.get("forced_text"):
        clean_response = "Forced final answer step: " + format_math_prose_plain(record.get("forced_text"))

    response_block = (
        f'<div class="ld-final-summary">{_esc(clean_response)}</div>'
        if clean_response
        else ""
    )

    return (
        '<section class="ld-output-card">'
        '<div class="ld-card-header">'
        f'<div><h3 class="ld-card-title">{_esc(title)}</h3>'
        f'<div class="ld-card-sub">{_esc(subtitle)}</div></div>'
        f'<span class="{verdict_cls}">{_esc(verdict_txt)}</span>'
        "</div>"
        '<div class="ld-answer-box">'
        f'<div class="ld-answer-label">Model Final Answer{_esc(ans_note)}</div>'
        f'<div class="ld-answer-value">{_esc(clean_ans)}</div>'
        "</div>"
        '<div class="ld-metrics-row">'
        f'<span class="ld-metric-pill"><strong>{n_tok:,}</strong> tokens</span>'
        f'<span class="ld-metric-pill"><strong>{_esc(sec_str)}</strong> GPU time</span>'
        f'<span class="ld-metric-pill"><strong>{int(layers_per_tok)}</strong> layer steps/token</span>'
        f'<span class="ld-metric-pill">{_esc(loop_str)}</span>'
        "</div>"
        f"{response_block}"
        '<details class="ld-details">'
        f"<summary>Show step-by-step reasoning ({n_tok:,} tokens)</summary>"
        f'<pre class="ld-trace-pre" tabindex="0"><code>{_esc(clean_think)}</code></pre>'
        "</details>"
        "</section>"
    )


def resolve_pair_views(
    store: ReplayStore,
    problem_id: str,
    left_passes: int,
    compare_qwen: bool,
    right_passes: int,
) -> Tuple[str, str, str]:
    """Pure helper returning `(problem_html, left_card_html, right_card_html)` for unit tests and Gradio."""
    left_n = clamp_passes(left_passes, default=2)
    right_n = clamp_passes(right_passes, default=1)
    left_cond = f"loop_n{left_n}"
    right_cond = "qwen35_9b" if bool(compare_qwen) else f"loop_n{right_n}"

    prob = store.problem_by_id.get(str(problem_id))
    if prob is None and store.problems:
        prob = store.problems[0]
        problem_id = prob["unique_id"]

    prob_html = render_problem_html(prob)
    left_html = render_condition_card_html(left_cond, store.records[left_cond].get(problem_id))
    right_html = render_condition_card_html(right_cond, store.records[right_cond].get(problem_id))
    return prob_html, left_html, right_html


def build_app(store: Optional[ReplayStore] = None, *, enable_live_gpu: bool = False) -> gr.Blocks:
    store = store or ReplayStore()
    initial_choices = store.filter_problem_choices(FILTER_CHOICES[0])
    initial_uid = initial_choices[0][1] if initial_choices else ""

    with gr.Blocks(title="Loop Dial — SenseNova-Loop-9B vs Qwen3.5-9B") as demo:
        banner_html = gr.HTML(value=store.render_scoreboard_html())

        with gr.Row():
            filter_radio = gr.Radio(
                choices=FILTER_CHOICES,
                value=FILTER_CHOICES[0],
                label="Filter saved problems",
                scale=3,
            )
            problem_dropdown = gr.Dropdown(
                choices=initial_choices,
                value=initial_uid,
                label="Math problem from fixed subset",
                scale=4,
            )

        with gr.Row():
            with gr.Column(scale=1):
                left_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                    label="Left card: Middle-layer passes per token (1 to 4)",
                )
            with gr.Column(scale=1):
                qwen_toggle = gr.Checkbox(
                    value=True,
                    label="Right card: Compare against Plain Qwen3.5-9B (uncheck to pick another pass count)",
                )
                right_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    visible=False,
                    label="Right card: Middle-layer passes per token (1 to 4)",
                )

        prob_card, init_left, init_right = resolve_pair_views(
            store,
            initial_uid,
            left_passes=2,
            compare_qwen=True,
            right_passes=1,
        )
        problem_view = gr.HTML(value=prob_card)

        with gr.Row():
            left_view = gr.HTML(value=init_left)
            right_view = gr.HTML(value=init_right)

        def _on_filter_change(mode: str) -> Tuple[Any, str, str, str]:
            choices = store.filter_problem_choices(mode)
            uid = choices[0][1] if choices else ""
            p_html, l_html, r_html = resolve_pair_views(store, uid, 2, True, 1)
            return gr.update(choices=choices, value=uid), p_html, l_html, r_html

        def _on_qwen_toggle(
            use_qwen: bool,
            uid: str,
            l_passes: int,
            r_passes: int,
        ) -> Tuple[Any, str, str, str]:
            p_html, l_html, r_html = resolve_pair_views(store, uid, l_passes, use_qwen, r_passes)
            return gr.update(visible=not bool(use_qwen)), p_html, l_html, r_html

        def _on_selection_change(
            uid: str,
            l_passes: int,
            use_qwen: bool,
            r_passes: int,
        ) -> Tuple[str, str, str]:
            return resolve_pair_views(store, uid, l_passes, use_qwen, r_passes)

        filter_radio.change(
            fn=_on_filter_change,
            inputs=[filter_radio],
            outputs=[problem_dropdown, problem_view, left_view, right_view],
        )
        qwen_toggle.change(
            fn=_on_qwen_toggle,
            inputs=[qwen_toggle, problem_dropdown, left_slider, right_slider],
            outputs=[right_slider, problem_view, left_view, right_view],
        )
        for ctrl in (problem_dropdown, left_slider, right_slider):
            ctrl.change(
                fn=_on_selection_change,
                inputs=[problem_dropdown, left_slider, qwen_toggle, right_slider],
                outputs=[problem_view, left_view, right_view],
            )

        if enable_live_gpu:
            with gr.Accordion("Live GPU Mode (Colab GPU)", open=False):
                live_prompt = gr.Textbox(
                    lines=3,
                    max_lines=6,
                    label="Custom math question (clamped to 2,000 characters)",
                    placeholder="Find the sum of all positive divisors of 360.",
                )
                live_gold = gr.Textbox(
                    lines=1,
                    label="Optional expected answer for automatic grading",
                    placeholder="1170",
                )
                live_max_toks = gr.Slider(
                    minimum=256,
                    maximum=4096,
                    step=256,
                    value=2048,
                    label="Maximum generated tokens",
                )
                live_btn = gr.Button("Run Live on GPU", variant="primary")
                with gr.Row():
                    live_left_out = gr.HTML()
                    live_right_out = gr.HTML()

                def _run_live(
                    question: str,
                    gold: str,
                    max_toks: int,
                    l_passes: int,
                    use_qwen: bool,
                    r_passes: int,
                ) -> Tuple[str, str]:
                    import time
                    import torch
                    from loop_dial.model_utils import load_qwen35_model, load_sensenova_loop_model, unload_model

                    q_clean = str(question or "").strip()[:2000]
                    if not q_clean:
                        msg = '<div class="ld-output-card">Enter a math question above first.</div>'
                        return msg, msg
                    cap = max(256, min(4096, int(max_toks)))
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    def _exec_one(cond_key: str) -> str:
                        if cond_key == "qwen35_9b":
                            tok, mdl, _ = load_qwen35_model(QWEN35_MODEL_ID, device=device)
                        else:
                            n_p = clamp_passes(cond_key.replace("loop_n", ""), default=2)
                            tok, mdl, _ = load_sensenova_loop_model(
                                SENSENOVA_LOOP_MODEL_ID, n_passes=n_p, device=device
                            )
                        try:
                            chat = tok.apply_chat_template(
                                [{"role": "user", "content": build_math_prompt(q_clean)}],
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True,
                            )
                            enc = tok(chat, return_tensors="pt").to(device)
                            plen = int(enc["input_ids"].shape[1])
                            t0 = time.perf_counter()
                            out = mdl.generate(
                                **enc,
                                max_new_tokens=cap,
                                do_sample=True,
                                temperature=float(DEFAULT_SAMPLING["temperature"]),
                                top_p=float(DEFAULT_SAMPLING["top_p"]),
                                top_k=int(DEFAULT_SAMPLING["top_k"]),
                                pad_token_id=int(tok.pad_token_id),
                                use_cache=True,
                            )
                            if device == "cuda" and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            dt = time.perf_counter() - t0
                            gen_ids = out[0, plen:].tolist()
                            txt = tok.decode(gen_ids, skip_special_tokens=False)
                            for eos_s in ("<|im_end|>", "<|endoftext|>"):
                                if txt.endswith(eos_s):
                                    txt = txt[: -len(eos_s)]
                            hit_c = len(gen_ids) >= cap
                            pred = extract_last_boxed_strict(txt, hit_cap=hit_c) or extract_last_boxed(txt)
                            ok = grade_math_answer(pred, gold.strip()) if gold and gold.strip() else bool(pred)
                            rec = {
                                "condition": cond_key,
                                "correct": ok,
                                "pred": pred,
                                "used_forced_answer": False,
                                "n_tokens": len(gen_ids),
                                "elapsed_s": dt,
                                "hit_cap": hit_c,
                                "loop_onset_token": None,
                                "text": txt,
                                "forced_text": "",
                            }
                            return render_condition_card_html(cond_key, rec)
                        finally:
                            unload_model(mdl, tok)

                    l_key = f"loop_n{clamp_passes(l_passes, 2)}"
                    r_key = "qwen35_9b" if bool(use_qwen) else f"loop_n{clamp_passes(r_passes, 1)}"
                    return _exec_one(l_key), _exec_one(r_key)

                live_btn.click(
                    fn=_run_live,
                    inputs=[live_prompt, live_gold, live_max_toks, left_slider, qwen_toggle, right_slider],
                    outputs=[live_left_out, live_right_out],
                )

        _ = banner_html
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Loop Dial side-by-side Gradio explorer.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (defaults to 127.0.0.1 for local safety)")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (default: 7860)")
    parser.add_argument(
        "--enable-live-gpu",
        action="store_true",
        help="Enable optional live GPU inference tab (for Colab GPU sessions)",
    )
    args = parser.parse_args()

    store = ReplayStore()
    demo = build_app(store=store, enable_live_gpu=args.enable_live_gpu)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        head=HEAD_META,
    )


if __name__ == "__main__":
    main()
