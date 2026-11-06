"""Unit & security tests for `app.py` offline replay, XSS escaping, and zero-raw-LaTeX rendering."""

from __future__ import annotations

import json
from pathlib import Path

from loop_dial.common import load_subset_from_disk
from loop_dial.metrics import format_math_prose_plain
import app as loop_app


def test_all_200_problem_statements_render_zero_raw_latex():
    problems = load_subset_from_disk(n=200)
    assert len(problems) == 200
    for prob in problems:
        plain = format_math_prose_plain(prob["problem"])
        assert plain
        for bad_ch in ("\\", "$", "{", "}"):
            assert bad_ch not in plain, f"Leaked '{bad_ch}' in {prob['unique_id']}: {plain[:160]}"


def test_xss_escaping_and_slider_clamping(tmp_path: Path):
    data_dir = tmp_path / "data"
    res_dir = tmp_path / "results"
    data_dir.mkdir()
    res_dir.mkdir()

    malicious_uid = "test/algebra/1.json"
    subset_payload = {
        "problems": [
            {
                "idx": 0,
                "unique_id": malicious_uid,
                "subject": 'Algebra <img src=x onerror="alert(1)">',
                "level": "4",
                "problem": r"Solve $\frac{x}{2}=3$. <script>alert('xss')</script>",
                "answer": r"\frac{6}{1}",
            }
        ]
    }
    (data_dir / "subset_200.json").write_text(json.dumps(subset_payload), encoding="utf-8")

    row_n2 = {
        "condition": "loop_n2",
        "unique_id": malicious_uid,
        "correct_final": True,
        "pred_final": "6",
        "n_tokens": 312,
        "elapsed_s": 4.2,
        "hit_cap": False,
        "loop_onset_token": None,
        "text": "<think>Step 1: <script>evil()</script> $\\frac{x}{2}=3$ so $x=6$.</think>\\boxed{6}",
    }
    (res_dir / "loop_n2.jsonl").write_text(json.dumps(row_n2) + "\n", encoding="utf-8")

    store = loop_app.ReplayStore(data_dir=data_dir, results_dir=res_dir)
    prob_html, left_html, right_html = loop_app.resolve_pair_views(
        store,
        problem_id=malicious_uid,
        left_passes=99,   # clamps to 4
        compare_qwen=False,
        right_passes=-5,  # clamps to 1
    )

    # Clamping check
    assert "4 middle passes" in left_html
    assert "1 middle pass" in right_html

    # Now check loop_n2 rendering specifically
    _, n2_html, _ = loop_app.resolve_pair_views(
        store,
        problem_id=malicious_uid,
        left_passes=2,
        compare_qwen=True,
        right_passes=1,
    )
    for snippet in (prob_html, n2_html, right_html):
        assert "<script>" not in snippet
        assert 'onerror="alert(1)"' not in snippet
        assert "\\" not in snippet
        assert "$" not in snippet

    assert "<details class=\"ld-details\">" in n2_html
    assert "<summary>Show step-by-step reasoning (312 tokens)</summary>" in n2_html
    assert "&lt;script&gt;evil()&lt;/script&gt;" in n2_html
