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


def test_all_60_aime_problem_statements_render_zero_raw_latex():
    aime_path = Path(__file__).resolve().parent.parent / "data" / "aime_2024_2025.json"
    payload = json.loads(aime_path.read_text(encoding="utf-8"))
    problems = payload["problems"]
    assert len(problems) == 60
    for prob in problems:
        plain = format_math_prose_plain(prob["problem"])
        assert plain
        for bad_ch in ("\\", "$", "{", "}"):
            assert bad_ch not in plain, f"Leaked '{bad_ch}' in {prob['unique_id']}: {plain[:160]}"


def test_aime_benchmark_selector_and_xss_escaping(tmp_path: Path):
    data_dir = tmp_path / "data"
    res_dir = tmp_path / "results"
    aime_res_dir = res_dir / "aime"
    data_dir.mkdir()
    aime_res_dir.mkdir(parents=True)

    uid = "aime_2025/1"
    aime_payload = {
        "problems": [
            {
                "idx": 30,
                "unique_id": uid,
                "year": 2025,
                "subject": "AIME 2025",
                "level": "AIME",
                "problem": r"Find the sum $\sum_{k=1}^{9} k$. <img src=x onerror=alert(9)>",
                "answer": "045",
            }
        ]
    }
    (data_dir / "aime_2024_2025.json").write_text(json.dumps(aime_payload), encoding="utf-8")

    row_n2 = {
        "condition": "loop_n2",
        "unique_id": uid,
        "correct_strict": True,
        "correct_forced": True,
        "pred_strict": "45",
        "pred_forced": "45",
        "n_tokens": 2048,
        "gpu_seconds_share": 11.4,
        "hit_cap": False,
        "loop_onset_token": None,
        "text": r"<think>Compute $\frac{9 \cdot 10}{2} = 45$.</think>\boxed{45}",
    }
    row_qwen = {
        "condition": "qwen35_9b",
        "unique_id": uid,
        "correct_strict": False,
        "correct_forced": False,
        "pred_strict": None,
        "pred_forced": "44",
        "used_forced_answer": True,
        "n_tokens": 32768,
        "gpu_seconds_share": 95.0,
        "hit_cap": True,
        "loop_onset_token": None,
        "text": r"<think>Still checking $\binom{10}{2}$...</think>",
        "forced_continuation": r"44}",
    }
    (aime_res_dir / "loop_n2.jsonl").write_text(json.dumps(row_n2) + "\n", encoding="utf-8")
    (aime_res_dir / "qwen35_9b.jsonl").write_text(json.dumps(row_qwen) + "\n", encoding="utf-8")

    store = loop_app.ReplayStore(data_dir=data_dir, results_dir=res_dir)
    label = loop_app.BENCHMARK_SPECS["aime"]["label"]
    assert loop_app.normalize_benchmark_key(label) == "aime"

    choices = store.filter_problem_choices("2 passes beats Plain Qwen3.5-9B", benchmark=label)
    assert len(choices) == 1
    assert choices[0][1] == uid
    assert "AIME 2025" in choices[0][0]

    board_html = store.render_scoreboard_html(label)
    assert "AIME 2024+2025" in board_html
    assert "1/1 solved (100.0%)" in board_html

    prob_html, left_html, right_html = loop_app.resolve_pair_views(
        store,
        problem_id=uid,
        left_passes=2,
        compare_qwen=True,
        right_passes=1,
        benchmark=label,
    )
    assert "AIME 2025 Olympiad" in prob_html
    assert "<img" not in prob_html
    assert "&lt;img src=x onerror=alert(9)&gt;" in prob_html
    for snippet in (prob_html, left_html, right_html):
        assert "\\" not in snippet
        assert "$" not in snippet
        assert "{" not in snippet
        assert "}" not in snippet

