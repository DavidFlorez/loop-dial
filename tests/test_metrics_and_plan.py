"""Unit tests for metrics, boxed extraction, grading, loop detection, execution plans, and plain-text sanitization."""

from __future__ import annotations

import math

import pytest

from loop_dial.metrics import (
    compute_loop_onset_from_token_ids,
    exact_mcnemar_test,
    extract_forced_boxed,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    inspect_trace_coherence,
    latex_to_plain_text,
    repeated_8gram_ratio,
    wilson_ci,
)
from loop_dial.model_utils import expected_execution_plan


def test_expected_execution_plan_lengths_and_structure() -> None:
    for n, expected_len in ((1, 32), (2, 56), (3, 80), (4, 104)):
        plan = expected_execution_plan(n)
        assert len(plan) == expected_len
        assert plan[:4] == (0, 1, 2, 3)
        assert plan[-4:] == (28, 29, 30, 31)
        assert plan.count(4) == n
        assert plan.count(27) == n
        assert plan.count(0) == 1
        assert plan.count(31) == 1


def test_boxed_extraction_strict_and_forced() -> None:
    trace_done = "<think>\nMaybe \\boxed{99}? Wait, 6*7=42.\n</think>\n\nThe answer is \\boxed{\\frac{3}{4}}."
    assert extract_last_boxed_strict(trace_done, hit_cap=False) == r"\frac{3}{4}"
    assert extract_last_boxed_strict(trace_done, hit_cap=True) == r"\frac{3}{4}"

    trace_capped_mid_think = "<think>\nStill thinking, interim \\boxed{12} but wait..."
    assert extract_last_boxed_strict(trace_capped_mid_think, hit_cap=True) is None
    assert extract_last_boxed(trace_capped_mid_think) == "12"

    assert extract_forced_boxed("218}\\n<|im_end|>") == "218"
    assert extract_forced_boxed("\\frac{1}{2}}.") == r"\frac{1}{2}"


def test_grade_math_answer_equivalence() -> None:
    assert grade_math_answer(r"\frac{6}{8}", r"\frac{3}{4}")
    assert grade_math_answer(r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})")
    assert grade_math_answer("42.0", "42")
    assert grade_math_answer("90^\\circ", "90")
    assert not grade_math_answer("41", "42")
    assert not grade_math_answer(None, "42")


def test_loop_onset_and_8gram_ratio() -> None:
    # 1024 unique tokens -> no loop
    unique_ids = list(range(1500))
    assert compute_loop_onset_from_token_ids(unique_ids) is None

    # 512 unique tokens followed by a 16-token phrase repeated 80 times (1280 tokens)
    phrase = list(range(5000, 5016))
    looping_ids = list(range(512)) + phrase * 80
    onset = compute_loop_onset_from_token_ids(looping_ids)
    assert onset is not None
    assert onset in (256, 512)

    rep_txt = " ".join(["the quick brown fox jumps over lazy dog"] * 10)
    assert repeated_8gram_ratio(rep_txt) > 0.80


def test_mcnemar_and_wilson_ci() -> None:
    lo, hi = wilson_ci(40, 50)
    assert 0.66 < lo < 0.80 < hi < 0.90

    a = [True] * 30 + [True] * 2 + [False] * 10 + [False] * 8
    b = [True] * 30 + [False] * 2 + [True] * 10 + [False] * 8
    res = exact_mcnemar_test(a, b)
    assert res["n_paired"] == 50
    assert res["n11_both_correct"] == 30
    assert res["lost_by_b"] == 2
    assert res["gained_by_b"] == 10
    assert math.isclose(res["diff_b_minus_a_pp"], 16.0, abs_tol=1e-6)
    assert math.isclose(res["p_value"], 0.03857421875, rel_tol=1e-5)


@pytest.mark.parametrize(
    "raw_latex",
    [
        r"\boxed{\frac{3}{4}}",
        r"$\left( 3, \frac{\pi}{2} \right)$",
        r"\sqrt{13} + \frac{1+\sqrt{5}}{2}",
        r"90^\circ",
        r"\text{Monday}",
        r"\binom{10}{3}",
    ],
)
def test_latex_to_plain_text_strips_all_raw_tex(raw_latex: str) -> None:
    plain = latex_to_plain_text(raw_latex)
    assert plain and plain != "—"
    for forbidden in ("\\", "$", "{", "}", "boxed", "frac", "left", "right"):
        assert forbidden not in plain


def test_coherence_inspector() -> None:
    good = "<think>\nLet us solve this equation step by step using algebra and factoring.\n</think>\n\\boxed{7}"
    info = inspect_trace_coherence(good, hit_cap=False)
    assert info["coherent"] is True
    assert info["closed_think"] is True
    assert info["has_strict_boxed"] is True


def test_aime_2024_2025_frozen_subset_and_grading() -> None:
    from loop_dial.common import DATA_DIR, load_subset_from_disk
    from loop_dial.metrics import format_math_prose_plain

    aime_path = DATA_DIR / "aime_2024_2025.json"
    assert aime_path.is_file(), "Run scripts/prepare_aime.py first to build data/aime_2024_2025.json"

    probs = load_subset_from_disk(n=60, subset_path=aime_path)
    assert len(probs) == 60
    assert sum(1 for p in probs if p["year"] == 2024) == 30
    assert sum(1 for p in probs if p["year"] == 2025) == 30
    assert [p["idx"] for p in probs] == list(range(60))

    # Verify known AIME answers grade cleanly across string/int/zero-padded representations
    first_24 = probs[0]
    first_25 = probs[30]
    assert grade_math_answer(first_24["answer"], first_24["answer"])
    assert grade_math_answer(first_24["answer_aime3"], first_24["answer"])
    assert grade_math_answer(first_25["answer_aime3"], first_25["answer"])
    assert grade_math_answer("070", "70")
    assert grade_math_answer("70", "70")
    assert not grade_math_answer("71", "70")

    for p in probs:
        clean_prob = format_math_prose_plain(p["problem"])
        assert clean_prob
        for forbidden in ("\\", "$", "{", "}"):
            assert forbidden not in clean_prob, f"Raw markup leaked in {p['unique_id']}: {clean_prob}"


def test_colab_notebook_vllm_setup_and_aime_cells() -> None:
    import json
    from loop_dial.common import REPO_ROOT

    nb_path = REPO_ROOT / "notebooks" / "loop_dial_colab.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}

    assert "vllm-setup-code" in by_id
    assert "RUN_VLLM_SETUP = False" in by_id["vllm-setup-code"]
    assert '"scripts/probe_vllm.py", "--setup-only"' in by_id["vllm-setup-code"]

    assert "sweep-code" in by_id
    assert "RUN_PHASE_B_LIVE = False" in by_id["sweep-code"]
    assert "vllm_py = ensure_vllm_python()" in by_id["sweep-code"]

    assert "aime-live-code" in by_id
    assert "RUN_AIME_LIVE = False" in by_id["aime-live-code"]
    assert "vllm_py = ensure_vllm_python()" in by_id["aime-live-code"]
    assert '"--max-new-tokens", "32768"' in by_id["aime-live-code"]
    assert '"--out-dir", "results/aime"' in by_id["aime-live-code"]


