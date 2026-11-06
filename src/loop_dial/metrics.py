"""Grading, boxed-answer extraction, loop detection, paired statistics, and plain-text math formatting."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

SENSENOVA_LOOP_MODEL_ID = "sensenova/SenseNova-Loop-9B-Preview"
SENSENOVA_LOOP_REVISION = "9937e91c7301b7dcbb5bf17e3e4e95c499a11e99"

QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN35_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
FORCE_ANSWER_SUFFIX = "\n</think>\n\nTherefore, the final answer is \\boxed{"
FORCE_BOX_OPEN_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"

DEFAULT_SAMPLING: Dict[str, Any] = {
    "enable_thinking": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "seed": 0,
}

# Optional PyTorch / Transformers PresencePenaltyLogitsProcessor (only instantiated during GPU inference)
try:
    import torch
    from transformers import LogitsProcessor

    class PresencePenaltyLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
        """OpenAI/vLLM-style additive presence penalty over tokens generated after `prompt_length`.

        Subtracts `presence_penalty` once from the logit of each unique token ID generated so far in
        each row of the batch, excluding special padding tokens (`exclude_token_ids`).
        """

        def __init__(
            self,
            prompt_length: int,
            presence_penalty: float = 1.5,
            exclude_token_ids: Optional[Sequence[int]] = None,
        ) -> None:
            if prompt_length < 0:
                raise ValueError(f"prompt_length must be >= 0, got {prompt_length}")
            self.prompt_length = int(prompt_length)
            self.presence_penalty = float(presence_penalty)
            self.exclude_token_ids = set(int(t) for t in (exclude_token_ids or ()))

        def __call__(
            self,
            input_ids: "torch.LongTensor",
            scores: "torch.FloatTensor",
        ) -> "torch.FloatTensor":
            if self.presence_penalty == 0.0 or input_ids.shape[1] <= self.prompt_length:
                return scores

            gen_tokens = input_ids[:, self.prompt_length :]
            penalized = scores.clone()
            for row_idx in range(gen_tokens.shape[0]):
                unique_ids = torch.unique(gen_tokens[row_idx])
                if self.exclude_token_ids:
                    keep = [int(t) for t in unique_ids.tolist() if int(t) not in self.exclude_token_ids]
                    if not keep:
                        continue
                    unique_ids = torch.tensor(keep, device=scores.device, dtype=torch.long)
                penalized[row_idx, unique_ids] -= self.presence_penalty
            return penalized

except Exception:  # pragma: no cover - CPU/offline environment without torch/transformers
    PresencePenaltyLogitsProcessor = None  # type: ignore[assignment,misc]


def build_math_prompt(problem: str) -> str:
    """Format a MATH-500 problem with the standard Qwen3.5 boxed-answer instruction."""
    return f"{problem.strip()}\n\n{MATH_INSTRUCTION}"


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the contents of the last balanced `\\boxed{...}` or `\\fbox{...}` in `text`."""
    if not text:
        return None

    last_start = -1
    prefix_len = 0
    for token in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(token)
        if idx > last_start:
            last_start = idx
            prefix_len = len(token)

    if last_start == -1:
        return None

    i = last_start + prefix_len
    depth = 1
    chars: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)
        i += 1

    if depth != 0:
        return None
    return "".join(chars).strip()


def extract_last_boxed_strict(text: str, hit_cap: bool = False) -> Optional[str]:
    """Extract final `\\boxed{...}` answer using strict post-`</think>` evaluation rules.

    - If `</think>` appears in `text`, searches ONLY after the final `</think>`.
    - If `</think>` is absent and `hit_cap=True`, returns `None` (truncated mid-thought).
    - If `</think>` is absent and `hit_cap=False`, falls back to full `text`.
    """
    if not text:
        return None
    close_tag = "</think>"
    close_idx = text.rfind(close_tag)
    if close_idx != -1:
        post_think = text[close_idx + len(close_tag) :]
        return extract_last_boxed(post_think)
    if hit_cap:
        return None
    return extract_last_boxed(text)


def extract_forced_boxed(continuation_text: str) -> Optional[str]:
    """Extract a `\\boxed{...}` answer from a Pass-2 continuation prefixed with `\\boxed{`."""
    if continuation_text is None:
        return None
    reconstructed = "\\boxed{" + continuation_text
    ans = extract_last_boxed(reconstructed)
    if ans is not None:
        return ans
    first_line = continuation_text.strip().splitlines()[0] if continuation_text.strip() else ""
    cleaned = first_line.rstrip(".").rstrip("$").rstrip("}").strip()
    return cleaned or None


def _normalize_latex_str(expr: str) -> str:
    """Normalize common Hendrycks MATH LaTeX formatting differences for fast string comparison."""
    s = expr.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = re.sub(r"\\(?:text|mathrm|mathbf|mbox)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\s*([0-9])\s*([0-9])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\^\s*\{\\circ\}", "", s)
    s = s.replace("\\circ", "")
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def grade_math_answer(pred_box: Optional[str], gold_answer: str) -> bool:
    """Grade `pred_box` against `gold_answer` via `math_verify` with normalized string fallback."""
    if pred_box is None or not pred_box.strip():
        return False

    pred_clean = pred_box.strip()
    gold_clean = gold_answer.strip()

    if _normalize_latex_str(pred_clean) == _normalize_latex_str(gold_clean):
        return True

    try:
        from math_verify import parse, verify

        gold_wrapped = gold_clean if ("$" in gold_clean or "\\boxed" in gold_clean) else f"${gold_clean}$"
        pred_wrapped = pred_clean if ("$" in pred_clean or "\\boxed" in pred_clean) else f"${pred_clean}$"

        gold_parsed = parse(gold_wrapped, parsing_timeout=2)
        pred_parsed = parse(pred_wrapped, parsing_timeout=2)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed, timeout_seconds=2))
    except Exception:
        pass

    return False


def repeated_8gram_ratio(text: str, n: int = 8) -> float:
    """Fraction of whitespace-delimited `n`-grams in `text` that repeat an earlier `n`-gram."""
    if n <= 0:
        raise ValueError("n must be positive")
    tokens = text.split()
    total_ngrams = len(tokens) - n + 1
    if total_ngrams <= 0:
        return 0.0
    unique_ngrams = len({tuple(tokens[i : i + n]) for i in range(total_ngrams)})
    return float(1.0 - (unique_ngrams / total_ngrams))


def tail_repeated_8gram_ratio(text: str, tail_words: int = 256, n: int = 8) -> float:
    """Compute repeated `n`-gram ratio over the trailing `tail_words` words of `text`."""
    words = text.split()
    if len(words) <= tail_words:
        return repeated_8gram_ratio(text, n=n)
    return repeated_8gram_ratio(" ".join(words[-tail_words:]), n=n)


def compute_loop_onset_from_token_ids(
    token_ids: Sequence[Any],
    *,
    window_tokens: int = 1024,
    stride_tokens: int = 256,
    threshold: float = 0.50,
    n: int = 8,
) -> Optional[int]:
    """Return the first window start index where repeated token `n`-grams exceed `threshold`."""
    n_toks = len(token_ids)
    if n_toks < window_tokens or window_tokens < n:
        return None

    total_ngrams = window_tokens - n + 1
    for start in range(0, n_toks - window_tokens + 1, stride_tokens):
        window = token_ids[start : start + window_tokens]
        unique_cnt = len({tuple(window[i : i + n]) for i in range(total_ngrams)})
        rep = 1.0 - (unique_cnt / total_ngrams)
        if rep > threshold:
            return start
    return None


def compute_loop_onset_token(
    text: str,
    tokenizer: Any,
    *,
    window_tokens: int = 1024,
    stride_tokens: int = 256,
    threshold: float = 0.50,
    n: int = 8,
) -> Optional[int]:
    """Tokenize `text` and detect the earliest 1,024-token window with >50% 8-gram repetition."""
    if not text:
        return None
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return compute_loop_onset_from_token_ids(
        token_ids,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
        threshold=threshold,
        n=n,
    )


def inspect_trace_coherence(text: str, hit_cap: bool = False) -> Dict[str, Any]:
    """Summarize structural coherence of a generated reasoning trace for smoke-test inspection."""
    if not text:
        return {
            "empty": True,
            "closed_think": False,
            "has_boxed": False,
            "rep_8gram_ratio": 0.0,
            "tail_rep_8gram_ratio": 0.0,
            "replacement_char_count": 0,
            "english_word_like": False,
            "coherent": False,
        }
    closed_think = "</think>" in text
    pred_any = extract_last_boxed(text)
    pred_strict = extract_last_boxed_strict(text, hit_cap=hit_cap)
    rep8 = repeated_8gram_ratio(text)
    tail_rep8 = tail_repeated_8gram_ratio(text)
    repl_chars = text.count("\ufffd")
    # Basic sanity check that generation contains normal English/math words rather than gibberish symbols
    alpha_words = re.findall(r"[A-Za-z]{2,}", text[:1500])
    english_like = len(alpha_words) >= 8
    coherent = bool(english_like and repl_chars == 0 and tail_rep8 < 0.75)
    return {
        "empty": False,
        "closed_think": closed_think,
        "has_boxed": pred_any is not None,
        "has_strict_boxed": pred_strict is not None,
        "rep_8gram_ratio": round(rep8, 4),
        "tail_rep_8gram_ratio": round(tail_rep8, 4),
        "replacement_char_count": repl_chars,
        "english_word_like": english_like,
        "coherent": coherent,
    }


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute the 95% Wilson score confidence interval `(low, high)` in `[0.0, 1.0]`."""
    if total <= 0:
        return (0.0, 0.0)
    k = max(0, min(int(successes), int(total)))
    n = float(total)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def exact_mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, Any]:
    """Compute the exact two-sided binomial McNemar test between paired outcomes `(A, B)`.

    Here `A` is typically the reference baseline (e.g., `n=2` default or `Qwen3.5-9B`) and
    `B` is the comparison condition:
    - `n01_b_only` (`gained_by_b`): problems wrong in A, solved in B.
    - `n10_a_only` (`lost_by_b`): problems solved in A, wrong in B.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(f"Paired vectors must have equal length ({len(correct_a)} != {len(correct_b)})")

    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0
    for a, b in zip(correct_a, correct_b):
        ba = bool(a)
        bb = bool(b)
        if ba and bb:
            n11 += 1
        elif ba and not bb:
            n10 += 1
        elif not ba and bb:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
    else:
        k = min(n10, n01)
        tail = sum(math.comb(discordant, i) for i in range(0, k + 1)) * (0.5**discordant)
        p_val = min(1.0, 2.0 * tail)

    total = len(correct_a)
    acc_a = ((n11 + n10) / total) if total > 0 else 0.0
    acc_b = ((n11 + n01) / total) if total > 0 else 0.0
    return {
        "n_paired": total,
        "n11_both_correct": n11,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "n00_both_wrong": n00,
        "gained_by_b": n01,
        "lost_by_b": n10,
        "discordant": discordant,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff_b_minus_a_pp": round((acc_b - acc_a) * 100.0, 2),
        "p_value": float(p_val),
    }


_GREEK_MAP = {
    r"\pi": "π",
    r"\theta": "θ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\Delta": "Δ",
    r"\sigma": "σ",
    r"\omega": "ω",
    r"\phi": "φ",
    r"\lambda": "λ",
    r"\infty": "∞",
    r"\pm": "±",
    r"\mp": "∓",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\ne": "≠",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\in": "∈",
    r"\circ": "°",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
}

_SUBSCRIPT_MAP = str.maketrans(
    {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "+": "₊",
        "-": "₋",
        "=": "₌",
        "(": "₍",
        ")": "₎",
        "a": "ₐ",
        "e": "ₑ",
        "h": "ₕ",
        "i": "ᵢ",
        "j": "ⱼ",
        "k": "ₖ",
        "l": "ₗ",
        "m": "ₘ",
        "n": "ₙ",
        "o": "ₒ",
        "p": "ₚ",
        "r": "ᵣ",
        "s": "ₛ",
        "t": "ₜ",
        "u": "ᵤ",
        "v": "ᵥ",
        "x": "ₓ",
    }
)


def _to_unicode_subscript(sub: str) -> str:
    cleaned = sub.strip()
    if not cleaned:
        return ""
    if all(ch in "0123456789+-=()aehijklmnoprstuvx " for ch in cleaned):
        return cleaned.translate(_SUBSCRIPT_MAP)
    return f"[{cleaned}]"


def latex_to_plain_text(expr: Optional[str]) -> str:
    """Convert LaTeX math snippets (`\\boxed{...}`, `\\frac{a}{b}`, `\\sqrt{x}`, `$...$`) into clean Unicode text.

    Guaranteed to contain zero raw LaTeX tokens (`\\`, `$`, `{`, `}`) so charts and UI cards never leak raw TeX.
    """
    if expr is None:
        return "—"
    s = str(expr).strip()
    if not s:
        return "—"

    # Unwrap outer \boxed{...} if present
    boxed = extract_last_boxed(s)
    if boxed is not None and (s.startswith("\\boxed") or s.startswith("\\fbox")):
        s = boxed

    # Strip math delimiters and sizing commands
    s = s.replace("$$", "").replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    for sizing in (
        r"\left",
        r"\right",
        r"\bigl",
        r"\bigr",
        r"\Bigl",
        r"\Bigr",
        r"\displaystyle",
        r"\!",
        r"\,",
        r"\;",
        r"\:",
        r"\quad",
        r"\qquad",
    ):
        s = s.replace(sizing, "")

    # Superscript degree before general exponent stripping
    s = re.sub(r"\^\s*\{\\circ\}", "°", s)
    s = re.sub(r"\^\s*\\circ", "°", s)

    # Greek & operators first so inner braces like {\pi} become {π}
    for k in sorted(_GREEK_MAP.keys(), key=len, reverse=True):
        s = s.replace(k, _GREEK_MAP[k])

    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    # Format matrix environments cleanly before general backslash handling
    def _fmt_matrix(match: re.Match[str]) -> str:
        body = match.group(1)
        rows = [
            ", ".join(cell.strip() for cell in row.split("&") if cell.strip())
            for row in body.split(r"\\")
            if row.strip()
        ]
        return "[" + "; ".join(rows) + "]"

    s = re.sub(
        r"\\begin\{[pbvBV]?matrix\}(.*?)\\end\{[pbvBV]?matrix\}",
        _fmt_matrix,
        s,
        flags=re.DOTALL,
    )
    s = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", s)

    # Iteratively unwind innermost {...} constructs (roots, text, exponents, subscripts, fractions, binomials)
    for _ in range(8):
        prev = s
        s = re.sub(r"\\(?:text|textrm|mathrm|mathbf|mathit|textbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\substack\s*\{([^{}]*)\}", lambda m: "(" + m.group(1).replace("\\\\", ", ").strip() + ")", s)
        s = re.sub(r"\\sqrt\s*\[([^\[\]]+)\]\s*\{([^{}]+)\}", r"root(\1, \2)", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"√(\1)", s)
        s = re.sub(r"\\sqrt\s*([0-9a-zA-Zπ])", r"√\1", s)
        s = re.sub(r"√\(([0-9a-zA-Zπ]+)\)", r"√\1", s)
        s = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", s)
        s = re.sub(r"\^\(([0-9a-zA-Zπ+-]+)\)", r"^\1", s)
        s = re.sub(
            r"_\s*\{([^{}]+)\}",
            lambda m: _to_unicode_subscript(m.group(1)),
            s,
        )
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*([0-9a-zA-Zπ])", r"(\1)/\2", s)
        s = re.sub(r"\\frac\s*([0-9a-zA-Zπ])\s*\{([^{}]+)\}", r"\1/(\2)", s)
        s = re.sub(r"\\frac\s*([0-9a-zA-Zπ])\s*([0-9a-zA-Zπ])", r"\1/\2", s)
        s = re.sub(r"\\binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"C(\1, \2)", s)
        if s == prev:
            break

    # Convert single-character subscripts like ∑_k=1 or a_n
    s = re.sub(
        r"_\s*([0-9a-zA-Z]+)",
        lambda m: _to_unicode_subscript(m.group(1)),
        s,
    )

    # Simplify single-token numerator/denominator parentheses like (3)/(4) -> 3/4 or (1+√5)/(2) -> (1+√5)/2
    s = re.sub(r"\(([0-9a-zA-Zπ√]+)\)/\(([0-9a-zA-Zπ√]+)\)", r"\1/\2", s)
    s = re.sub(r"\(([0-9a-zA-Zπ√]+)\)/([0-9a-zA-Zπ√]+)", r"\1/\2", s)
    s = re.sub(r"/\(([0-9a-zA-Zπ√]+)\)", r"/\1", s)

    # Strip any remaining backslash commands and braces/dollars
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s or "—"


def split_thinking_and_response(text: Optional[str]) -> Tuple[str, str]:
    """Split a generated trace into `(thinking_text, final_response_text)` without raw `<think>` tags."""
    if not text:
        return "", ""
    s = str(text)
    if "</think>" in s:
        think_part, rest = s.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part, rest.strip()
    if s.lstrip().startswith("<think>"):
        return s.replace("<think>", "", 1).strip(), ""
    return "", s.strip()


def format_math_prose_plain(text: Optional[str]) -> str:
    """Format multi-line math problem statements or reasoning prose into readable plain Unicode math.

    Preserves line/paragraph structure while removing `[asy]...[/asy]`, `<think>` tags, markdown markers,
    and all raw TeX markup (`\\`, `$`, `{`, `}`, `\\boxed`, `\\frac`).
    """
    if not text:
        return ""
    s = str(text)
    s = s.replace("<think>", "").replace("</think>", "\n\nFinal response:\n")
    s = re.sub(r"\[asy\].*?\[/asy\]", "[Diagram]", s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{[^{}]+\}", "\n", s)
    s = re.sub(r"\\end\{[^{}]+\}", "\n", s)
    s = re.sub(r"\\substack\s*\{([^{}]*)\}", lambda m: "(" + m.group(1).replace("\\\\", ", ").strip() + ")", s)
    s = s.replace("\\\\", "\n").replace("&=", "=").replace("&", " ")
    # Convert \boxed{...} inside prose into [Answer: ...] before line-by-line conversion
    for _ in range(4):
        prev = s
        s = re.sub(r"\\(?:boxed|fbox)\s*\{([^{}]+)\}", r"[Answer: \1]", s)
        if s == prev:
            break

    out_lines: List[str] = []
    for raw_line in s.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            out_lines.append("")
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = stripped.replace("**", "")
        out_lines.append(latex_to_plain_text(stripped))

    joined = "\n".join(out_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined

