#!/usr/bin/env python3
"""Phase B Main Sweep Runner for `loop-dial` (`n=1,2,3,4` + `Qwen/Qwen3.5-9B`).

Supports two backends with identical prompts, identical sampling (`DEFAULT_SAMPLING`),
identical Pass-2 forced answer continuation (`512` tokens when hitting cap without strict `\\boxed{}`),
and exact subword prefix-budget grading (`2048, 4096, 8192, 16384[, 32768]`):
- `--backend hf`: Left-padded batched Hugging Face `transformers==5.18.0` (`bfloat16`).
- `--backend vllm`: Continuous-batched `vLLM` (`bfloat16`), supporting `--hf-overrides` for Loop passes
  (when PR #64832 is applied) as well as `Qwen/Qwen3.5-9B`.

Resumable: reads existing `unique_id` records in `results/<tag>.jsonl` and only runs unfinished problems.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from loop_dial.common import RESULTS_DIR, append_jsonl, atomic_write_json, load_subset_from_disk, read_jsonl
from loop_dial.metrics import (
    DEFAULT_SAMPLING,
    FORCE_ANSWER_SUFFIX,
    FORCE_BOX_OPEN_SUFFIX,
    PresencePenaltyLogitsProcessor,
    QWEN35_MODEL_ID,
    SENSENOVA_LOOP_MODEL_ID,
    build_math_prompt,
    compute_loop_onset_from_token_ids,
    extract_forced_boxed,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
)
from loop_dial.model_utils import (
    load_qwen35_model,
    load_sensenova_loop_model,
    unload_model,
)

STANDARD_BUDGETS = (2048, 4096, 8192, 16384, 32768)


def strip_trailing_eos_text(text: str) -> str:
    cleaned = text
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if cleaned.endswith(tok):
            cleaned = cleaned[: -len(tok)]
    return cleaned


def build_forced_pass2_prompt(raw_prompt: str, pass1_clean_text: str) -> str:
    """Append `FORCE_ANSWER_SUFFIX` (if still inside `<think>`) or `FORCE_BOX_OPEN_SUFFIX` (if `</think>` closed)."""
    suffix = FORCE_BOX_OPEN_SUFFIX if "</think>" in pass1_clean_text else FORCE_ANSWER_SUFFIX
    return raw_prompt + pass1_clean_text + suffix


def evaluate_nested_budgets_from_tokens(
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    gold_answer: str,
    hit_eos_full: bool,
    max_new_tokens: int,
    full_clean_text: str,
    full_pred_strict: Optional[str],
    full_pred_forced: Optional[str],
    full_correct_strict: bool,
    full_correct_forced: bool,
) -> Dict[str, Dict[str, Any]]:
    """Compute exact nested token-budget outcomes (`2k/4k/8k/16k[/32k]`) from subword `token_ids`."""
    active_budgets = [b for b in STANDARD_BUDGETS if b <= max_new_tokens]
    if max_new_tokens not in active_budgets:
        active_budgets.append(max_new_tokens)

    n_total = len(token_ids)
    out: Dict[str, Dict[str, Any]] = {}
    for b in sorted(active_budgets):
        if n_total < b or (n_total == b and hit_eos_full):
            # Finished naturally at or before budget `b`
            out[str(b)] = {
                "budget": b,
                "n_tokens": n_total,
                "hit_cap": False,
                "pred_strict": full_pred_strict,
                "correct_strict": full_correct_strict,
                "pred_forced": full_pred_forced if b == max_new_tokens else full_pred_strict,
                "correct_forced": full_correct_forced if b == max_new_tokens else full_correct_strict,
                "loop_onset_token": compute_loop_onset_from_token_ids(token_ids),
            }
        elif b == max_new_tokens:
            out[str(b)] = {
                "budget": b,
                "n_tokens": n_total,
                "hit_cap": not hit_eos_full,
                "pred_strict": full_pred_strict,
                "correct_strict": full_correct_strict,
                "pred_forced": full_pred_forced,
                "correct_forced": full_correct_forced,
                "loop_onset_token": compute_loop_onset_from_token_ids(token_ids),
            }
        else:
            prefix_ids = token_ids[:b]
            prefix_text = strip_trailing_eos_text(tokenizer.decode(prefix_ids, skip_special_tokens=False))
            p_strict = extract_last_boxed_strict(prefix_text, hit_cap=True)
            c_strict = grade_math_answer(p_strict, gold_answer)
            # Also check any boxed answer already emitted inside or after thoughts before `b`
            p_any = p_strict if p_strict is not None else extract_last_boxed(prefix_text)
            c_any = c_strict if p_strict is not None else grade_math_answer(p_any, gold_answer)
            out[str(b)] = {
                "budget": b,
                "n_tokens": b,
                "hit_cap": True,
                "pred_strict": p_strict,
                "correct_strict": c_strict,
                "pred_any_prefix": p_any,
                "correct_any_prefix": c_any,
                "pred_forced": None,  # Populated if Pass-2 is run for intermediate prefix `b`
                "correct_forced": c_strict,
                "loop_onset_token": compute_loop_onset_from_token_ids(prefix_ids),
            }
    return out


def run_condition_hf(
    *,
    condition: str,
    model_id: str,
    n_passes: int,
    problems: List[Dict[str, Any]],
    out_jsonl: Path,
    max_new_tokens: int,
    forced_max_tokens: int,
    batch_size: int,
    seed: int,
    force_intermediate_budgets: bool = False,
) -> Dict[str, Any]:
    """Run resumable batched Hugging Face generation with Pass-2 forced answers on capped problems."""
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    existing = read_jsonl(out_jsonl)
    done_ids = {str(r["unique_id"]) for r in existing}
    todo = [p for p in problems if str(p["unique_id"]) not in done_ids]
    print(
        f"[{condition} | backend=hf] Total requested={len(problems)}, already done={len(done_ids)}, remaining={len(todo)}"
    )
    if not todo:
        return {"condition": condition, "backend": "hf", "skipped_all_existing": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_id == SENSENOVA_LOOP_MODEL_ID:
        tokenizer, model, load_meta = load_sensenova_loop_model(n_passes=n_passes, device=device)
    else:
        tokenizer, model, load_meta = load_qwen35_model(model_id=model_id, device=device)

    eos_set = set()
    for cand in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if isinstance(cand, int):
            eos_set.add(cand)
        elif isinstance(cand, (list, tuple)):
            eos_set.update(int(x) for x in cand)
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_set.add(tid)
    eos_ids = sorted(eos_set)
    pad_id = int(tokenizer.pad_token_id)

    class RowFinishTimerProcessor(LogitsProcessor):  # type: ignore[misc]
        """Records exact stopwatch elapsed time when each row in the batch first emits an EOS token."""

        def __init__(self, prompt_len: int, batch_n: int, eos_token_set: set[int], start_time: float) -> None:
            self.prompt_len = prompt_len
            self.eos_token_set = eos_token_set
            self.start_time = start_time
            self.finish_s: List[Optional[float]] = [None] * batch_n

        def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
            if input_ids.shape[1] > self.prompt_len:
                now_s = time.perf_counter() - self.start_time
                last_toks = input_ids[:, -1].tolist()
                for idx, tid in enumerate(last_toks):
                    if self.finish_s[idx] is None and int(tid) in self.eos_token_set:
                        self.finish_s[idx] = now_s
            return scores

    native_pp = hasattr(GenerationConfig(), "presence_penalty")
    total_wall_s = sum(float(r.get("gpu_seconds_share", 0.0)) for r in existing)

    for b_start in range(0, len(todo), batch_size):
        batch_probs = todo[b_start : b_start + batch_size]
        raw_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_math_prompt(p["problem"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(DEFAULT_SAMPLING["enable_thinking"]),
            )
            for p in batch_probs
        ]
        tokenizer.padding_side = "left"
        enc = tokenizer(raw_prompts, return_tensors="pt", padding=True, truncation=False).to(device)
        padded_prompt_len = int(enc["input_ids"].shape[1])

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        timer_proc = RowFinishTimerProcessor(padded_prompt_len, len(batch_probs), eos_set, t0)
        processors = LogitsProcessorList([timer_proc])

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": float(DEFAULT_SAMPLING["temperature"]),
            "top_p": float(DEFAULT_SAMPLING["top_p"]),
            "top_k": int(DEFAULT_SAMPLING["top_k"]),
            "min_p": float(DEFAULT_SAMPLING["min_p"]),
            "eos_token_id": eos_ids,
            "pad_token_id": pad_id,
            "use_cache": True,
        }
        if native_pp:
            gen_kwargs["presence_penalty"] = float(DEFAULT_SAMPLING["presence_penalty"])
            pp_impl = "native_generation_config"
        else:
            processors.append(
                PresencePenaltyLogitsProcessor(
                    prompt_length=padded_prompt_len,
                    presence_penalty=float(DEFAULT_SAMPLING["presence_penalty"]),
                    exclude_token_ids=[pad_id, *eos_ids],
                )
            )
            pp_impl = "PresencePenaltyLogitsProcessor"
        gen_kwargs["logits_processor"] = processors

        torch.manual_seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        batch_pass1_s = time.perf_counter() - t0

        # Parse Pass-1 outputs and collect any traces needing Pass-2 forced answer
        parsed_rows: List[Dict[str, Any]] = []
        pass2_indices: List[Tuple[int, str, str]] = []  # (row_idx, budget_key, forced_prompt)

        for r_idx, prob in enumerate(batch_probs):
            raw_gen = out_ids[r_idx, padded_prompt_len:].tolist()
            trimmed_ids: List[int] = []
            hit_eos = False
            for tid in raw_gen:
                trimmed_ids.append(int(tid))
                if int(tid) in eos_set:
                    hit_eos = True
                    break
            hit_cap = (len(trimmed_ids) >= max_new_tokens) and (not hit_eos)
            clean_text = strip_trailing_eos_text(tokenizer.decode(trimmed_ids, skip_special_tokens=False))
            pred_strict = extract_last_boxed_strict(clean_text, hit_cap=hit_cap)

            row_finish_s = timer_proc.finish_s[r_idx] if timer_proc.finish_s[r_idx] is not None else batch_pass1_s
            parsed_rows.append(
                {
                    "prob": prob,
                    "raw_prompt": raw_prompts[r_idx],
                    "trimmed_ids": trimmed_ids,
                    "hit_eos": hit_eos,
                    "hit_cap": hit_cap,
                    "clean_text": clean_text,
                    "pred_strict": pred_strict,
                    "row_finish_s": float(row_finish_s),
                    "forced_by_budget": {},
                }
            )
            if pred_strict is None and hit_cap:
                pass2_indices.append(
                    (r_idx, str(max_new_tokens), build_forced_pass2_prompt(raw_prompts[r_idx], clean_text))
                )
            if force_intermediate_budgets:
                for b in STANDARD_BUDGETS:
                    if b < max_new_tokens and len(trimmed_ids) > b:
                        pref_txt = strip_trailing_eos_text(
                            tokenizer.decode(trimmed_ids[:b], skip_special_tokens=False)
                        )
                        if extract_last_boxed_strict(pref_txt, hit_cap=True) is None:
                            pass2_indices.append(
                                (r_idx, str(b), build_forced_pass2_prompt(raw_prompts[r_idx], pref_txt))
                            )

        # Run Pass-2 forced answers in sub-batches if needed
        batch_pass2_s = 0.0
        if pass2_indices and forced_max_tokens > 0:
            t_p2 = time.perf_counter()
            p2_sub = max(1, min(batch_size, 8))
            for k in range(0, len(pass2_indices), p2_sub):
                chunk = pass2_indices[k : k + p2_sub]
                p2_prompts = [item[2] for item in chunk]
                enc_p2 = tokenizer(p2_prompts, return_tensors="pt", padding=True, truncation=False).to(device)
                p2_plen = int(enc_p2["input_ids"].shape[1])
                torch.manual_seed(int(seed))
                if device == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                with torch.no_grad():
                    out_p2 = model.generate(
                        **enc_p2,
                        max_new_tokens=int(forced_max_tokens),
                        do_sample=True,
                        temperature=float(DEFAULT_SAMPLING["temperature"]),
                        top_p=float(DEFAULT_SAMPLING["top_p"]),
                        top_k=int(DEFAULT_SAMPLING["top_k"]),
                        min_p=float(DEFAULT_SAMPLING["min_p"]),
                        eos_token_id=eos_ids,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )
                for c_i, (orig_row_idx, b_key, _) in enumerate(chunk):
                    p2_gen = out_p2[c_i, p2_plen:].tolist()
                    p2_trim: List[int] = []
                    for tid in p2_gen:
                        p2_trim.append(int(tid))
                        if int(tid) in eos_set:
                            break
                    cont_txt = strip_trailing_eos_text(tokenizer.decode(p2_trim, skip_special_tokens=False))
                    parsed_rows[orig_row_idx]["forced_by_budget"][b_key] = cont_txt
            if device == "cuda":
                torch.cuda.synchronize()
            batch_pass2_s = time.perf_counter() - t_p2

        batch_total_s = batch_pass1_s + batch_pass2_s
        total_wall_s += batch_total_s
        batch_tok_sum = max(1, sum(len(r["trimmed_ids"]) for r in parsed_rows))

        out_records: List[Dict[str, Any]] = []
        for r in parsed_rows:
            prob = r["prob"]
            gold = prob["answer"]
            trimmed_ids = r["trimmed_ids"]
            clean_text = r["clean_text"]
            pred_strict = r["pred_strict"]
            correct_strict = grade_math_answer(pred_strict, gold)

            forced_cont_full = r["forced_by_budget"].get(str(max_new_tokens))
            if pred_strict is not None:
                pred_forced = pred_strict
                correct_forced = correct_strict
            elif forced_cont_full is not None:
                pred_forced = extract_forced_boxed(forced_cont_full)
                correct_forced = grade_math_answer(pred_forced, gold)
            else:
                pred_forced = extract_last_boxed(clean_text)
                correct_forced = grade_math_answer(pred_forced, gold)

            budgets_dict = evaluate_nested_budgets_from_tokens(
                trimmed_ids,
                tokenizer,
                gold_answer=gold,
                hit_eos_full=r["hit_eos"],
                max_new_tokens=max_new_tokens,
                full_clean_text=clean_text,
                full_pred_strict=pred_strict,
                full_pred_forced=pred_forced,
                full_correct_strict=correct_strict,
                full_correct_forced=correct_forced,
            )
            for b_key, cont_str in r["forced_by_budget"].items():
                if b_key in budgets_dict and budgets_dict[b_key]["pred_strict"] is None:
                    pf = extract_forced_boxed(cont_str)
                    budgets_dict[b_key]["pred_forced"] = pf
                    budgets_dict[b_key]["correct_forced"] = grade_math_answer(pf, gold)

            gpu_share_s = batch_total_s * (len(trimmed_ids) / batch_tok_sum)
            out_records.append(
                {
                    "condition": condition,
                    "model_id": model_id,
                    "n_passes": n_passes,
                    "backend": "hf",
                    "presence_penalty_impl": pp_impl,
                    "seed": seed,
                    "unique_id": prob["unique_id"],
                    "idx": prob["idx"],
                    "subject": prob["subject"],
                    "level": prob["level"],
                    "gold_answer": gold,
                    "max_new_tokens": max_new_tokens,
                    "n_tokens": len(trimmed_ids),
                    "hit_eos": r["hit_eos"],
                    "hit_cap": r["hit_cap"],
                    "pred_strict": pred_strict,
                    "pred_forced": pred_forced,
                    "correct_strict": correct_strict,
                    "correct_forced": correct_forced,
                    "rep_8gram_ratio": round(repeated_8gram_ratio(clean_text), 4),
                    "tail_rep_8gram_ratio": round(tail_repeated_8gram_ratio(clean_text), 4),
                    "loop_onset_token": compute_loop_onset_from_token_ids(trimmed_ids),
                    "row_finish_wall_s": round(r["row_finish_s"], 3),
                    "gpu_seconds_share": round(gpu_share_s, 3),
                    "batch_size": len(batch_probs),
                    "batch_wall_s": round(batch_total_s, 3),
                    "budgets": budgets_dict,
                    "forced_continuation": forced_cont_full,
                    "text": clean_text,
                }
            )

        append_jsonl(out_jsonl, out_records)
        done_now = len(done_ids) + b_start + len(batch_probs)
        n_corr_batch = sum(1 for x in out_records if x["correct_strict"])
        print(
            f"  [{condition} | hf] {done_now}/{len(problems)} done | batch_strict={n_corr_batch}/{len(out_records)} "
            f"| batch_wall={batch_total_s:.1f}s ({batch_tok_sum / max(batch_total_s, 1e-6):.1f} tok/s)",
            flush=True,
        )

    unload_model(model, tokenizer)
    return {
        "condition": condition,
        "backend": "hf",
        "load_meta": load_meta,
        "total_wall_s": round(total_wall_s, 2),
    }


def run_condition_vllm(
    *,
    condition: str,
    model_id: str,
    n_passes: int,
    problems: List[Dict[str, Any]],
    out_jsonl: Path,
    max_new_tokens: int,
    forced_max_tokens: int,
    seed: int,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 64,
    max_model_len: Optional[int] = None,
    force_intermediate_budgets: bool = True,
) -> Dict[str, Any]:
    """Run resumable continuous-batched vLLM generation with Pass-2 forced answers across all budgets."""
    import os

    bin_dir = str(Path(sys.executable).absolute().parent)
    if bin_dir not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    existing = read_jsonl(out_jsonl)
    done_ids = {str(r["unique_id"]) for r in existing}
    todo = [p for p in problems if str(p["unique_id"]) not in done_ids]
    print(
        f"[{condition} | backend=vllm] Total requested={len(problems)}, already done={len(done_ids)}, remaining={len(todo)}"
    )
    if not todo:
        return {"condition": condition, "backend": "vllm", "skipped_all_existing": True}

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    resolved_max_model_len = (
        int(max_model_len) if max_model_len is not None else min(40960, int(max_new_tokens) + 4096)
    )
    llm_kwargs: Dict[str, Any] = {
        "model": model_id,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": resolved_max_model_len,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_num_seqs": int(max_num_seqs),
    }
    if model_id == SENSENOVA_LOOP_MODEL_ID:
        llm_kwargs["hf_overrides"] = {
            "loop_settings": {
                "start": 4,
                "end": 28,
                "eval_count": int(n_passes),
                "counts": [int(n_passes)],
            }
        }

    llm = LLM(**llm_kwargs)
    raw_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_math_prompt(p["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(DEFAULT_SAMPLING["enable_thinking"]),
        )
        for p in todo
    ]

    sampling_p1 = SamplingParams(
        temperature=float(DEFAULT_SAMPLING["temperature"]),
        top_p=float(DEFAULT_SAMPLING["top_p"]),
        top_k=int(DEFAULT_SAMPLING["top_k"]),
        min_p=float(DEFAULT_SAMPLING["min_p"]),
        presence_penalty=float(DEFAULT_SAMPLING["presence_penalty"]),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )

    t0 = time.perf_counter()
    outputs_p1 = llm.generate(raw_prompts, sampling_p1)
    pass1_wall_s = time.perf_counter() - t0

    parsed_rows: List[Dict[str, Any]] = []
    pass2_jobs: List[Tuple[int, str, str]] = []

    for r_idx, (prob, out_obj) in enumerate(zip(todo, outputs_p1)):
        comp = out_obj.outputs[0]
        token_ids = [int(t) for t in comp.token_ids]
        finish_reason = str(getattr(comp, "finish_reason", "") or "")
        hit_cap = (finish_reason == "length") or (len(token_ids) >= max_new_tokens and finish_reason != "stop")
        hit_eos = not hit_cap
        clean_text = strip_trailing_eos_text(comp.text)
        pred_strict = extract_last_boxed_strict(clean_text, hit_cap=hit_cap)

        metrics_obj = getattr(out_obj, "metrics", None)
        req_lat_s = pass1_wall_s
        if metrics_obj is not None:
            arr = getattr(metrics_obj, "arrival_time", None)
            fin = getattr(metrics_obj, "finished_time", None)
            if arr is not None and fin is not None and fin >= arr:
                req_lat_s = float(fin - arr)

        parsed_rows.append(
            {
                "prob": prob,
                "raw_prompt": raw_prompts[r_idx],
                "trimmed_ids": token_ids,
                "hit_eos": hit_eos,
                "hit_cap": hit_cap,
                "clean_text": clean_text,
                "pred_strict": pred_strict,
                "row_finish_s": req_lat_s,
                "forced_by_budget": {},
            }
        )
        if pred_strict is None and hit_cap:
            pass2_jobs.append((r_idx, str(max_new_tokens), build_forced_pass2_prompt(raw_prompts[r_idx], clean_text)))
        if force_intermediate_budgets:
            for b in STANDARD_BUDGETS:
                if b < max_new_tokens and len(token_ids) > b:
                    pref_txt = strip_trailing_eos_text(tokenizer.decode(token_ids[:b], skip_special_tokens=False))
                    if extract_last_boxed_strict(pref_txt, hit_cap=True) is None:
                        pass2_jobs.append((r_idx, str(b), build_forced_pass2_prompt(raw_prompts[r_idx], pref_txt)))

    pass2_wall_s = 0.0
    if pass2_jobs and forced_max_tokens > 0:
        sampling_p2 = SamplingParams(
            temperature=float(DEFAULT_SAMPLING["temperature"]),
            top_p=float(DEFAULT_SAMPLING["top_p"]),
            top_k=int(DEFAULT_SAMPLING["top_k"]),
            min_p=float(DEFAULT_SAMPLING["min_p"]),
            presence_penalty=0.0,
            max_tokens=int(forced_max_tokens),
            seed=int(seed),
        )
        t_p2 = time.perf_counter()
        p2_outs = llm.generate([j[2] for j in pass2_jobs], sampling_p2)
        pass2_wall_s = time.perf_counter() - t_p2
        for (orig_idx, b_key, _), p2_out in zip(pass2_jobs, p2_outs):
            parsed_rows[orig_idx]["forced_by_budget"][b_key] = strip_trailing_eos_text(p2_out.outputs[0].text)

    total_wall_s = pass1_wall_s + pass2_wall_s
    total_toks = max(1, sum(len(r["trimmed_ids"]) for r in parsed_rows))

    out_records: List[Dict[str, Any]] = []
    for fallback_idx, r in enumerate(parsed_rows):
        prob = r["prob"]
        gold = prob["answer"]
        trimmed_ids = r["trimmed_ids"]
        clean_text = r["clean_text"]
        pred_strict = r["pred_strict"]
        correct_strict = grade_math_answer(pred_strict, gold)

        forced_cont_full = r["forced_by_budget"].get(str(max_new_tokens))
        if pred_strict is not None:
            pred_forced = pred_strict
            correct_forced = correct_strict
        elif forced_cont_full is not None:
            pred_forced = extract_forced_boxed(forced_cont_full)
            correct_forced = grade_math_answer(pred_forced, gold)
        else:
            pred_forced = extract_last_boxed(clean_text)
            correct_forced = grade_math_answer(pred_forced, gold)

        budgets_dict = evaluate_nested_budgets_from_tokens(
            trimmed_ids,
            tokenizer,
            gold_answer=gold,
            hit_eos_full=r["hit_eos"],
            max_new_tokens=max_new_tokens,
            full_clean_text=clean_text,
            full_pred_strict=pred_strict,
            full_pred_forced=pred_forced,
            full_correct_strict=correct_strict,
            full_correct_forced=correct_forced,
        )
        for b_key, cont_str in r["forced_by_budget"].items():
            if b_key in budgets_dict and budgets_dict[b_key]["pred_strict"] is None:
                pf = extract_forced_boxed(cont_str)
                budgets_dict[b_key]["pred_forced"] = pf
                budgets_dict[b_key]["correct_forced"] = grade_math_answer(pf, gold)

        gpu_share_s = total_wall_s * (len(trimmed_ids) / total_toks)
        rec_obj: Dict[str, Any] = {
            "condition": condition,
            "model_id": model_id,
            "n_passes": n_passes,
            "backend": "vllm",
            "presence_penalty_impl": "vllm_sampling_params",
            "seed": seed,
            "unique_id": prob["unique_id"],
            "idx": prob.get("idx", fallback_idx),
            "subject": prob.get("subject", "Math"),
            "level": prob.get("level"),
            "gold_answer": gold,
            "max_new_tokens": max_new_tokens,
            "n_tokens": len(trimmed_ids),
            "hit_eos": r["hit_eos"],
            "hit_cap": r["hit_cap"],
            "pred_strict": pred_strict,
            "pred_forced": pred_forced,
            "correct_strict": correct_strict,
            "correct_forced": correct_forced,
            "rep_8gram_ratio": round(repeated_8gram_ratio(clean_text), 4),
            "tail_rep_8gram_ratio": round(tail_repeated_8gram_ratio(clean_text), 4),
            "loop_onset_token": compute_loop_onset_from_token_ids(trimmed_ids),
            "row_finish_wall_s": round(r["row_finish_s"], 3),
            "gpu_seconds_share": round(gpu_share_s, 3),
            "batch_size": len(todo),
            "batch_wall_s": round(total_wall_s, 3),
            "budgets": budgets_dict,
            "forced_continuation": forced_cont_full,
            "text": clean_text,
        }
        if "year" in prob:
            rec_obj["year"] = prob["year"]
        if "source" in prob:
            rec_obj["source"] = prob["source"]
        out_records.append(rec_obj)

    append_jsonl(out_jsonl, out_records)
    print(
        f"  [{condition} | vllm] Completed {len(out_records)} problems in {total_wall_s:.1f}s "
        f"({total_toks / max(total_wall_s, 1e-6):.1f} tok/s) | "
        f"strict={sum(1 for x in out_records if x['correct_strict'])}/{len(out_records)} | "
        f"forced={sum(1 for x in out_records if x['correct_forced'])}/{len(out_records)}",
        flush=True,
    )
    del llm
    unload_model(tokenizer)
    return {
        "condition": condition,
        "backend": "vllm",
        "max_num_seqs": int(max_num_seqs),
        "max_model_len": resolved_max_model_len,
        "pass1_wall_s": round(pass1_wall_s, 2),
        "pass2_wall_s": round(pass2_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
    }


def resolve_condition_spec(cond_name: str) -> Tuple[str, str, int]:
    """Map condition name (`loop_n1..4` or `qwen35_9b`) to `(canonical_name, model_id, n_passes)`."""
    c = cond_name.strip().lower()
    if c in ("qwen", "qwen35", "qwen35_9b", "qwen3.5-9b"):
        return ("qwen35_9b", QWEN35_MODEL_ID, 1)
    if c.startswith("loop_n") and c[6:].isdigit():
        n = int(c[6:])
        return (f"loop_n{n}", SENSENOVA_LOOP_MODEL_ID, n)
    if c.isdigit():
        n = int(c)
        return (f"loop_n{n}", SENSENOVA_LOOP_MODEL_ID, n)
    raise ValueError(f"Unrecognized condition '{cond_name}'. Expected loop_n1..loop_n4 or qwen35_9b.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MATH-500 or AIME sweep for loop-dial.")
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["loop_n2", "loop_n1", "loop_n3", "loop_n4", "qwen35_9b"],
        help="Conditions to evaluate",
    )
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument(
        "--problems-json",
        type=Path,
        default=None,
        help="Path to frozen problem subset JSON (default: data/subset_200.json; e.g. data/aime_2024_2025.json)",
    )
    parser.add_argument("--n-problems", type=int, default=50, help="Number of subset problems to run")
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Pass-1 token budget")
    parser.add_argument("--forced-max-tokens", type=int, default=512, help="Pass-2 forced answer cap")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for --backend hf")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences for --backend vllm")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional explicit max_model_len for --backend vllm (default: min(40960, max_new_tokens + 4096))",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling RNG seed")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="",
        help="Optional suffix for JSONL filename (e.g., '_hf_speed' -> qwen35_9b_hf_speed.jsonl)",
    )
    parser.add_argument(
        "--force-intermediate-budgets",
        action="store_true",
        default=False,
        help="Also run Pass-2 forced answers at intermediate 2k/4k/8k prefix cuts",
    )
    args = parser.parse_args()

    if args.backend == "vllm" and len(args.conditions) > 1:
        import subprocess

        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_meta_rows = []
        for raw_cond in args.conditions:
            cond, _, _ = resolve_condition_spec(raw_cond)
            child_cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--backend",
                "vllm",
                "--conditions",
                cond,
                "--n-problems",
                str(args.n_problems),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--forced-max-tokens",
                str(args.forced_max_tokens),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(args.out_dir),
            ]
            if args.problems_json is not None:
                child_cmd.extend(["--problems-json", str(args.problems_json)])
            if args.max_model_len is not None:
                child_cmd.extend(["--max-model-len", str(args.max_model_len)])
            if args.file_suffix:
                child_cmd.extend(["--file-suffix", args.file_suffix])
            if args.force_intermediate_budgets:
                child_cmd.append("--force-intermediate-budgets")
            print(f"[run_sweep] Launching isolated vLLM worker for {cond}...", flush=True)
            subprocess.run(child_cmd, check=True)
            sub_manifest = args.out_dir / f"sweep_manifest_vllm_n{args.n_problems}{args.file_suffix}_{cond}.json"
            if sub_manifest.is_file():
                sub_data = json.loads(sub_manifest.read_text(encoding="utf-8"))
                run_meta_rows.extend(sub_data.get("runs", []))
        manifest_path = args.out_dir / f"sweep_manifest_vllm_n{args.n_problems}{args.file_suffix}.json"
        atomic_write_json(
            manifest_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "backend": "vllm",
                "problems_json": str(args.problems_json) if args.problems_json else "data/subset_200.json",
                "n_problems": args.n_problems,
                "max_new_tokens": args.max_new_tokens,
                "forced_max_tokens": args.forced_max_tokens,
                "max_num_seqs": args.max_num_seqs,
                "max_model_len": args.max_model_len,
                "sampling": DEFAULT_SAMPLING,
                "runs": run_meta_rows,
            },
        )
        print(f"Saved combined vLLM sweep manifest to {manifest_path}")
        return

    problems = load_subset_from_disk(n=args.n_problems, subset_path=args.problems_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_meta_rows = []

    for raw_cond in args.conditions:
        cond, model_id, n_passes = resolve_condition_spec(raw_cond)
        out_jsonl = args.out_dir / f"{cond}{args.file_suffix}.jsonl"
        if args.backend == "hf":
            meta = run_condition_hf(
                condition=cond,
                model_id=model_id,
                n_passes=n_passes,
                problems=problems,
                out_jsonl=out_jsonl,
                max_new_tokens=args.max_new_tokens,
                forced_max_tokens=args.forced_max_tokens,
                batch_size=args.batch_size,
                seed=args.seed,
                force_intermediate_budgets=args.force_intermediate_budgets,
            )
        else:
            meta = run_condition_vllm(
                condition=cond,
                model_id=model_id,
                n_passes=n_passes,
                problems=problems,
                out_jsonl=out_jsonl,
                max_new_tokens=args.max_new_tokens,
                forced_max_tokens=args.forced_max_tokens,
                seed=args.seed,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                force_intermediate_budgets=args.force_intermediate_budgets,
            )
        run_meta_rows.append(meta)

    cond_tag = f"_{resolve_condition_spec(args.conditions[0])[0]}" if (args.backend == "vllm" and len(args.conditions) == 1) else ""
    manifest_path = args.out_dir / f"sweep_manifest_{args.backend}_n{args.n_problems}{args.file_suffix}{cond_tag}.json"
    atomic_write_json(
        manifest_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "problems_json": str(args.problems_json) if args.problems_json else "data/subset_200.json",
            "n_problems": args.n_problems,
            "max_new_tokens": args.max_new_tokens,
            "forced_max_tokens": args.forced_max_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "sampling": DEFAULT_SAMPLING,
            "runs": run_meta_rows,
        },
    )
    print(f"Saved sweep manifest to {manifest_path}")


if __name__ == "__main__":
    main()
