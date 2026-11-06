#!/usr/bin/env python3
"""Phase A Smoke Test for `loop-dial` on Colab GPU (`transformers==5.18.0`, `bfloat16`).

Performs and logs three checks:
1. Environment & hardware snapshot (`python`, `torch`, `cuda`, `transformers`, `accelerate`, `safetensors`, `nvidia-smi`).
2. Loads `sensenova/SenseNova-Loop-9B-Preview` for `n in {1, 2, 3, 4}` (plus `Qwen/Qwen3.5-9B` baseline),
   verifies `loop_settings` and `len(plan) in {32, 56, 80, 104}`, and registers forward hooks on all 32 physical
   decoder layers over 64 generated tokens to prove `8 + 24*n` layer calls/token and measure `ms/token`.
   Saves structured results to `results/smoke.json`.
3. Runs 5 fixed `MATH-500` problems (`data/subset_200.json[:5]`, `4,096`-token cap) at `n = 1, 2, 3, 4` using
   left-padded batched Hugging Face generation with `DEFAULT_SAMPLING`, grades outputs, checks coherence,
   estimates full-sweep HF wall-clock time, and saves full traces to `results/smoke_5problems.jsonl`.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import transformers
from transformers import GenerationConfig, LogitsProcessorList

from loop_dial.common import RESULTS_DIR, atomic_write_json, load_subset_from_disk
from loop_dial.metrics import (
    DEFAULT_SAMPLING,
    PresencePenaltyLogitsProcessor,
    QWEN35_MODEL_ID,
    SENSENOVA_LOOP_MODEL_ID,
    build_math_prompt,
    compute_loop_onset_from_token_ids,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    inspect_trace_coherence,
)
from loop_dial.model_utils import (
    DecoderLayerCallCounter,
    load_qwen35_model,
    load_sensenova_loop_model,
    unload_model,
)

TIMING_PROMPT = (
    "Find all positive integers $n$ such that $n^2 + 19n + 48$ is a perfect square."
)


def get_nvidia_smi() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def get_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
    }
    for mod_name in ("accelerate", "safetensors", "datasets", "math_verify", "fla", "causal_conv1d", "vllm"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            versions[mod_name] = "not_installed"
    return versions


def collect_eos_ids(tokenizer: Any, model: Any) -> List[int]:
    eos_set = set()
    for candidate in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if isinstance(candidate, int):
            eos_set.add(candidate)
        elif isinstance(candidate, (list, tuple)):
            eos_set.update(int(x) for x in candidate)
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_set.add(tid)
    return sorted(eos_set)


@torch.no_grad()
def measure_64_token_scaling(
    model: Any,
    tokenizer: Any,
    *,
    device: str = "cuda",
    n_gen_tokens: int = 64,
) -> Dict[str, Any]:
    """Run 1 warmup + 1 timed `n_gen_tokens` generation with decoder-layer forward hooks."""
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_math_prompt(TIMING_PROMPT)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tokenizer(chat_prompt, return_tensors="pt").to(device)
    prompt_len = int(enc["input_ids"].shape[1])

    # Warmup (8 tokens) so allocator / lazy Triton / cuBLAS init do not pollute timing
    _ = model.generate(
        **enc,
        max_new_tokens=8,
        min_new_tokens=8,
        do_sample=False,
        pad_token_id=int(tokenizer.pad_token_id),
        use_cache=True,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    with DecoderLayerCallCounter(model) as counter:
        t0 = time.perf_counter()
        out_ids = model.generate(
            **enc,
            max_new_tokens=n_gen_tokens,
            min_new_tokens=n_gen_tokens,
            do_sample=False,
            pad_token_id=int(tokenizer.pad_token_id),
            use_cache=True,
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

    gen_ids = out_ids[0, prompt_len:].tolist()
    actual_gen_len = len(gen_ids)
    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": actual_gen_len,
        "elapsed_s": round(elapsed_s, 4),
        "ms_per_token": round((elapsed_s * 1000.0) / max(actual_gen_len, 1), 3),
        "tokens_per_s": round(actual_gen_len / max(elapsed_s, 1e-6), 2),
        "total_decoder_layer_calls": counter.total_layer_calls,
        "prefill_decoder_layer_calls": counter.prefill_layer_calls,
        "decode_decoder_layer_calls": counter.decode_layer_calls,
        "decoder_layer_calls_per_generated_token": round(
            counter.total_layer_calls / max(actual_gen_len, 1), 4
        ),
        "decode_calls_per_decode_step": round(
            counter.decode_layer_calls / max(actual_gen_len - 1, 1), 4
        ),
        "physical_layer_0_calls": counter.per_layer_calls.get(0, 0),
        "physical_layer_4_calls": counter.per_layer_calls.get(4, 0),
        "physical_layer_15_calls": counter.per_layer_calls.get(15, 0),
        "physical_layer_27_calls": counter.per_layer_calls.get(27, 0),
        "physical_layer_28_calls": counter.per_layer_calls.get(28, 0),
        "peak_gpu_mem_gb": peak_mem_gb,
        "sample_preview": tokenizer.decode(gen_ids[:32], skip_special_tokens=False),
    }


@torch.no_grad()
def run_5_problem_sanity_batch(
    model: Any,
    tokenizer: Any,
    problems: List[Dict[str, Any]],
    *,
    condition: str,
    n_passes: int,
    max_new_tokens: int = 4096,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run left-padded batched HF generation on 5 fixed MATH-500 problems with `DEFAULT_SAMPLING`."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_math_prompt(p["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(DEFAULT_SAMPLING["enable_thinking"]),
        )
        for p in problems
    ]
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    padded_prompt_len = int(enc["input_ids"].shape[1])
    eos_ids = collect_eos_ids(tokenizer, model)
    pad_id = int(tokenizer.pad_token_id)

    native_pp = hasattr(GenerationConfig(), "presence_penalty")
    logits_processors = LogitsProcessorList()
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
        logits_processors.append(
            PresencePenaltyLogitsProcessor(
                prompt_length=padded_prompt_len,
                presence_penalty=float(DEFAULT_SAMPLING["presence_penalty"]),
                exclude_token_ids=[pad_id, *eos_ids],
            )
        )
        gen_kwargs["logits_processor"] = logits_processors
        pp_impl = "PresencePenaltyLogitsProcessor"

    torch.manual_seed(int(DEFAULT_SAMPLING["seed"]))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(DEFAULT_SAMPLING["seed"]))
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out_ids = model.generate(**enc, **gen_kwargs)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    eos_set = set(eos_ids)
    records: List[Dict[str, Any]] = []
    total_gen_tokens = 0

    for row_idx, prob in enumerate(problems):
        raw_gen = out_ids[row_idx, padded_prompt_len:].tolist()
        # Trim trailing pad tokens appended after the first EOS token in batched generation
        trimmed_ids: List[int] = []
        hit_eos = False
        for tid in raw_gen:
            trimmed_ids.append(int(tid))
            if int(tid) in eos_set:
                hit_eos = True
                break
        hit_cap = (len(trimmed_ids) >= max_new_tokens) and (not hit_eos)
        text = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
        # Cleanly remove trailing <|im_end|> / <|endoftext|> marker for display/grading parity
        clean_text = text
        for eos_tok in ("<|im_end|>", "<|endoftext|>"):
            if clean_text.endswith(eos_tok):
                clean_text = clean_text[: -len(eos_tok)]

        pred_strict = extract_last_boxed_strict(clean_text, hit_cap=hit_cap)
        pred_any = extract_last_boxed(clean_text)
        correct_strict = grade_math_answer(pred_strict, prob["answer"])
        correct_any = grade_math_answer(pred_any, prob["answer"])
        loop_onset = compute_loop_onset_from_token_ids(trimmed_ids)
        coherence = inspect_trace_coherence(clean_text, hit_cap=hit_cap)

        total_gen_tokens += len(trimmed_ids)
        records.append(
            {
                "condition": condition,
                "n_passes": n_passes,
                "unique_id": prob["unique_id"],
                "idx": prob["idx"],
                "subject": prob["subject"],
                "level": prob["level"],
                "gold_answer": prob["answer"],
                "max_new_tokens": max_new_tokens,
                "n_tokens": len(trimmed_ids),
                "hit_eos": hit_eos,
                "hit_cap": hit_cap,
                "pred_strict": pred_strict,
                "pred_any": pred_any,
                "correct_strict": correct_strict,
                "correct_any": correct_any,
                "loop_onset_token": loop_onset,
                "coherence": coherence,
                "presence_penalty_impl": pp_impl,
                "batch_wall_s": round(wall_s, 3),
                "text": clean_text,
            }
        )

    summary = {
        "condition": condition,
        "n_passes": n_passes,
        "n_problems": len(records),
        "max_new_tokens": max_new_tokens,
        "presence_penalty_impl": pp_impl,
        "batch_wall_s": round(wall_s, 3),
        "total_generated_tokens": total_gen_tokens,
        "aggregate_tok_per_s": round(total_gen_tokens / max(wall_s, 1e-6), 2),
        "mean_tokens": round(total_gen_tokens / max(len(records), 1), 1),
        "n_correct_strict": sum(1 for r in records if r["correct_strict"]),
        "n_hit_cap": sum(1 for r in records if r["hit_cap"]),
        "n_coherent": sum(1 for r in records if r["coherence"]["coherent"]),
        "n_loop_flagged": sum(1 for r in records if r["loop_onset_token"] is not None),
        "mean_rep_8gram": round(
            sum(r["coherence"]["rep_8gram_ratio"] for r in records) / max(len(records), 1), 4
        ),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A smoke test for loop-dial.")
    parser.add_argument(
        "--passes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Loop pass counts to test (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--include-qwen-timing",
        action="store_true",
        default=True,
        help="Also measure 64-token single-stream timing + 5-problem sanity on Qwen/Qwen3.5-9B",
    )
    parser.add_argument("--sanity-cap", type=int, default=4096, help="Token cap for 5-problem sanity check")
    parser.add_argument("--n-sanity-problems", type=int, default=5, help="Number of MATH-500 sanity problems")
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "smoke.json")
    parser.add_argument("--out-jsonl", type=Path, default=RESULTS_DIR / "smoke_5problems.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    gpu_total_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if device == "cuda"
        else 0.0
    )

    versions = get_package_versions()
    smi_text = get_nvidia_smi()
    print("=== Phase A: Environment & GPU ===")
    print(f"GPU: {gpu_name} ({gpu_total_gb} GB)")
    print(f"Versions: {versions}")
    print(smi_text)

    problems_5 = load_subset_from_disk(n=args.n_sanity_problems)
    all_5prob_records: List[Dict[str, Any]] = []
    scaling_rows: List[Dict[str, Any]] = []
    sanity_summaries: List[Dict[str, Any]] = []

    for n in args.passes:
        cond = f"loop_n{n}"
        print(f"\n=== Loading {SENSENOVA_LOOP_MODEL_ID} at n={n} passes ===")
        tok, model, load_meta = load_sensenova_loop_model(n_passes=n, device=device)
        print(
            f"  loop_settings={load_meta['loop_settings']} | len(plan)={load_meta['plan_len']} "
            f"(expected {load_meta['expected_plan_len']}) | autoconfig_clean={load_meta['autoconfig_propagated_cleanly']} "
            f"| mhc_dtypes={load_meta['mhc_parameter_dtypes']} | mem={load_meta['gpu_mem_allocated_gb']} GB"
        )
        assert load_meta["plan_matches_expected"], f"Plan mismatch at n={n}: {load_meta}"

        timing = measure_64_token_scaling(model, tok, device=device, n_gen_tokens=64)
        expected_calls = 8 + 24 * n
        print(
            f"  64-tok scaling: calls/token={timing['decoder_layer_calls_per_generated_token']} "
            f"(expected {expected_calls}) | {timing['ms_per_token']} ms/tok ({timing['tokens_per_s']} tok/s)"
        )
        assert abs(timing["decoder_layer_calls_per_generated_token"] - expected_calls) < 1e-6, (
            f"Expected {expected_calls} layer calls/tok, got {timing['decoder_layer_calls_per_generated_token']}"
        )

        scaling_rows.append(
            {
                "condition": cond,
                "model_id": SENSENOVA_LOOP_MODEL_ID,
                "n_passes": n,
                **load_meta,
                **timing,
            }
        )

        print(f"  Running {len(problems_5)} MATH-500 sanity problems (cap={args.sanity_cap})...")
        recs, summ = run_5_problem_sanity_batch(
            model,
            tok,
            problems_5,
            condition=cond,
            n_passes=n,
            max_new_tokens=args.sanity_cap,
            device=device,
        )
        print(
            f"  5-prob summary ({cond}): correct={summ['n_correct_strict']}/{summ['n_problems']} | "
            f"hit_cap={summ['n_hit_cap']} | coherent={summ['n_coherent']}/{summ['n_problems']} | "
            f"mean_toks={summ['mean_tokens']} | wall={summ['batch_wall_s']}s ({summ['aggregate_tok_per_s']} tok/s)"
        )
        all_5prob_records.extend(recs)
        sanity_summaries.append(summ)

        unload_model(model, tok)

    if args.include_qwen_timing:
        cond = "qwen35_9b"
        print(f"\n=== Loading baseline {QWEN35_MODEL_ID} ===")
        tok_q, model_q, load_meta_q = load_qwen35_model(device=device)
        print(f"  len(plan)={load_meta_q['plan_len']} | mem={load_meta_q['gpu_mem_allocated_gb']} GB")
        timing_q = measure_64_token_scaling(model_q, tok_q, device=device, n_gen_tokens=64)
        print(
            f"  64-tok scaling (Qwen3.5-9B): calls/token={timing_q['decoder_layer_calls_per_generated_token']} "
            f"| {timing_q['ms_per_token']} ms/tok ({timing_q['tokens_per_s']} tok/s)"
        )
        scaling_rows.append(
            {
                "condition": cond,
                **load_meta_q,
                **timing_q,
            }
        )
        recs_q, summ_q = run_5_problem_sanity_batch(
            model_q,
            tok_q,
            problems_5,
            condition=cond,
            n_passes=1,
            max_new_tokens=args.sanity_cap,
            device=device,
        )
        print(
            f"  5-prob summary ({cond}): correct={summ_q['n_correct_strict']}/{summ_q['n_problems']} | "
            f"hit_cap={summ_q['n_hit_cap']} | coherent={summ_q['n_coherent']}/{summ_q['n_problems']} | "
            f"mean_toks={summ_q['mean_tokens']} | wall={summ_q['batch_wall_s']}s ({summ_q['aggregate_tok_per_s']} tok/s)"
        )
        all_5prob_records.extend(recs_q)
        sanity_summaries.append(summ_q)
        unload_model(model_q, tok_q)

    # Save JSONL and smoke.json atomically
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        import json

        for rec in all_5prob_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    smoke_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "gpu_total_gb": gpu_total_gb,
        "versions": versions,
        "nvidia_smi": smi_text,
        "sampling": DEFAULT_SAMPLING,
        "sanity_problem_ids": [p["unique_id"] for p in problems_5],
        "scaling_64_tokens": scaling_rows,
        "sanity_5_problems_summary": sanity_summaries,
    }
    atomic_write_json(args.out_json, smoke_payload)
    print(f"\nSaved Phase A results to {args.out_json} and {args.out_jsonl}")


if __name__ == "__main__":
    main()
