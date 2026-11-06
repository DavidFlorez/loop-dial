# Research Notes — `loop-dial`

## Goal

`sensenova/SenseNova-Loop-9B-Preview` is a 9B-parameter hybrid reasoning model built on the 32-layer `Qwen3.5` text-and-vision backbone (`hidden_size=4096`, 8 blocks of `3 × linear_attention + 1 × full_attention`), modified so its middle 24 decoder layers (`4..27`, 0-indexed) run **twice** per token by default (`4 + 24×2 + 4 = 56` virtual layer executions) with two-stream Manifold-Constrained Hyper-Connections (`mHC`, 20 Sinkhorn iterations) and separate KV / FP32 Mamba SSM caches per pass.

This repository tests one plain-English question on `HuggingFaceH4/MATH-500` (`N = 200`, `16,384` max tokens) and follows up on **`AIME 2024 + 2025`** (`N = 60` olympiad problems, `32,768` max tokens) where default `2` passes is well below ceiling:
> **What happens at inference time when you turn SenseNova-Loop-9B-Preview's middle-layer "loop dial" to `n = 1, 2, 3, or 4` passes (`32, 56, 80, or 104` layer executions per token), and is the looped model better than just running plain `Qwen/Qwen3.5-9B` (`32` layers once)?**

Every number in this repository comes from a real, logged model run stored under `results/` (`MATH-500`) and `results/aime/` (`AIME 2024 + 2025`).

---

## Setup

- **Models compared**:
  1. `sensenova/SenseNova-Loop-9B-Preview` at `n ∈ {1, 2, 3, 4}` passes of layers `4..27` (`32, 56, 80, 104` total decoder-layer executions per token; `n=2` is the checkpoint's trained default).
  2. `Qwen/Qwen3.5-9B` (`32` decoder layers run once; non-looped sibling with the same 32-layer hybrid GatedDeltaNet + gated full-attention dimensions, same `248,320` vocabulary, and same `<think>...</think>` template).
- **Benchmarks & deterministic subsets**:
  1. **`MATH-500`**: `HuggingFaceH4/MATH-500` (`test` split, 500 competition math problems), fixed `seed = 0`, deterministic nested subset `S_50 ⊂ S_200` saved in `data/subset_200.json` using the exact `load_math500_subset(seed=0)` sampling rule from `DavidFlorez/watch-the-loop-break`.
  2. **`AIME 2024 + 2025` olympiad follow-up**: All `60` official AIME problems from `HuggingFaceH4/aime_2024` (`30` problems, `unique_id = aime_2024/<id>`) + `MathArena/aime_2025` (`30` problems, `unique_id = aime_2025/<problem_idx>`), frozen in `data/aime_2024_2025.json` via `scripts/prepare_aime.py` (`sorted_unique_ids_sha256 = d712c0e465d8bf7af7890d3dc567c20a93ae187e10cef1deaa079ddd0978bd56`).
- **Prompt & decoding parameters** (identical across all conditions and both benchmarks):
  - User prompt: `<problem>\n\nPlease reason step by step, and put your final answer within \boxed{}.`
  - Chat template: `enable_thinking=True` (assistant prefill `<think>\n`).
  - Sampling (`Qwen3.5-9B` recommended thinking-mode general parameters + `SenseNova-Loop` `generation_config.json` defaults):
    - `temperature = 1.0`
    - `top_p = 0.95`
    - `top_k = 20`
    - `min_p = 0.0`
    - `presence_penalty = 1.5` (applied on generated tokens via `PresencePenaltyLogitsProcessor` under Hugging Face `transformers`, or native `SamplingParams(presence_penalty=1.5)` under vLLM)
    - `seed = 0`, `1` sample per problem.
- **Token budget, nested prefix curve & Pass-2 forced answer**:
  - `MATH-500` (`N = 200`): `max_new_tokens = 16,384` (`max_num_seqs = 64`), nested prefix truncation at `B ∈ {2,048, 4,096, 8,192, 16,384}`.
  - `AIME 2024 + 2025` (`N = 60`): `max_new_tokens = 32,768` (`max_num_seqs = 16`, `max_model_len = 36,864`), nested prefix truncation at `B ∈ {2,048, 4,096, 8,192, 16,384, 32,768}`.
  - When a trace hits the budget cap without emitting a final `\boxed{}` answer, a Pass-2 continuation (`512` max new tokens) closes `</think>` and prompts `\boxed{` so truncated reasoning still gets a fair shot at stating its answer (`correct_forced` alongside `correct_strict`).
- **Grading, loop detection & paired statistics**:
  - Equivalence grading via `math-verify` (`parse` + `verify`) plus Hendrycks string-normalization fallback.
  - Loop detection: `1,024`-token sliding window (`stride = 256`) flagging loop onset when repeated 8-gram fraction exceeds `0.50`, plus full-trace and 256-word tail repeated 8-gram ratios.
  - Paired two-sided exact McNemar tests (`n ∈ {1, 3, 4}` vs `n = 2`, and `n = 2` vs `Qwen3.5-9B`) + 95% Wilson score confidence intervals.

---

## Log

### 2026-11-05 — Fact Re-check Against Live Hugging Face & GitHub Repos
Re-verified all upstream model files, configs, and serving PRs prior to writing code:
- **`sensenova/SenseNova-Loop-9B-Preview`** (HF SHA `9937e91c7301b7dcbb5bf17e3e4e95c499a11e99`, `lastModified: 2026-11-02T15:43:56Z`, Apache-2.0):
  - `config.json`: `transformers_version` target is `5.18.0`; `architectures = ["SenseNovaLoopForConditionalGeneration"]`; `loop_settings = {"counts": [2], "end": 28, "eval_count": 2, "start": 4}`; `mhc = {"end": 28, "fixed_identity": false, "iterations": 20, "pre_scale": 1.0, "start": 4, "streams": 2}`; `text_config.num_hidden_layers = 32`, `hidden_size = 4096`, `intermediate_size = 12288`, `vocab_size = 248320`, `max_position_embeddings = 262144`, `mamba_ssm_dtype = "float32"`.
  - `generation_config.json`: `do_sample = true`, `temperature = 1.0`, `top_p = 0.95`, `eos_token_id = [248044, 248046]`, `pad_token_id = 248044`.
  - `modeling_sensenova_loop.py` (406 lines):
    - `execution_plan(config)` checks `if loop.get("counts") != [count]: raise ValueError("Only fixed-count Loop checkpoints are supported")`. Therefore, dialing loop passes to `n` cleanly via `AutoConfig` requires setting **both** `config.loop_settings["eval_count"] = n` and `config.loop_settings["counts"] = [n]`.
    - `execution_plan(config)` returns `tuple(range(4)) + tuple(range(4, 28)) * n + tuple(range(28, 32))` of length `8 + 24n` (`32, 56, 80, 104` for `n = 1, 2, 3, 4`).
    - `_mhc_bounds(config)` also asserts that `mhc` (`[4, 28)`) does not cross `(0, start, end, 32)`.
    - Parameter count: `9,655,021,424` tensors (`9,651,183,216` BF16 + `3,838,208` F32 for norms/SSM/mHC routers; ~19.31 GB across 4 safetensors shards).
- **`Qwen/Qwen3.5-9B`** (HF SHA `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, `lastModified: 2026-03-02T00:51:43Z`, Apache-2.0):
  - Parameter count: `9,653,104,368` (`9,653,100,528` BF16 + `3,840` F32) — exactly `1,917,056` fewer parameters than `SenseNova-Loop-9B-Preview`, which is the exact size of the 24 two-stream mHC Sinkhorn router modules added to layers `4..27`.
  - Official README recommends `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0` for general thinking-mode tasks (`temperature=0.6` is listed as an alternative specifically for precise coding tasks) and `"Please reason step by step, and put your final answer within \\boxed{}."` for math.
- **Serving PRs & community builds (checked 2026-11-05)**:
  - `vllm-project/vllm#64832` (`sufubao:pr/sensenova-loop`, SHA `86014ad4`): **Still open (unmerged)**. Notably, it is pure Python + Triton (`12` files, `+822` lines, zero C++ rebuild) and `SenseNovaLoopConfig` explicitly reads `loop_settings["eval_count"]` / `counts` (supporting `--hf-overrides`).
  - `sgl-project/sglang#47281` (`WANDY666:feat/sensenova-loop`, SHA `336c54ac`): **Still open (unmerged)**.
  - `OpenSenseNova/SenseNova-Loop`: Contains `README.md`, `README_CN.md`, `assets/`, and `examples/` (front-end HTML/SVG demos; no standalone training/eval package).
  - `AtomicChat/SenseNova-Loop-9B-Preview-GGUF`: Metadata shows `"architecture": "qwen35"` (`total: 8,956,949,376`), indicating the GGUF converter exported the 32 physical layers as a plain `qwen35` graph **without** middle-layer looping or mHC routing.
  - `hermitdave/SenseNova-Loop-9B-Preview-MLX-slq-4.5bpw`: Packed 4.5-bpw SLQ MLX-LM checkpoint (`7.631 GiB`). Despite its `model_type: sensenova_loop` metadata, inspecting and running its `slq_model.py` on an Apple M3 Mac (`results/mac_mlx_check.json`) proved that `slq_model.py` subclasses plain `mlx_lm.models.qwen3_5.Model` (`32` layers, 1 pass, no middle-layer loop) and strips all `.mhc_attn.` and `.mhc_ffn.` tensors inside `load_weights()`.

### 2026-11-06 — Phase A: Local Apple M3 16 GB MLX Verification (`results/mac_mlx_check.json`)
- Tested `hermitdave/SenseNova-Loop-9B-Preview-MLX-slq-4.5bpw` on the local `Apple M3` (`16 GB` unified memory, macOS `26.7.1`, `mlx==0.32.4`, `mlx-lm==0.32.2`) before deleting the temporary `7.63 GiB` download cache.
- Loaded in `7.625 GB` active Metal memory (`7.858 GB` peak) and generated `128` tokens at **`12.34 tok/s`** (`10.371 s`).
- However, inspecting `model.model.layers` (`len = 32`, `hasattr(model.model, "plan") == False`) and `slq_model.py` confirmed **neither community quantized build (`MLX` nor `GGUF`) implements the middle-layer loop or `mHC` Sinkhorn routers**. Because layers `28..31` receive a single-pass representation they were never trained for, the quantized MLX build immediately hallucinated basic algebra (`(n + 19/2)^2` written with `(19/2)^2 = 81/4` instead of `361/4`, plus mid-equation Chinese `的平方` in an English prompt).

### 2026-11-06 — Phase A: Smoke Test & Layer-Count Verification (`results/smoke.json`, `results/smoke_5problems.jsonl`)
- Provisioned Colab `G4` runtime (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, `94.97 GiB` VRAM, Driver `580.82.07`, CUDA `13.0`, Python `3.13.16`, `torch==2.11.0+cu130`, `transformers==5.18.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `fla==0.6.0`).
- **Mechanism & forward-hook verification** (`64` generated tokens, `bfloat16` weights with `mhc` parameters verified at `torch.float32`, unused 447M `model.visual` tower dropped cleanly):
  - `loop_n1` (`eval_count=1, counts=[1]`): `plan_len = 32`, `AutoConfig` propagated cleanly (`True`), forward-hook `decoder_layer_calls_per_generated_token = 32.0` (`64` calls each on layers `0, 4, 15, 27, 28`), eager HF latency `95.145 ms/tok` (`10.51 tok/s`).
  - `loop_n2` (`eval_count=2, counts=[2]`, trained default): `plan_len = 56`, forward-hook `decoder_layer_calls_per_generated_token = 56.0` (`64` calls on entry/exit `0, 28`; `128` calls on middle `4, 15, 27`), eager HF latency `184.356 ms/tok` (`5.42 tok/s`).
  - `loop_n3` (`eval_count=3, counts=[3]`): `plan_len = 80`, forward-hook `decoder_layer_calls_per_generated_token = 80.0` (`192` calls on middle `4, 15, 27`), eager HF latency `270.590 ms/tok` (`3.70 tok/s`).
  - `loop_n4` (`eval_count=4, counts=[4]`): `plan_len = 104`, forward-hook `decoder_layer_calls_per_generated_token = 104.0` (`256` calls on middle `4, 15, 27`), eager HF latency `359.289 ms/tok` (`2.78 tok/s`).
  - `qwen35_9b` (`Qwen/Qwen3.5-9B`): `plan_len = 32`, forward-hook `decoder_layer_calls_per_generated_token = 32.0`, eager HF latency `18.399 ms/tok` (`54.35 tok/s`).
  - *Why eager HF was ~5× slower at `n=1` than `Qwen3.5-9B`*: `modeling_sensenova_loop.py` executes 20 Sinkhorn-Knopp normalization iterations in uncompiled Python at every `mHC` wrapper (`48` wrappers × `20` iterations = `960` tiny eager ops per token at `n=1`).
- **5-problem sanity check (`subset_200[:5]`, `max_new_tokens = 4,096`)**:
  - `loop_n1`: **`0/5` correct**, `3,897.0` mean tokens, `4/5` hit cap, `2/5` flagged for degenerate `∀∀∀∀...` token loops.
  - `loop_n2` (default): **`5/5` correct**, `710.2` mean tokens, `0/5` hit cap, `0` loops.
  - `loop_n3`: **`5/5` correct**, `802.8` mean tokens, `0/5` hit cap, `0` loops.
  - `loop_n4`: **`4/5` correct**, `1,429.8` mean tokens, `1/5` hit cap, `0` loops.
  - `qwen35_9b`: **`1/5` strict correct** at `4,096` tokens (`3,702.0` mean tokens; `4/5` coherent traces were still mid-derivation when the `4,096` cap hit — motivating our `16,384` main budget + nested `2k/4k/8k/16k` budget curve + Pass-2 forced answer).

### 2026-11-06 — Phase A: Partial Middle-Layer Loop Probe (`results/partial_layer_probe.json`)
- Tested whether a subset of the 24 middle layers (`4..27`) can be looped independently:
  1. Overriding `loop_settings = {"start": 4, "end": 16, "eval_count": 2, "counts": [2]}` via `AutoConfig` raises `ValueError: MHC cannot cross a Loop phase boundary` because layer `4` expands `1 -> 2` streams (`mhc.start = 4`) and layer `27` merges `2 -> 1` stream (`mhc.end = 28`).
  2. Keeping the `1 -> 2` entry (`layer 4`) and `2 -> 1` exit (`layer 27`) intact while repeating **interior** two-stream layers (`43` total layer calls/token vs `56` default, `cap = 2,048` on the 5 sanity problems):
     - **`early_middle_2x`** (layers `5..15` run twice, `16..26` once): **`1/5` correct** (`4/5` hit cap, `1,805.2` mean tokens, occasional mid-trace language drift).
     - **`late_middle_2x`** (layers `5..15` run once, `16..26` twice): **`4/5` correct** (`5/5` coherent, `0` loops, `1,241.0` mean tokens)! Repeating only the *second* half of the middle block recovers almost all of `n=2`'s calibration at `43` vs `56` layer steps.

### 2026-11-06 — Phase A: `vLLM` + PR `#64832` Serving & Speed Probe (`results/vllm_probe.json`)
- Built isolated `/content/vllm-venv` (`vllm==0.33.0`, `torch==2.14.0+cu130`, `transformers==5.20.2`, `ninja` for FlashInfer Blackwell JIT):
  - Stock `vllm==0.33.0` runs `Qwen/Qwen3.5-9B` at **`14.338 ms/tok`** (`69.75 tok/s` single-stream), and fails on `SenseNova-Loop-9B-Preview` (`SenseNovaLoopForConditionalGeneration` unregistered).
  - Applied open PR `vllm-project/vllm#64832` (`dry_run_rc = 0`, zero rejected hunks). Setting `max_num_seqs = 64` (needed at `n=4` because `6 + 18×4 = 78` GatedDeltaNet Mamba-cache layers fit `984` blocks, slightly under vLLM's `1024` default), **all four loop-dial settings (`n = 1, 2, 3, 4`) succeeded and produced distinct pass-dependent trajectories** (`honors_loop_settings_distinct_outputs = true`):
    - `Qwen/Qwen3.5-9B`: **`14.338 ms/tok`** (`69.75 tok/s`, `32` steps)
    - `SenseNova-Loop` `n=1`: **`14.635 ms/tok`** (`68.33 tok/s`, `32` steps — Triton Sinkhorn kernel cuts `mHC` overhead to just **`+0.30 ms/tok` / `+2.1%`**!)
    - `SenseNova-Loop` `n=2`: **`24.622 ms/tok`** (`40.61 tok/s`, `56` steps — `1.717×` Qwen latency for `1.750×` layer executions)
    - `SenseNova-Loop` `n=3`: **`34.441 ms/tok`** (`29.04 tok/s`, `80` steps — `2.402×` Qwen latency for `2.500×` layer executions)
    - `SenseNova-Loop` `n=4`: **`43.446 ms/tok`** (`23.02 tok/s`, `104` steps — `3.030×` Qwen latency for `3.250×` layer executions)
  - Because `vllm==0.33.0` + PR `#64832` supports both `SenseNova-Loop-9B-Preview` (`n=1..4`) and `Qwen/Qwen3.5-9B` on the **exact same V1 engine, FlashAttention-2 + FlashInfer GDN kernels, CUDA graphs, and `SamplingParams`**, we selected `--backend vllm` (`max_num_seqs=64`) as the unified Phase B sweep backend across all 5 conditions.

### 2026-11-06 — Phase B: `N = 200` Full Sweep (`16,384` Token Cap + Pass-2 Forced Answer)
- Executed all five conditions on the identical `--backend vllm` engine (`vllm==0.33.0` + PR `#64832`, `max_num_seqs=64`, `bfloat16` weights with FP32 `mHC` / SSM states) on Colab `G4` (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, `96 GB`):
  1. Completed `N = 50` pilot first (`data/subset_200.json[:50]`) and checked the `n = 2` cap-hit decision rule: **`0/50 (0.0%)` hit the `16,384` cap at `n = 2`** (well below the `10%` threshold that would trigger expanding to `32,768`).
  2. Expanded in resumable chunks (`50 -> 125 -> 200`) across all five conditions (`1,000` full reasoning trajectories total, saved in `results/{loop_n1,loop_n2,loop_n3,loop_n4,qwen35_9b}.jsonl`), then immediately terminated the Colab GPU session (`[colab] No active sessions found on server`).
- Even across all `200` problems, default `loop_n2` hit the `16,384` cap **`0/200 (0.0%)`** times (`max` trace length `14,896` tokens on `test/intermediate_algebra/1454.json`, median `572` tokens), whereas plain `Qwen/Qwen3.5-9B` hit the `16,384` cap on **`29/200 (14.5%)`** problems (`5,229.5` median tokens).

### 2026-11-06 — Phase C–E: Paired Stats, Plain-English Visuals, Gradio Explorer, & Colab Notebook
- Computed `results/summary.csv`, `results/budget_curve.csv`, `results/paired_stats.csv`, and `results/gained_lost_examples.json` via `scripts/stats.py`.
- Generated and visually verified all five plain-English publication charts (`assets/*.png`) with programmatic assertions confirming zero raw LaTeX tokens (`\`, `$`, `{`, `}`, `\boxed`, `\frac`) and zero config jargon (`eval_count`, `</think>`, `bf16`).
- Built `app.py` (offline replay on Mac + `--enable-live-gpu` on Colab) and recorded `assets/app_demo.gif` directly from a live local `app.py` instance via headless Chrome DevTools Protocol (`scripts/record_demo_gif.py`), visually verifying all extracted frames.
- Generated `notebooks/loop_dial_colab.ipynb` via `scripts/build_colab_notebook.py`.

### 2026-11-06 — Phase B2: `AIME 2024 + 2025` Olympiad Follow-Up Sweep (`N = 60`, `32,768` Token Cap)
- Because `loop_n2` reached **`99.5%` (`199/200`)** on `MATH-500`, only `1` unsolved problem remained for `n = 3` or `n = 4` to potentially improve upon. To test whether extra middle-layer passes help once `n = 2` has real headroom (`~73%`), we built `data/aime_2024_2025.json` (`scripts/prepare_aime.py`: `30` problems from `HuggingFaceH4/aime_2024` + `30` problems from `MathArena/aime_2025`, `sorted_unique_ids_sha256 = d712c0e465d8bf7af7890d3dc567c20a93ae187e10cef1deaa079ddd0978bd56`; also verified `MathArena/aime_2026` live on Hugging Face with `30` rows and schema `problem_idx, answer, problem`).
- Provisioned a fresh Colab `G4` runtime (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, `94.97 GiB` VRAM) with the **exact same `vllm==0.33.0` + `vllm-project/vllm#64832` engine** (`/content/vllm-venv` via `scripts/probe_vllm.py --setup-only`) and ran all 5 conditions (`loop_n2`, `qwen35_9b`, `loop_n4`, `loop_n3`, `loop_n1`; `300` olympiad trajectories total) at **`max_new_tokens = 32,768`**, `forced_max_tokens = 512`, `max_num_seqs = 16`, `max_model_len = 36,864`, `seed = 0`, saving strictly under `results/aime/` without touching any `MATH-500` file.
- Even at `n = 4` (`104` layer steps/token, `78` FP32-state GatedDeltaNet Mamba-cache layers + `26` full-attention KV layers), `max_num_seqs = 16` and `max_model_len = 36,864` fit comfortably inside `94.97 GiB` VRAM (`~66.5 GiB` KV/Mamba cache, zero OOMs or concurrency fallbacks). Immediately stopped the Colab `G4` session upon completion (`[colab] No active sessions.`).
- Computed `results/aime/{summary.csv,budget_curve.csv,paired_stats.csv,gained_lost_examples.json}`, generated and visually verified `assets/cover_5x2.png` (`2000×800`), `assets/hero_trace.png` (`1600×1080`), `assets/aime_vs_math500.png`, and `assets/aime_budget_curve.png`, extended `app.py` + `assets/app_demo.gif` with a `MATH-500` / `AIME 2024+2025` selector, and updated `notebooks/loop_dial_colab.ipynb` so `RUN_VLLM_SETUP` / `RUN_PHASE_B_LIVE` / `RUN_AIME_LIVE` automatically install `vllm==0.33.0` + PR `#64832` via `scripts/probe_vllm.py --setup-only`.

---

## Key results

### 1. `MATH-500` Headline Summary (`N = 200`, `HuggingFaceH4/MATH-500`, `16,384` Max Tokens)

| Condition | Layer Steps / Token | Solved Cleanly (`≤16k`) | 95% Wilson CI (Clean) | Solved w/ Cap Forced Answer | 95% Wilson CI (Forced) | Hit `16k` Cap | Repetition Loop | Median / Mean Tokens | Total GPU Time (`s`) | Correct / GPU-hr (Clean / Forced) |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` (`loop_n1`) | `32` | `1/200` (**0.5%**) | `[0.09%, 2.78%]` | `7/200` (**3.5%**) | `[1.71%, 7.05%]` | `72.5%` (`145`) | `33.5%` (`67`) | `16,384` / `12,362` | `1,699.9` | `2.1` / `14.8` |
| **`Loop 2 passes` (`loop_n2`, default)** | **`56`** | **`199/200` (99.5%)** | **`[97.22%, 99.91%]`** | **`199/200` (99.5%)** | **`[97.22%, 99.91%]`** | **`0.0%` (`0`)** | **`1.0%` (`2`)** | **`572` / `1,414`** | **`1,076.1`** | **`665.8` / `665.8`** |
| `Loop 3 passes` (`loop_n3`) | `80` | `187/200` (**93.5%**) | `[89.20%, 96.16%]` | `191/200` (**95.5%**) | `[91.67%, 97.61%]` | `4.5%` (`9`) | `1.0%` (`2`) | `685` / `2,035` | `1,816.8` | `370.6` / `378.5` |
| `Loop 4 passes` (`loop_n4`) | `104` | `177/200` (**88.5%**) | `[83.34%, 92.21%]` | `185/200` (**92.5%**) | `[88.00%, 95.40%]` | `9.0%` (`18`) | `5.5%` (`11`) | `856.5` / `3,036` | `2,788.4` | `228.5` / `238.9` |
| **`Plain Qwen3.5-9B` (`qwen35_9b`)** | **`32`** | **`171/200` (85.5%)** | **`[79.95%, 89.71%]`** | **`192/200` (96.0%)** | **`[92.31%, 97.96%]`** | **`14.5%` (`29`)** | **`0.0%` (`0`)** | **`5,229.5` / `7,230`** | **`1,059.0`** | **`581.3` / `652.7`** |

### 2. `MATH-500` Nested Token-Budget Curve (`N = 200`, Cleanly Solved % / Forced-at-Cap % where `B = 16,384`)

| Condition | `2,048` Tokens (Clean) | `4,096` Tokens (Clean) | `8,192` Tokens (Clean) | `16,384` Tokens (Clean) | `16,384` + Forced Answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` | `0.5%` (`78.5%` cap) | `0.5%` (`74.0%` cap) | `0.5%` (`73.5%` cap) | `0.5%` (`72.5%` cap) | `3.5%` |
| **`Loop 2 passes` (default)** | **`84.5%`** (`15.0%` cap) | **`91.0%`** (`8.5%` cap) | **`97.0%`** (`2.5%` cap) | **`99.5%`** (`0.0%` cap) | **`99.5%`** |
| `Loop 3 passes` | `81.5%` (`17.0%` cap) | `86.5%` (`11.5%` cap) | `91.0%` (`7.0%` cap) | `93.5%` (`4.5%` cap) | `95.5%` |
| `Loop 4 passes` | `71.0%` (`28.5%` cap) | `79.5%` (`19.0%` cap) | `85.5%` (`12.5%` cap) | `88.5%` (`9.0%` cap) | `92.5%` |
| **`Plain Qwen3.5-9B`** | **`14.0%`** (`86.0%` cap) | **`40.0%`** (`60.0%` cap) | **`66.0%`** (`34.0%` cap) | **`85.5%`** (`14.5%` cap) | **`96.0%`** |

*Note*: In `results/budget_curve.csv`, simulated `2k/4k/8k` forced-answer estimates also check whether an interim boxed expression appeared before truncation (`92.0%` vs `80.5%` at `2k`, `95.0%` vs `87.0%` at `4k`, `98.0%` vs `92.0%` at `8k` for `loop_n2` vs `Qwen3.5-9B`).

### 3. `MATH-500` Paired Exact Two-Sided McNemar Comparisons (`N = 200`)

| Comparison | Metric | Net Accuracy Change | Gained | Lost | Discordant Pairs | Exact McNemar `p`-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`2` passes vs Plain `Qwen3.5-9B`** | Clean (`≤16k`) | **`+14.0 pp`** | `+29` | `-1` | `30` | **`5.77 × 10⁻⁸`** |
| **`2` passes vs Plain `Qwen3.5-9B`** | Forced at cap | **`+3.5 pp`** | `+8` | `-1` | `9` | **`0.0391`** |
| `1` pass vs `2` passes (default) | Clean (`≤16k`) | **`-99.0 pp`** | `+0` | `-198` | `198` | **`4.98 × 10⁻⁶⁰`** |
| `1` pass vs `2` passes (default) | Forced at cap | **`-96.0 pp`** | `+0` | `-192` | `192` | **`3.19 × 10⁻⁵⁸`** |
| `3` passes vs `2` passes (default) | Clean (`≤16k`) | **`-6.0 pp`** | `+1` | `-13` | `14` | **`0.00183`** |
| `3` passes vs `2` passes (default) | Forced at cap | **`-4.0 pp`** | `+1` | `-9` | `10` | **`0.0215`** |
| `4` passes vs `2` passes (default) | Clean (`≤16k`) | **`-11.0 pp`** | `+0` | `-22` | `22` | **`4.77 × 10⁻⁷`** |
| `4` passes vs `2` passes (default) | Forced at cap | **`-7.0 pp`** | `+0` | `-14` | `14` | **`1.22 × 10⁻⁴`** |
| `3` passes vs Plain `Qwen3.5-9B` | Clean / Forced | `+8.0 pp` / `-0.5 pp` | `+22` / `+6` | `-6` / `-7` | `28` / `13` | `0.00372` / `1.00` |
| `4` passes vs Plain `Qwen3.5-9B` | Clean / Forced | `+3.0 pp` / `-3.5 pp` | `+15` / `+6` | `-9` / `-13` | `24` / `19` | `0.307` / `0.167` |

### 4. `MATH-500` Concrete Problem-Level Examples (`results/gained_lost_examples.json`)
- **The single problem out of 200 where `loop_n2` missed (`test/prealgebra/2066.json`, Level 5 Prealgebra, gold `22`)**:
  - `loop_n2` finished in `487` tokens with an off-by-one boundary count (`23`), whereas both `Qwen3.5-9B` (`4,694` tokens -> `22`) and `loop_n3` (`775` tokens -> `22`) double-checked the endpoint and got `22`. That single item is the **only** problem gained by `loop_n3` (`+1` gained vs `-13` lost) and the **only** problem where `Qwen3.5-9B` beat `loop_n2` (`-1` lost vs `+29` gained).
- **Where `loop_n2` beats `Qwen3.5-9B` (`+29` clean gains, `+8` even after forced answer)**:
  - On `test/intermediate_algebra/1454.json` (Level 5, gold `70√2`), `loop_n2` spots the geometric distance rewrite and finishes cleanly in `14,896` tokens, while `Qwen3.5-9B` grinds algebraic partial derivatives until hitting the `16,384` cap with no answer even after Pass-2 forcing.
  - On `test/algebra/2058.json` (Level 4, gold `13`) and `test/precalculus/1199.json` (Level 4, gold `π`), `loop_n2` solves cleanly in `788` and `1,397` tokens respectively, whereas `Qwen3.5-9B` exhausts all `16,384` tokens mid-thought.
- **Where over-looping (`n = 3, 4`) degrades `loop_n2`**:
  - Two distinct failure modes appear when turning the dial past `2`: (a) **arithmetic/sign drift** on short problems, such as `test/counting_and_probability/119.json` (gold `-125`: `n=2` gets `-125` in `755` tokens, while both `n=3` and `n=4` output `-125/2`; rendered side-by-side in `assets/hero_trace.png`) and `test/precalculus/43.json` (Level 1, gold `(15, -29)`: `n=2` gets `(15, -29)` in `214` tokens, while `n=4` outputs `(15, -35)`), and (b) **self-doubt spirals into the `16,384` cap** (`0%` cap hits at `n=2` -> `4.5%` at `n=3` -> `9.0%` at `n=4`, plus `5.5%` 8-gram repetition loops at `n=4`).

### 5. `AIME 2024 + 2025` Olympiad Headline Summary (`N = 60` = `30 + 30`, `32,768` Max Tokens, `results/aime/summary.csv`)

| Condition | Layer Steps / Token | Solved Cleanly (`≤32k`) | 95% Wilson CI (Clean) | Solved w/ Cap Forced Answer | 95% Wilson CI (Forced) | Hit `32k` Cap | Repetition Loop | Median / Mean Tokens | Total GPU Time (`s`) | Correct / GPU-hr (Clean / Forced) |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` (`loop_n1`) | `32` | `0/60` (**0.00%**) | `[0.00%, 6.02%]` | `0/60` (**0.00%**) | `[0.00%, 6.02%]` | `80.0%` (`48`) | `31.67%` (`19`) | `32,768.0` / `26,568.8` | `2,396.6` | `0.0` / `0.0` |
| **`Loop 2 passes` (`loop_n2`, default)** | **`56`** | **`44/60` (73.33%)** | **`[60.99%, 82.87%]`** | **`46/60` (76.67%)** | **`[64.56%, 85.56%]`** | **`20.0%` (`12`)** | **`8.33%` (`5`)** | **`7,843.5` / `12,660.1`** | **`2,201.3`** | **`72.0` / `75.2`** |
| `Loop 3 passes` (`loop_n3`) | `80` | `40/60` (**66.67%**) | `[54.06%, 77.27%]` | `47/60` (**78.33%**) | `[66.38%, 86.88%]` | `30.0%` (`18`) | `0.00%` (`0`) | `10,786.5` / `15,340.0` | `3,529.1` | `40.8` / `47.9` |
| `Loop 4 passes` (`loop_n4`) | `104` | `23/60` (**38.33%**) | `[27.09%, 50.98%]` | `32/60` (**53.33%**) | `[40.89%, 65.37%]` | `50.0%` (`30`) | `26.67%` (`16`) | `32,061.5` / `20,608.5` | `6,285.2` | `13.2` / `18.3` |
| **`Plain Qwen3.5-9B` (`qwen35_9b`)** | **`32`** | **`38/60` (63.33%)** | **`[50.68%, 74.38%]`** | **`46/60` (76.67%)** | **`[64.56%, 85.56%]`** | **`35.0%` (`21`)** | **`0.00%` (`0`)** | **`27,267.5` / `24,607.4`** | **`1,979.9`** | **`69.1` / `83.6`** |

- **Per-year consistency (`AIME 2024` `N = 30` vs `AIME 2025` `N = 30`)**:
  - `loop_n1`: `0/30` (`0.0%`) clean & forced on 2024 | `0/30` (`0.0%`) clean & forced on 2025.
  - **`loop_n2`**: **`22/30` (`73.3%`)** clean, **`23/30` (`76.7%`)** forced on 2024 | **`22/30` (`73.3%`)** clean, **`23/30` (`76.7%`)** forced on 2025 (identical across both competition years).
  - `loop_n3`: `21/30` (`70.0%`) clean, `25/30` (`83.3%`) forced on 2024 | `19/30` (`63.3%`) clean, `22/30` (`73.3%`) forced on 2025.
  - `loop_n4`: `12/30` (`40.0%`) clean, `16/30` (`53.3%`) forced on 2024 | `11/30` (`36.7%`) clean, `16/30` (`53.3%`) forced on 2025.
  - `qwen35_9b`: `22/30` (`73.3%`) clean, `25/30` (`83.3%`) forced on 2024 | `16/30` (`53.3%`) clean, `21/30` (`70.0%`) forced on 2025.

### 6. `AIME 2024 + 2025` Nested Token-Budget Curve (`N = 60`, `results/aime/budget_curve.csv`)

| Condition | `2,048` Tokens (Clean) | `4,096` Tokens (Clean) | `8,192` Tokens (Clean) | `16,384` Tokens (Clean) | `32,768` Tokens (Clean) | `32,768` + Forced Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` | `0.0%` (`85.0%` cap) | `0.0%` (`81.7%` cap) | `0.0%` (`80.0%` cap) | `0.0%` (`80.0%` cap) | `0.0%` (`80.0%` cap) | `0.0%` |
| **`Loop 2 passes` (default)** | **`16.7%`** (`83.3%` cap) | **`38.3%`** (`60.0%` cap) | **`46.7%`** (`50.0%` cap) | **`61.7%`** (`31.7%` cap) | **`73.3%`** (`20.0%` cap) | **`76.7%`** |
| `Loop 3 passes` | `11.7%` (`88.3%` cap) | `33.3%` (`66.7%` cap) | `43.3%` (`56.7%` cap) | `60.0%` (`40.0%` cap) | `66.7%` (`30.0%` cap) | `78.3%` |
| `Loop 4 passes` | `6.7%` (`90.0%` cap) | `18.3%` (`78.3%` cap) | `23.3%` (`70.0%` cap) | `31.7%` (`58.3%` cap) | `38.3%` (`50.0%` cap) | `53.3%` |
| **`Plain Qwen3.5-9B`** | **`0.0%`** (`100.0%` cap) | **`0.0%`** (`100.0%` cap) | **`5.0%`** (`95.0%` cap) | **`25.0%`** (`75.0%` cap) | **`63.3%`** (`35.0%` cap) | **`76.7%`** |

### 7. `AIME 2024 + 2025` Paired Exact Two-Sided McNemar Comparisons (`N = 60`, `results/aime/paired_stats.csv`)

| Comparison | Metric | Net Accuracy Change | Gained | Lost | Discordant Pairs | Exact McNemar `p`-value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`2` passes vs Plain `Qwen3.5-9B`** | Clean (`≤32k`) | **`+10.00 pp`** | `+12` | `-6` | `18` | `0.2379` |
| **`2` passes vs Plain `Qwen3.5-9B`** | Forced at cap | **`0.00 pp`** | `+6` | `-6` | `12` | `1.0000` |
| `1` pass vs `2` passes (default) | Clean (`≤32k`) | **`-73.33 pp`** | `+0` | `-44` | `44` | **`1.14 × 10⁻¹³`** |
| `1` pass vs `2` passes (default) | Forced at cap | **`-76.67 pp`** | `+0` | `-46` | `46` | **`2.84 × 10⁻¹⁴`** |
| `3` passes vs `2` passes (default) | Clean (`≤32k`) | **`-6.67 pp`** | `+6` | `-10` | `16` | `0.4545` |
| `3` passes vs `2` passes (default) | Forced at cap | **`+1.67 pp`** | `+6` | `-5` | `11` | `1.0000` |
| `4` passes vs `2` passes (default) | Clean (`≤32k`) | **`-35.00 pp`** | `+2` | `-23` | `25` | **`1.94 × 10⁻⁵`** |
| `4` passes vs `2` passes (default) | Forced at cap | **`-23.33 pp`** | `+3` | `-17` | `20` | **`0.00258`** |
| `3` passes vs Plain `Qwen3.5-9B` | Clean / Forced | `+3.33 pp` / `+1.67 pp` | `+7` / `+6` | `-5` / `-5` | `12` / `11` | `0.774` / `1.00` |
| `4` passes vs Plain `Qwen3.5-9B` | Clean / Forced | `-25.00 pp` / `-23.33 pp` | `+3` / `+3` | `-18` / `-17` | `21` / `20` | **`0.00149`** / **`0.00258`** |

### 8. Concrete `AIME 2024 + 2025` Olympiad Examples (`results/aime/gained_lost_examples.json`)
- **Where `3` passes genuinely rescues `2`-pass arithmetic slips (`+6` clean gains), yet loses `-10` to cap timeouts**:
  - Unlike `MATH-500` (where `2` passes missed only 1 item total), on AIME `loop_n3` fixes several concrete `loop_n2` calculation errors—for instance `aime_2024/74` (gold `480`: `n=2` outputs `4` in `5,256` tokens, while `n=3` outputs `480` in `5,099` tokens), `aime_2024/87` (gold `699`: `n=2` gets `701` vs `n=3` `699`), `aime_2024/89` (gold `902`: `n=2` gets `962` vs `n=3` `902`), and `aime_2025/24` (gold `149`: `n=2` off-by-one `150` vs `n=3` `149`).
  - However, `loop_n3` simultaneously pushes `10` problems that `loop_n2` solved cleanly into `32,768`-token timeouts (`8` of the `10` losses hit the `32k` cap, e.g., `aime_2024/70`, `aime_2024/82`, `aime_2025/12`, `aime_2025/25`) or arithmetic drift (`aime_2025/7`, `aime_2025/23`). After Pass-2 forcing rescues `7` of `loop_n3`'s `18` capped traces, `loop_n3` (`47/60`, `78.3%`) and `loop_n2` (`46/60`, `76.7%`) are statistically indistinguishable (`p = 1.00`) while `loop_n3` costs `1.60×` more GPU seconds (`3,529 s` vs `2,201 s`).
- **Why `4` passes collapses by `-35.0 pp` on AIME (`23/60` vs `44/60`, `p = 1.94 × 10⁻⁵`)**:
  - Over long olympiad derivations (`20,608` mean tokens), running `104` layer executions per token compounds activation drift: `50.0%` (`30/60`) hit the `32,768` cap and `26.7%` (`16/60`) fall into 8-gram repetition loops, losing `23` problems that `2` passes solved cleanly (including short problems like `aime_2024/60`, gold `204`, where `n=2` solves in `1,161` tokens while `n=4` outputs `192`).
- **Where `loop_n2` vs `Qwen3.5-9B` trade olympiad wins (`+12` clean gains vs `-6` clean losses; `46/60` vs `46/60` forced tie)**:
  - All `12` clean gains by `loop_n2` over `Qwen3.5-9B` are problems where `Qwen3.5-9B`'s ultra-long verbal monologue (`27,267.5` median tokens!) hit the `32,768` cap while `loop_n2` finished cleanly inside `8k–28k` tokens (e.g., `aime_2024/62` octagon coloring, gold `371`: `loop_n2` `27,962` tokens `✓ 371` vs `Qwen3.5-9B` `32,768` cap `✗ 257`; `aime_2024/64` gold `110` in `8,543` tokens; `aime_2025/18` gold `82` in `8,234` tokens). Conversely, `Qwen3.5-9B`'s exhaustive `25k–28k` verbal checking catches `3` `loop_n2` calculation slips (`aime_2024/74`, `87`, `89`) and `3` `loop_n2` cap hits (`aime_2024/61`, `73`, `85`), resulting in an exact `46/60` (`76.7%`) tie once cap-forced answers are included.

---

## Visuals

1. [`assets/cover_5x2.png`](assets/cover_5x2.png) — Editorial `2000×800` (`5:2`) cover contrasting `1 pass` (`0/60`), `2 passes — default` (`44/60`, `73.3%`), and `4 passes` (`23/60`, `38.3%`) on `AIME 2024 + 2025` alongside plain `Qwen3.5-9B` (`38/60`, `63.3%`).
2. [`assets/hero_trace.png`](assets/hero_trace.png) — Side-by-side real logged reasoning excerpts (`MATH-500` `test/counting_and_probability/119.json`, gold `-125`) showing `1 pass` looping to `16,384` tokens (`-250`), `2 passes` solving cleanly in `755` tokens (`-125`), and `4 passes` slipping on the final factor (`-125/2`).
3. [`assets/loop_architecture.png`](assets/loop_architecture.png) — Plain-English diagram contrasting `SenseNova-Loop-9B`'s 4 entry layers + 24 looped middle layers (`1–4` passes) + 4 exit layers against plain `Qwen3.5-9B`'s single 32-layer pass.
4. [`assets/accuracy_vs_passes.png`](assets/accuracy_vs_passes.png) — `MATH-500` (`N=200`) accuracy and 95% Wilson confidence intervals across `1, 2, 3, 4` middle-layer passes vs plain `Qwen3.5-9B`, showing both clean (`≤16,384` tokens) and cap-forced answers.
5. [`assets/budget_curve.png`](assets/budget_curve.png) — `MATH-500` nested token-budget curve (`2,048`, `4,096`, `8,192`, `16,384` tokens) illustrating how 2-pass latent looping solves `84.5%` cleanly by `2,048` tokens while plain `Qwen3.5-9B` is at `14.0%`.
6. [`assets/speed_vs_accuracy.png`](assets/speed_vs_accuracy.png) — Throughput vs accuracy frontier showing why default 2-pass `SenseNova-Loop-9B` delivers the highest rate of correct answers per GPU-hour (`666` vs `581` clean / `653` forced for `Qwen3.5-9B`).
7. [`assets/paired_changes.png`](assets/paired_changes.png) — Paired problem-by-problem gains and losses (`MATH-500`, `N=200`) with exact two-sided McNemar `p`-values when dialing passes down (`1`), up (`3, 4`), or switching from plain `Qwen3.5-9B` to `2` passes.
8. [`assets/aime_vs_math500.png`](assets/aime_vs_math500.png) — Grouped comparison of `MATH-500` (`N=200`, `16k` cap) vs `AIME 2024 + 2025` (`N=60`, `32k` cap) clean accuracy with 95% Wilson CIs across all 5 conditions.
9. [`assets/aime_budget_curve.png`](assets/aime_budget_curve.png) — `AIME 2024 + 2025` nested token-budget curve (`2,048` to `32,768` tokens) showing `loop_n2` reaching `61.7%` at `16k` while `Qwen3.5-9B` sits at `25.0%` before catching up between `16k` and `32k`.
10. [`assets/app_demo.gif`](assets/app_demo.gif) — Live screen recording of `app.py` sliding the loop dial across `1–4` passes, expanding step-by-step reasoning traces, filtering disagreements, and switching live between `MATH-500` and `AIME 2024+2025`.

---

## Surprises and limitations

1. **Biggest surprise — latent depth substitutes for chain-of-thought length on *both* `MATH-500` (`9.1×` shorter median trace) and `AIME` (`3.5×` shorter median trace)**:
   - On `MATH-500`, `loop_n2` finishes in **`572` median tokens** vs **`5,229.5`** for `Qwen3.5-9B`, winning clean answers per GPU-hour (`665.8` vs `581.3`). On `AIME 2024 + 2025`, the same compression holds: `loop_n2` finishes in **`7,843.5` median tokens** (`12,660` mean) vs **`27,267.5` median tokens** (`24,607` mean) for `Qwen3.5-9B`! At `16,384` tokens, `loop_n2` already solves **`61.7%` (`37/60`)** of AIME cleanly while `Qwen3.5-9B` has only finished **`25.0%` (`15/60`)**; `Qwen3.5-9B` needs `32,768` tokens *plus* Pass-2 forced-answer rescue on its `21` capped runs to tie `loop_n2` at `46/60` (`76.7%`).
2. **Moving off the `MATH-500` ceiling proves the `n = 4` degradation is real—and much steeper on long derivations (`-35.0 pp`, `p = 1.94 × 10⁻⁵`)**:
   - On `MATH-500`, one might wonder whether `n = 3` (`93.5%`) and `n = 4` (`88.5%`) trailed `n = 2` (`99.5%`) simply because `n = 2` had zero upward headroom (`199/200`). `AIME 2024 + 2025` settles that question: with `16` unsolved problems of headroom above `n = 2` (`44/60`, `73.3%`), `n = 3` rescues `6` slips but loses `10` to timeouts (`66.7%` clean, `78.3%` forced — a statistical tie at `1.6×` higher GPU time), and `n = 4` collapses to **`38.3%` clean (`23/60`, `26.7%` repetition loops)** because fixed-count (`eval_count = 2`) training leaves `104`-step activations uncalibrated over `20k+`-token contexts.
3. **Late middle layers (`16..26`) carry most of the second-pass calibration**:
   - Our Phase A partial-loop probe (`results/partial_layer_probe.json`) showed that repeating only layers `16..26` (`43` total layer steps) preserved `4/5` accuracy with zero loops, whereas repeating only layers `5..15` (`43` layer steps) dropped to `1/5`.
4. **Silent quantization trap on consumer Macs / llama.cpp**:
   - Both `hermitdave/SenseNova-Loop-9B-Preview-MLX-slq-4.5bpw` and `AtomicChat/SenseNova-Loop-9B-Preview-GGUF` export the 32 physical layers as a flat single-pass `Qwen3.5` graph and discard the `mHC` Sinkhorn routers (`results/mac_mlx_check.json`). Users downloading those community builds are unknowingly running a quantized `n = 1` model without `mHC`.
5. **Scope & methodological limitations**:
   - Single sample per problem (`seed = 0`) at `temperature = 1.0` (`top_p = 0.95, top_k = 20, presence_penalty = 1.5`). On `N = 60` AIME problems, large effects like `n = 4` vs `n = 2` (`-35.0 pp`, `p = 1.94 × 10⁻⁵`) and `n = 1` vs `n = 2` (`-73.3 pp`, `p = 1.14 × 10⁻¹³`) are highly significant, whereas ~`6–10 pp` differences (`loop_n2` vs `Qwen3.5-9B` clean `+10.0 pp`, `p = 0.238`; `loop_n3` vs `loop_n2` clean `-6.7 pp`, `p = 0.454`) would require `Pass@k` / multi-seed runs (`k = 8..16`) or pooling additional olympiad years to narrow confidence intervals below `±5 pp`.
   - `AIME` ran at `max_num_seqs = 16` (`max_model_len = 36,864`) on the 96 GB Blackwell `G4` so `n = 4`'s 78 FP32-state GatedDeltaNet layers fit `36k` context with zero OOMs; at `batch = 16`, `loop_n2` (`72.0` clean answers/GPU-hr) and `Qwen3.5-9B` (`69.1` clean answers/GPU-hr) achieve nearly identical clean throughput efficiency, while `Qwen3.5-9B` edges ahead on forced answers/GPU-hr (`83.6` vs `75.2`).

---

## Open questions / next steps

1. **Extend olympiad sample size with `MathArena/aime_2026` (`30` fresh 2026 problems, verified live on Hugging Face with schema `problem_idx, answer, problem`) and multi-seed `Pass@8` / `Maj@8`** to test whether `loop_n3`'s `+6` rescued calculation slips can be combined with `loop_n2` via confidence-gated fallback (run `n = 3` only when `n = 2` hits cap or exhibits high answer entropy).
2. **Fine-tune with randomized pass counts (`counts ∈ {1, 2, 3, 4}`)** or lightweight LoRA adapters per pass count to test whether explicit multi-depth training unlocks true monotonic test-time compute scaling (`acc(1) < acc(2) < acc(3) < acc(4)`).
3. **Port `execution_plan` + two-stream `mHC` Sinkhorn routing to `mlx-lm` and `llama.cpp`** so 16 GB Apple Silicon Macs can run genuine 2-pass `SenseNova-Loop-9B` locally instead of broken single-pass exports.


