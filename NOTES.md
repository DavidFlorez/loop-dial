# Research Notes — `loop-dial`

## Goal

`sensenova/SenseNova-Loop-9B-Preview` is a 9B-parameter hybrid reasoning model built on the 32-layer `Qwen3.5` text-and-vision backbone (`hidden_size=4096`, 8 blocks of `3 × linear_attention + 1 × full_attention`), modified so its middle 24 decoder layers (`4..27`, 0-indexed) run **twice** per token by default (`4 + 24×2 + 4 = 56` virtual layer executions) with two-stream Manifold-Constrained Hyper-Connections (`mHC`, 20 Sinkhorn iterations) and separate KV / FP32 Mamba SSM caches per pass.

This repository tests one plain-English question on `HuggingFaceH4/MATH-500`:
> **What happens at inference time when you turn SenseNova-Loop-9B-Preview's middle-layer "loop dial" to `n = 1, 2, 3, or 4` passes (`32, 56, 80, or 104` layer executions per token), and is the looped model better than just running plain `Qwen/Qwen3.5-9B` (`32` layers once)?**

Every number in this repository comes from a real, logged model run stored under `results/`.

---

## Setup

- **Models compared**:
  1. `sensenova/SenseNova-Loop-9B-Preview` at `n ∈ {1, 2, 3, 4}` passes of layers `4..27` (`32, 56, 80, 104` total decoder-layer executions per token; `n=2` is the checkpoint's trained default).
  2. `Qwen/Qwen3.5-9B` (`32` decoder layers run once; non-looped sibling with the same 32-layer hybrid GatedDeltaNet + gated full-attention dimensions, same `248,320` vocabulary, and same `<think>...</think>` template).
- **Benchmark & deterministic subset**:
  - `HuggingFaceH4/MATH-500` (`test` split, 500 competition math problems).
  - Fixed `seed = 0`, deterministic nested subset `S_50 ⊂ S_200` saved in `data/subset_200.json` using the exact `load_math500_subset(seed=0)` sampling rule from `DavidFlorez/watch-the-loop-break`.
- **Prompt & decoding parameters** (identical across all conditions):
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
  - Long-budget generation per condition (`max_new_tokens = 16,384`; raised to `32,768` if `>10%` of problems hit the cap at `n=2` and GPU budget allows).
  - Nested prefix truncation at `B ∈ {2,048, 4,096, 8,192, 16,384}` (plus `32,768` if run) tokens on saved token-ID prefixes.
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

---

## Key results

### 1. Headline Summary (`N = 200`, `HuggingFaceH4/MATH-500`, `16,384` Max Tokens)

| Condition | Layer Steps / Token | Solved Cleanly (`≤16k`) | 95% Wilson CI (Clean) | Solved w/ Cap Forced Answer | 95% Wilson CI (Forced) | Hit `16k` Cap | Repetition Loop | Median / Mean Tokens | Total GPU Time (`s`) | Correct / GPU-hr (Clean / Forced) |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` (`loop_n1`) | `32` | `1/200` (**0.5%**) | `[0.09%, 2.78%]` | `7/200` (**3.5%**) | `[1.71%, 7.05%]` | `72.5%` (`145`) | `33.5%` (`67`) | `16,384` / `12,362` | `1,699.9` | `2.1` / `14.8` |
| **`Loop 2 passes` (`loop_n2`, default)** | **`56`** | **`199/200` (99.5%)** | **`[97.22%, 99.91%]`** | **`199/200` (99.5%)** | **`[97.22%, 99.91%]`** | **`0.0%` (`0`)** | **`1.0%` (`2`)** | **`572` / `1,414`** | **`1,076.1`** | **`665.8` / `665.8`** |
| `Loop 3 passes` (`loop_n3`) | `80` | `187/200` (**93.5%**) | `[89.20%, 96.16%]` | `191/200` (**95.5%**) | `[91.67%, 97.61%]` | `4.5%` (`9`) | `1.0%` (`2`) | `685` / `2,035` | `1,816.8` | `370.6` / `378.5` |
| `Loop 4 passes` (`loop_n4`) | `104` | `177/200` (**88.5%**) | `[83.34%, 92.21%]` | `185/200` (**92.5%**) | `[88.00%, 95.40%]` | `9.0%` (`18`) | `5.5%` (`11`) | `856.5` / `3,036` | `2,788.4` | `228.5` / `238.9` |
| **`Plain Qwen3.5-9B` (`qwen35_9b`)** | **`32`** | **`171/200` (85.5%)** | **`[79.95%, 89.71%]`** | **`192/200` (96.0%)** | **`[92.31%, 97.96%]`** | **`14.5%` (`29`)** | **`0.0%` (`0`)** | **`5,229.5` / `7,230`** | **`1,059.0`** | **`581.3` / `652.7`** |

### 2. Nested Token-Budget Curve (`N = 200`, Cleanly Solved % / Forced-at-Cap % where `B = 16,384`)

| Condition | `2,048` Tokens (Clean) | `4,096` Tokens (Clean) | `8,192` Tokens (Clean) | `16,384` Tokens (Clean) | `16,384` + Forced Answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Loop 1 pass` | `0.5%` (`78.5%` cap) | `0.5%` (`74.0%` cap) | `0.5%` (`73.5%` cap) | `0.5%` (`72.5%` cap) | `3.5%` |
| **`Loop 2 passes` (default)** | **`84.5%`** (`15.0%` cap) | **`91.0%`** (`8.5%` cap) | **`97.0%`** (`2.5%` cap) | **`99.5%`** (`0.0%` cap) | **`99.5%`** |
| `Loop 3 passes` | `81.5%` (`17.0%` cap) | `86.5%` (`11.5%` cap) | `91.0%` (`7.0%` cap) | `93.5%` (`4.5%` cap) | `95.5%` |
| `Loop 4 passes` | `71.0%` (`28.5%` cap) | `79.5%` (`19.0%` cap) | `85.5%` (`12.5%` cap) | `88.5%` (`9.0%` cap) | `92.5%` |
| **`Plain Qwen3.5-9B`** | **`14.0%`** (`86.0%` cap) | **`40.0%`** (`60.0%` cap) | **`66.0%`** (`34.0%` cap) | **`85.5%`** (`14.5%` cap) | **`96.0%`** |

*Note*: In `results/budget_curve.csv`, simulated `2k/4k/8k` forced-answer estimates also check whether an interim boxed expression appeared before truncation (`92.0%` vs `80.5%` at `2k`, `95.0%` vs `87.0%` at `4k`, `98.0%` vs `92.0%` at `8k` for `loop_n2` vs `Qwen3.5-9B`).

### 3. Paired Exact Two-Sided McNemar Comparisons (`N = 200`)

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

### 4. Concrete Problem-Level Examples (`results/gained_lost_examples.json`)
- **The single problem out of 200 where `loop_n2` missed (`test/prealgebra/2066.json`, Level 5 Prealgebra, gold `22`)**:
  - `loop_n2` finished in `487` tokens with an off-by-one boundary count (`23`), whereas both `Qwen3.5-9B` (`4,694` tokens -> `22`) and `loop_n3` (`775` tokens -> `22`) double-checked the endpoint and got `22`. That single item is the **only** problem gained by `loop_n3` (`+1` gained vs `-13` lost) and the **only** problem where `Qwen3.5-9B` beat `loop_n2` (`-1` lost vs `+29` gained).
- **Where `loop_n2` beats `Qwen3.5-9B` (`+29` clean gains, `+8` even after forced answer)**:
  - On `test/intermediate_algebra/1454.json` (Level 5, gold `70√2`), `loop_n2` spots the geometric distance rewrite and finishes cleanly in `14,896` tokens, while `Qwen3.5-9B` grinds algebraic partial derivatives until hitting the `16,384` cap with no answer even after Pass-2 forcing.
  - On `test/algebra/2058.json` (Level 4, gold `13`) and `test/precalculus/1199.json` (Level 4, gold `π`), `loop_n2` solves cleanly in `788` and `1,397` tokens respectively, whereas `Qwen3.5-9B` exhausts all `16,384` tokens mid-thought.
- **Where over-looping (`n = 3, 4`) degrades `loop_n2`**:
  - Two distinct failure modes appear when turning the dial past `2`: (a) **arithmetic/sign drift** on short problems, such as `test/counting_and_probability/119.json` (gold `-125`: `n=2` gets `-125` in `755` tokens, while both `n=3` and `n=4` output `-125/2`) and `test/precalculus/43.json` (Level 1, gold `(15, -29)`: `n=2` gets `(15, -29)` in `214` tokens, while `n=4` outputs `(15, -35)`), and (b) **self-doubt spirals into the `16,384` cap** (`0%` cap hits at `n=2` -> `4.5%` at `n=3` -> `9.0%` at `n=4`, plus `5.5%` 8-gram repetition loops at `n=4`).

---

## Visuals

1. [`assets/loop_architecture.png`](assets/loop_architecture.png) — Plain-English diagram contrasting `SenseNova-Loop-9B`'s 4 entry layers + 24 looped middle layers (`1–4` passes) + 4 exit layers against plain `Qwen3.5-9B`'s single 32-layer pass.
2. [`assets/accuracy_vs_passes.png`](assets/accuracy_vs_passes.png) —MATH-500 (`N=200`) accuracy and 95% Wilson confidence intervals across `1, 2, 3, 4` middle-layer passes vs plain `Qwen3.5-9B`, showing both clean (`≤16,384` tokens) and cap-forced answers.
3. [`assets/budget_curve.png`](assets/budget_curve.png) — Nested token-budget curve (`2,048`, `4,096`, `8,192`, `16,384` tokens) illustrating how 2-pass latent looping solves `84.5%` cleanly by `2,048` tokens while plain `Qwen3.5-9B` is at `14.0%`.
4. [`assets/speed_vs_accuracy.png`](assets/speed_vs_accuracy.png) — Throughput vs accuracy frontier showing why default 2-pass `SenseNova-Loop-9B` delivers the highest rate of correct answers per GPU-hour (`666` vs `581` clean / `653` forced for `Qwen3.5-9B`).
5. [`assets/paired_changes.png`](assets/paired_changes.png) — Paired problem-by-problem gains and losses (`N=200`) with exact two-sided McNemar `p`-values when dialing passes down (`1`), up (`3, 4`), or switching from plain `Qwen3.5-9B` to `2` passes.
6. [`assets/app_demo.gif`](assets/app_demo.gif) — Live screen recording of `app.py` sliding the loop dial across `1–4` passes, expanding step-by-step reasoning traces, and filtering disagreements against plain `Qwen3.5-9B`.

---

## Surprises and limitations

1. **Biggest surprise — latent depth substitutes for chain-of-thought length (`9.1×` shorter median thinking trace)**:
   - On single-stream decoding, running `56` layer executions per token (`n=2`) is `1.72×` slower per token than `Qwen3.5-9B` (`24.6` vs `14.3 ms/tok`), and under continuous batching (`max_num_seqs=64`) its extra per-pass KV + FP32 GatedDeltaNet caches make aggregate token throughput `5.2×` lower (`262.9` vs `1,365.4 tok/s`). Yet `loop_n2` still beats `Qwen3.5-9B` on **correct answers per GPU-hour (`665.8` vs `581.3` strict, `665.8` vs `652.7` forced)** because `loop_n2` finishes in a median of **`572` tokens** (`1,414` mean) vs **`5,229.5` median tokens** (`7,230` mean) for `Qwen3.5-9B`! Doing two latent passes per token lets the model compress multi-step algebra into dramatically shorter verbal chains of thought.
2. **Test-time depth extrapolation fails without randomized-loop training**:
   - `SenseNova-Loop-9B-Preview` is a **fixed-count (`eval_count = 2`)** checkpoint, not an adaptive/randomized-depth checkpoint. Turning the dial to `1` pass collapses accuracy to `0.5%` (`3.5%` forced) because exit layers `28..31` receive an unrefined hidden state; turning it to `3` or `4` passes pushes hidden states past the trained 2-pass attractor (`93.5%` -> `88.5%` clean; McNemar `p = 0.0018` and `p = 4.8 × 10⁻⁷`).
3. **Late middle layers (`16..26`) carry most of the second-pass calibration**:
   - Our Phase A partial-loop probe (`results/partial_layer_probe.json`) showed that repeating only layers `16..26` (`43` total layer steps) preserved `4/5` accuracy with zero loops, whereas repeating only layers `5..15` (`43` layer steps) dropped to `1/5`.
4. **Silent quantization trap on consumer Macs / llama.cpp**:
   - Both `hermitdave/SenseNova-Loop-9B-Preview-MLX-slq-4.5bpw` and `AtomicChat/SenseNova-Loop-9B-Preview-GGUF` export the 32 physical layers as a flat single-pass `Qwen3.5` graph and discard the `mHC` Sinkhorn routers (`results/mac_mlx_check.json`). Users downloading those community builds are unknowingly running a quantized `n = 1` model without `mHC`.
5. **Scope & methodological limitations**:
   - Single sample per problem (`seed = 0`) at `temperature = 1.0` (`top_p = 0.95, top_k = 20, presence_penalty = 1.5`); paired McNemar `p`-values account for sample variance across the `N = 200` problems, though multi-seed `Pass@k` / `Maj@k` would further sharpen tight comparisons like `n = 2` vs forced `Qwen3.5-9B` (`199/200` vs `192/200`, `p = 0.039`).
   - Near-ceiling accuracy of `loop_n2` (`99.5%`) on `MATH-500` leaves limited headroom (`1` unsolved item out of `200`) to detect whether `n = 3` or `n = 4` could help on harder olympiad benchmarks (`AIME 2025/2026`, `HMMT`).
   - Throughput was measured on a 96 GB Blackwell `G4` with `max_num_seqs = 64` (chosen so `n = 4`'s 78 GDN cache layers fit cleanly); on larger batch sizes, `Qwen3.5-9B`'s smaller per-sequence cache footprint would widen its raw `tok/s` lead, though `loop_n2`'s `5.1×` lower mean token count remains decisive at interactive (`batch = 1..16`) concurrency.

---

## Open questions / next steps

1. **Test harder olympiad sets where `n = 2` is well below ceiling** (`AIME 2024–2026`, `HMMT`, `LiveCodeBench`) to test whether `n = 3` gains problems once `n = 2` sits around `60–80%` instead of `99.5%`.
2. **Fine-tune with randomized pass counts (`counts ∈ {1, 2, 3, 4}`)** or lightweight LoRA adapters per pass count to test whether explicit multi-depth training unlocks true monotonic test-time compute scaling (`acc(1) < acc(2) < acc(3) < acc(4)`).
3. **Port `execution_plan` + two-stream `mHC` Sinkhorn routing to `mlx-lm` and `llama.cpp`** so 16 GB Apple Silicon Macs can run genuine 2-pass `SenseNova-Loop-9B` locally instead of broken single-pass exports.

