# Turn the Loop Dial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidFlorez/loop-dial/blob/main/notebooks/loop_dial_colab.ipynb)

**`SenseNova-Loop-9B-Preview` runs its middle 24 layers twice per token (`56` layer steps instead of `32`). What happens if you turn that "loop dial" to `1, 2, 3, or 4` passes at test time — and is it actually better than just running plain `Qwen3.5-9B`?**

---

## The Short Answer (`200` MATH-500 Problems, `16,384`-Token Cap)

1. **Yes — the default `2`-pass loop beats plain `Qwen3.5-9B` on both accuracy *and* correct answers per GPU-hour.**
   - **`99.5%` vs `85.5%`** solved cleanly inside `16,384` tokens (`199/200` vs `171/200`, paired McNemar `p = 5.8 × 10⁻⁸`), and **`99.5%` vs `96.0%`** even after giving capped traces a 512-token forced-answer rescue (`p = 0.039`).
   - **Why it also wins on GPU efficiency (`666` vs `581` clean correct answers/GPU-hr):** Although running `56` layer steps per token is slower *per token* than `Qwen3.5-9B`'s `32` steps, looping silently inside the network lets the model think in **`9.1×` fewer median words/tokens** (`572` median tokens vs `5,230` for `Qwen3.5-9B`, and `0%` cap hits vs `14.5%`). Latent depth substitutes for long verbal chains of thought.
2. **No — turning the loop dial away from `2` passes *without retraining* hurts immediately in both directions.**
   - **`1` pass (`32` steps) collapses to `0.5%` clean / `3.5%` forced** (`p = 5.0 × 10⁻⁶⁰`): `72.5%` of runs hit the `16,384`-token cap and `33.5%` spiral into 8-gram repetition loops because the final 4 exit layers receive an unfinished internal draft.
   - **`3` passes (`80` steps) drops to `93.5%` clean / `95.5%` forced** (`-6.0 pp` clean, `p = 0.0018`), and **`4` passes (`104` steps) drops to `88.5%` clean / `92.5%` forced** (`-11.0 pp` clean, `p = 4.8 × 10⁻⁷`). Because the checkpoint was trained exclusively at `2` passes, extra passes push activations past their calibrated sweet spot — introducing arithmetic slips and self-doubt loops while burning `1.7×–2.6×` more GPU time.

![How the Loop Dial works](assets/loop_architecture.png)

---

## Results at a Glance

Every number below comes directly from logged runs in [`results/`](results/) (`seed = 0`, identical sampling across all 5 conditions on the same `vLLM` + Triton engine on an NVIDIA RTX PRO 6000 Blackwell 96 GB GPU):

| Condition | Layer Steps / Token | Solved Cleanly (`≤16,384` tokens) | Solved w/ Cap Forced Answer | Hit `16k` Token Cap | Median Thinking Tokens | Correct Answers / GPU-Hour (Clean / Forced) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1 pass** (`SenseNova-Loop`) | `32` | `0.5%` (`1/200`) | `3.5%` (`7/200`) | `72.5%` | `16,384` | `2` / `15` |
| **2 passes — model default** (`SenseNova-Loop`) | **`56`** | **`99.5%` (`199/200`)** | **`99.5%` (`199/200`)** | **`0.0%`** | **`572`** | **`666` / `666`** |
| **3 passes** (`SenseNova-Loop`) | `80` | `93.5%` (`187/200`) | `95.5%` (`191/200`) | `4.5%` | `685` | `371` / `378` |
| **4 passes** (`SenseNova-Loop`) | `104` | `88.5%` (`177/200`) | `92.5%` (`185/200`) | `9.0%` | `856.5` | `229` / `239` |
| **Plain `Qwen3.5-9B`** (no loop) | `32` | `85.5%` (`171/200`) | `96.0%` (`192/200`) | `14.5%` | `5,229.5` | `581` / `653` |

---

## Four Charts That Tell the Story

### 1. Accuracy Peaks Sharply at the Trained 2-Pass Setting
![Accuracy across loop passes vs plain Qwen3.5-9B](assets/accuracy_vs_passes.png)

### 2. Under Tight Token Budgets, 2 Latent Passes Crush Long Verbal Monologues
At a **`2,048`-token** budget, `2`-pass `SenseNova-Loop-9B` already finishes **`84.5%`** of problems cleanly, while plain `Qwen3.5-9B` has only finished **`14.0%`** (`86.0%` of Qwen's traces are still mid-calculation when the `2,048` cap cuts them off).

![Accuracy across token budgets](assets/budget_curve.png)

### 3. Why a Slower Per-Token Model Wins Per GPU-Hour
Each `2`-pass token does more work (`56` layer steps + two-stream Sinkhorn routing + separate per-pass caches), but needing **`572` median tokens instead of `5,230`** puts `2` passes at the top-right efficiency frontier (`666` correct answers/GPU-hour).

![Throughput vs accuracy frontier](assets/speed_vs_accuracy.png)

### 4. Paired Problem-by-Problem Gains & Losses (`N = 200`)
Because every model condition saw the **exact same 200 problems**, we test differences with exact two-sided McNemar tests on the problems where two settings disagreed (`+29` gained vs `-1` lost cleanly against `Qwen3.5-9B`; `+1` gained vs `-13` lost at `3` passes; `+0` gained vs `-22` lost at `4` passes).

![Paired problem gains and losses](assets/paired_changes.png)

---

## Side-by-Side Interactive Explorer (`app.py`)

Slide the loop dial from `1` to `4` passes—or toggle against plain `Qwen3.5-9B`—on any of the 200 logged problems, filter directly to disagreements, and expand each model's plain-English step-by-step reasoning:

![Gradio side-by-side app demo](assets/app_demo.gif)

Run it locally on any laptop (replays the saved runs in `results/`, **zero GPU required**):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python app.py
```

On a GPU machine (or Colab), add `--enable-live-gpu` to also run custom math questions live across pass counts.

---

## Heads-Up for Mac / Local Quantization Users (`MLX` & `GGUF`)

> **Warning:** While testing local consumer inference on a 16 GB Apple M3 Mac ([`results/mac_mlx_check.json`](results/mac_mlx_check.json)), we verified that current community quantizations (`hermitdave/SenseNova-Loop-9B-Preview-MLX-slq-4.5bpw` and `AtomicChat/SenseNova-Loop-9B-Preview-GGUF`) **silently export the 32 physical layers as a flat single-pass `Qwen3.5` graph and strip the `mHC` routing weights**. Running those quantized builds gives you a broken `1`-pass model rather than `SenseNova-Loop-9B`. Until `mlx-lm` and `llama.cpp` add native multi-pass `execution_plan` + `mHC` support, run the official BF16 checkpoint via `transformers>=5.18.0` or `vLLM` + PR [`#64832`](https://github.com/vllm-project/vllm/pull/64832).

---

## Reproduce Every Number

1. **Colab Notebook (walkthrough + smoke test + charts + live explorer):**
   Open [`notebooks/loop_dial_colab.ipynb`](notebooks/loop_dial_colab.ipynb) in Google Colab.
2. **Command-line pipeline:**
   ```bash
   # 1. Verify 32 / 56 / 80 / 104 decoder-layer calls per token via forward hooks + 5-problem sanity check
   python scripts/smoke_test.py

   # 2. Run any condition (resumable per-problem JSONL; supports --backend hf or --backend vllm)
   python scripts/run_sweep.py --condition loop_n2 --n-problems 200 --max-new-tokens 16384 --backend vllm

   # 3. Recompute exact McNemar paired stats, Wilson 95% CIs, budget curves, and regenerate all 5 charts
   python scripts/stats.py
   python scripts/generate_charts.py
   pytest -q
   ```

---

## Repository Layout

- [`app.py`](app.py) — Side-by-side Gradio explorer (offline replay + optional `--enable-live-gpu`).
- [`NOTES.md`](NOTES.md) — Full lab notebook: upstream config audit, forward-hook verification, partial middle-layer probe (`layers 16..26` vs `5..15`), `vLLM` PR `#64832` benchmarks, complete `N = 200` tables, surprises, and limitations.
- [`notebooks/loop_dial_colab.ipynb`](notebooks/loop_dial_colab.ipynb) — End-to-end Colab notebook.
- [`src/loop_dial/`](src/loop_dial/) — Execution-plan loader, `math-verify` + Hendrycks grader, 8-gram loop detector, exact McNemar + Wilson CI stats, and plain-Unicode math formatter.
- [`scripts/`](scripts/) — `smoke_test.py`, `run_sweep.py`, `stats.py`, `generate_charts.py`, `record_demo_gif.py`, `partial_layer_probe.py`, `probe_vllm.py`.
- [`data/subset_200.json`](data/subset_200.json) — Fixed `seed = 0` `MATH-500` subset (`200` problems).
- [`results/`](results/) — Complete logged JSONL trajectories (`1,000` runs across 5 conditions), Phase A hardware/hook probes, and summary CSVs.
- [`assets/`](assets/) — Plain-English PNG charts and live Gradio demo GIF.
