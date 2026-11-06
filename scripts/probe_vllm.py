#!/usr/bin/env python3
"""Phase A Speed & Serving Probe: test `vllm==0.33.0` (stock + PR #64832) on `SenseNova-Loop-9B-Preview`.

Runs inside an isolated `/content/vllm-venv` so the main `/usr` environment (`transformers==5.18.0` + Colab CUDA `torch`)
is never modified. Logs to `results/vllm_probe.json`:
1. Stock `vllm==0.33.0` behavior on `sensenova/SenseNova-Loop-9B-Preview` (does it load? does it honor `loop_settings`?).
2. Stock `vllm==0.33.0` speed on `Qwen/Qwen3.5-9B`.
3. Upstream `vllm-project/vllm#64832` (`86014ad44c3fab4be89dfc169f5365ed850fd195`) pure-Python/Triton patch check:
   whether it applies cleanly onto `vllm==0.33.0` and honors `hf_overrides["loop_settings"]` across `n=1,2,3,4`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
VLLM_VENV = Path("/content/vllm-venv")
PR_64832_DIFF_URL = "https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/64832.diff"


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    print("[probe_vllm] $", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PATH"] = f"{VLLM_VENV / 'bin'}:{env.get('PATH', '')}"
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True, timeout=timeout)


def ensure_vllm_venv() -> Path:
    import shutil

    if VLLM_VENV.exists():
        shutil.rmtree(VLLM_VENV, ignore_errors=True)
    vpy = VLLM_VENV / "bin" / "python"
    res = run_cmd(["uv", "venv", str(VLLM_VENV), "--python", "3.12"])
    if res.returncode != 0:
        res = run_cmd(["uv", "venv", str(VLLM_VENV)])
        if res.returncode != 0:
            raise RuntimeError(f"uv venv failed: {res.stderr}")

    install_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(vpy),
        "vllm==0.33.0",
        "ninja",
        "datasets",
        "math-verify[antlr4_13_2]",
        "scipy",
        "pandas",
        "tqdm",
    ]
    res = run_cmd(install_cmd, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"vllm==0.33.0 install failed:\n{res.stderr[-2000:]}")
    run_cmd(["uv", "pip", "install", "--python", str(vpy), "--no-deps", "-e", str(REPO_ROOT)])
    return vpy


def test_stock_vllm_loop(vpy: Path) -> Dict[str, Any]:
    snippet = r"""
import json, time
out = {"tested": True}
try:
    import vllm, torch, transformers
    out["vllm_version"] = vllm.__version__
    out["torch_version"] = torch.__version__
    out["transformers_version"] = transformers.__version__
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="sensenova/SenseNova-Loop-9B-Preview",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        hf_overrides={"loop_settings": {"start": 4, "end": 28, "eval_count": 1, "counts": [1]}},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    t0 = time.perf_counter()
    res = llm.generate(["Solve x + 2 = 5."], sp)
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    out["sample_text"] = res[0].outputs[0].text
    out["supported_out_of_box"] = True
except Exception as e:
    out["supported_out_of_box"] = False
    out["error_type"] = type(e).__name__
    out["error_message"] = str(e)[:1200]
print("JSON_RESULT:" + json.dumps(out))
"""
    res = run_cmd([str(vpy), "-c", snippet], timeout=600)
    for line in (res.stdout or "").splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:") :])
    return {
        "supported_out_of_box": False,
        "returncode": res.returncode,
        "stdout_tail": (res.stdout or "")[-1000:],
        "stderr_tail": (res.stderr or "")[-1500:],
    }


def verify_passes_in_vllm(vpy: Path) -> Dict[str, Any]:
    """Run n=1,2,3,4 in separate subprocesses so V1 EngineCore tears down cleanly between passes."""
    outputs_by_n: Dict[str, Any] = {}
    for n_passes in (1, 2, 3, 4):
        verify_snippet = f"""
import json, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from loop_dial.metrics import build_math_prompt

PROMPT = "Find all positive integers $n$ such that $n^2 + 19n + 48$ is a perfect square."
tok = AutoTokenizer.from_pretrained("sensenova/SenseNova-Loop-9B-Preview", trust_remote_code=True)
chat = tok.apply_chat_template(
    [{{"role": "user", "content": build_math_prompt(PROMPT)}}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
llm = LLM(
    model="sensenova/SenseNova-Loop-9B-Preview",
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=64,
    gpu_memory_utilization=0.82,
    hf_overrides={{"loop_settings": {{"start": 4, "end": 28, "eval_count": {n_passes}, "counts": [{n_passes}]}}}},
)
# Warmup
_ = llm.generate([chat], SamplingParams(temperature=0.0, max_tokens=8))
t0 = time.perf_counter()
res = llm.generate([chat], SamplingParams(temperature=0.0, max_tokens=64, min_tokens=64))
dt = time.perf_counter() - t0
gen_ids = list(res[0].outputs[0].token_ids)
print("VLLM_N_OUT:" + json.dumps({{
    "n_passes": {n_passes},
    "n_tokens": len(gen_ids),
    "elapsed_s": round(dt, 4),
    "ms_per_token": round(dt * 1000.0 / max(len(gen_ids), 1), 3),
    "tok_per_s": round(len(gen_ids) / max(dt, 1e-6), 2),
    "preview_32": tok.decode(gen_ids[:32], skip_special_tokens=False),
    "preview_64": tok.decode(gen_ids, skip_special_tokens=False),
}}))
"""
        vr = run_cmd([str(vpy), "-c", verify_snippet], timeout=600)
        parsed = None
        for line in (vr.stdout or "").splitlines():
            if line.startswith("VLLM_N_OUT:"):
                parsed = json.loads(line[len("VLLM_N_OUT:") :])
        if parsed is None:
            outputs_by_n[str(n_passes)] = {
                "ok": False,
                "rc": vr.returncode,
                "stderr_tail": (vr.stderr or "")[-1500:],
                "stdout_tail": (vr.stdout or "")[-1000:],
            }
            break
        parsed["ok"] = True
        outputs_by_n[str(n_passes)] = parsed

    previews_64 = [v.get("preview_64") for v in outputs_by_n.values() if v.get("ok")]
    return {
        "per_pass_verification": outputs_by_n,
        "all_four_passes_succeeded": len(previews_64) == 4,
        "distinct_64tok_trajectories": len(set(previews_64)),
        "honors_loop_settings_distinct_outputs": len(previews_64) == 4 and len(set(previews_64)) >= 3,
    }


def test_stock_vllm_qwen35(vpy: Path) -> Dict[str, Any]:
    snippet = r"""
import json, time
out = {"tested": True}
try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from loop_dial.metrics import build_math_prompt
    PROMPT = "Find all positive integers $n$ such that $n^2 + 19n + 48$ is a perfect square."
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    chat = tok.apply_chat_template(
        [{"role": "user", "content": build_math_prompt(PROMPT)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=64,
        gpu_memory_utilization=0.82,
    )
    _ = llm.generate([chat], SamplingParams(temperature=0.0, max_tokens=8))
    t0 = time.perf_counter()
    res = llm.generate([chat], SamplingParams(temperature=0.0, max_tokens=64, min_tokens=64))
    dt = time.perf_counter() - t0
    gen_ids = list(res[0].outputs[0].token_ids)
    out["supported"] = True
    out["n_tokens"] = len(gen_ids)
    out["elapsed_s"] = round(dt, 4)
    out["ms_per_token"] = round(dt * 1000.0 / max(len(gen_ids), 1), 3)
    out["tok_per_s"] = round(len(gen_ids) / max(dt, 1e-6), 2)
    out["preview_32"] = tok.decode(gen_ids[:32], skip_special_tokens=False)
except Exception as e:
    out["supported"] = False
    out["error_type"] = type(e).__name__
    out["error_message"] = str(e)[:1200]
print("QWEN_VLLM_OUT:" + json.dumps(out))
"""
    res = run_cmd([str(vpy), "-c", snippet], timeout=600)
    for line in (res.stdout or "").splitlines():
        if line.startswith("QWEN_VLLM_OUT:"):
            return json.loads(line[len("QWEN_VLLM_OUT:") :])
    return {
        "supported": False,
        "returncode": res.returncode,
        "stderr_tail": (res.stderr or "")[-1500:],
    }


def apply_pr_64832_and_test(vpy: Path) -> Dict[str, Any]:
    # Locate site-packages/vllm inside VLLM_VENV
    loc = run_cmd([str(vpy), "-c", "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)"])
    site_pkg = Path(loc.stdout.strip().splitlines()[-1])

    diff_path = Path("/tmp/vllm_pr_64832.diff")
    with urllib.request.urlopen(PR_64832_DIFF_URL, timeout=30) as resp:
        diff_bytes = resp.read()
    diff_path.write_bytes(diff_bytes)

    # Filter diff to only `vllm/` files (docs/tests aren't shipped in PyPI wheel)
    raw_diff = diff_bytes.decode("utf-8", errors="replace")
    filtered_chunks: list[str] = []
    for chunk in raw_diff.split("diff --git "):
        if not chunk.strip():
            continue
        first_line = chunk.splitlines()[0]
        if first_line.startswith("a/vllm/"):
            filtered_chunks.append("diff --git " + chunk)
    vllm_only_diff = Path("/tmp/vllm_pr_64832_code_only.diff")
    vllm_only_diff.write_text("".join(filtered_chunks), encoding="utf-8")

    dry = run_cmd(["patch", "--dry-run", "-p1", "-i", str(vllm_only_diff)], cwd=site_pkg)
    result: Dict[str, Any] = {
        "pr_url": "https://github.com/vllm-project/vllm/pull/64832",
        "dry_run_rc": dry.returncode,
        "dry_run_stdout": dry.stdout[-1500:],
        "dry_run_stderr": dry.stderr[-1000:],
    }
    if dry.returncode != 0:
        dry_fuzz = run_cmd(["patch", "--dry-run", "--fuzz=3", "-p1", "-i", str(vllm_only_diff)], cwd=site_pkg)
        result["dry_run_fuzz3_rc"] = dry_fuzz.returncode
        result["dry_run_fuzz3_stdout"] = dry_fuzz.stdout[-1500:]
        if dry_fuzz.returncode != 0:
            result["applied"] = False
            return result
        apply_res = run_cmd(["patch", "--fuzz=3", "-p1", "-i", str(vllm_only_diff)], cwd=site_pkg)
    else:
        apply_res = run_cmd(["patch", "-p1", "-i", str(vllm_only_diff)], cwd=site_pkg)

    result["applied"] = apply_res.returncode == 0
    if not result["applied"]:
        result["apply_stderr"] = apply_res.stderr[-1000:]
        return result

    result.update(verify_passes_in_vllm(vpy))
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vpy = ensure_vllm_venv()
    qwen_res = test_stock_vllm_qwen35(vpy)
    stock_res = test_stock_vllm_loop(vpy)
    pr_res: Dict[str, Any] = {}
    if stock_res.get("supported_out_of_box"):
        stock_res.update(verify_passes_in_vllm(vpy))
    else:
        pr_res = apply_pr_64832_and_test(vpy)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "vllm_venv": str(vpy),
        "stock_vllm_0_33_0_qwen35_9b": qwen_res,
        "stock_vllm_0_33_0_sensenova_loop": stock_res,
        "pr_64832_overlay": pr_res,
    }
    out_file = RESULTS_DIR / "vllm_probe.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved vLLM probe results -> {out_file}", flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
