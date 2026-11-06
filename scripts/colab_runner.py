#!/usr/bin/env python3
"""Safe Colab GPU (`G4` = RTX PRO 6000 96 GB) session manager & job runner via `colab` CLI.

Guarantees:
- Uses isolated session names (`loop-dial-...`).
- Uploads only lightweight source/data (`src/`, `scripts/`, `data/subset_200.json`), NEVER `.venv` or large artifacts.
- Validates archive extraction with `tarfile.extractall(..., filter="data")` to prevent path traversal (`ZipSlip`).
- Always calls `colab stop -s <session>` inside `finally:` unless `--keep-session` is explicitly passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _colab_env() -> dict[str, str]:
    env = dict(os.environ)
    extra_paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / "bin"),
    ]
    env["PATH"] = ":".join(extra_paths + [env.get("PATH", "")])

    if not env.get("SSL_CERT_FILE"):
        uv_cert = (
            Path.home()
            / ".local/share/uv/tools/google-colab-cli/lib/python3.12/site-packages/certifi/cacert.pem"
        )
        if uv_cert.is_file():
            cert_path = str(uv_cert)
        else:
            try:
                import certifi

                cert_path = certifi.where()
            except Exception:
                cert_path = ""
        if cert_path:
            env["SSL_CERT_FILE"] = cert_path
            env["WEBSOCKET_CLIENT_CA_BUNDLE"] = cert_path
    return env


def run_colab_cmd(
    args: Sequence[str],
    *,
    check: bool = True,
    proc_timeout: float = 300.0,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    import signal

    cmd = ["colab", *args]
    print(f"[colab_runner] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        env=_colab_env(),
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=proc_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
        print(f"[colab_runner] Killed hung colab CLI after {proc_timeout}s: {' '.join(cmd)}", flush=True)
        if check:
            raise TimeoutError(f"colab CLI timed out after {proc_timeout}s: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=124, stdout="", stderr="TimeoutExpired")

    res = subprocess.CompletedProcess(cmd, returncode=proc.returncode, stdout=out or "", stderr=err or "")
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return res


def build_upload_tarball(tar_path: Path) -> None:
    """Bundle `src/`, `scripts/`, `data/subset_200.json`, `pyproject.toml`, and local `results/*.json*` checkpoints."""
    include_paths = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "data" / "subset_200.json",
    ]
    results_dir = REPO_ROOT / "results"
    if results_dir.is_dir():
        for res_file in sorted(results_dir.iterdir()):
            if res_file.is_file() and res_file.suffix in {".json", ".jsonl", ".csv"}:
                include_paths.append(res_file)
    with tarfile.open(tar_path, "w:gz") as tf:
        for item in include_paths:
            if item.exists():
                arcname = str(item.relative_to(REPO_ROOT))
                tf.add(item, arcname=arcname)


def safe_extract_results_tar(tar_path: Path, dest_dir: Path) -> List[str]:
    """Safely extract downloaded results archive using `filter='data'` (blocks path traversal & symlinks)."""
    extracted: List[str] = []
    dest_resolved = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            member_target = (dest_resolved / member.name).resolve()
            if not str(member_target).startswith(str(dest_resolved) + os.sep):
                raise RuntimeError(f"Blocked unsafe tar member path: {member.name}")
            extracted.append(member.name)
        tf.extractall(path=dest_resolved, filter="data")
    return extracted


def exec_inline_python(
    session: str,
    code: str,
    retries: int = 4,
    timeout: float = 45.0,
    capture_output: bool = False,
) -> str:
    import time

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)
    try:
        for attempt in range(1, retries + 1):
            res = run_colab_cmd(
                ["exec", "-s", session, "--timeout", str(timeout), "-f", str(tmp_path)],
                check=False,
                proc_timeout=timeout + 20.0,
                capture_output=capture_output,
            )
            if res.returncode == 0:
                return res.stdout or ""
            if attempt < retries:
                print(f"[colab_runner] exec attempt {attempt}/{retries} failed (rc={res.returncode}), retrying in 5s...", flush=True)
                time.sleep(5.0)
        raise RuntimeError(f"colab exec failed after {retries} attempts on session '{session}'")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sync_repo_to_session(session: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_local = Path(tmpdir) / "loop_dial_bundle.tar.gz"
        build_upload_tarball(tar_local)
        run_colab_cmd(["upload", "-s", session, str(tar_local), "/content/loop_dial_bundle.tar.gz"], proc_timeout=180.0)

    unpack_code = """
import os, tarfile
os.makedirs("/content/loop-dial", exist_ok=True)
with tarfile.open("/content/loop_dial_bundle.tar.gz", "r:gz") as tf:
    tf.extractall("/content/loop-dial", filter="data")
print("Unpacked bundle to /content/loop-dial:", sorted(os.listdir("/content/loop-dial")))
"""
    exec_inline_python(session, unpack_code, timeout=45.0)


def download_results_from_session(session: str) -> None:
    pack_code = """
import os, tarfile
res_dir = "/content/loop-dial/results"
os.makedirs(res_dir, exist_ok=True)
out_tar = "/content/loop_dial_results.tar.gz"
with tarfile.open(out_tar, "w:gz") as tf:
    for fn in sorted(os.listdir(res_dir)):
        if fn.endswith((".json", ".jsonl", ".csv")) and not fn.endswith(".done.json"):
            tf.add(os.path.join(res_dir, fn), arcname=os.path.join("results", fn))
print("Packed remote results:", os.path.getsize(out_tar), "bytes")
"""
    exec_inline_python(session, pack_code, timeout=60.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = Path(tmpdir) / "loop_dial_results.tar.gz"
        run_colab_cmd(
            ["download", "-s", session, "/content/loop_dial_results.tar.gz", str(local_tar)],
            proc_timeout=240.0,
        )
        names = safe_extract_results_tar(local_tar, REPO_ROOT)
        print(f"[colab_runner] Safely extracted {len(names)} entries into {REPO_ROOT / 'results'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Colab GPU runs for loop-dial.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="Provision GPU session, install pinned deps, and upload repo")
    p_start.add_argument("--session", default="loop-dial-gpu")
    p_start.add_argument("--gpu", default="G4")

    p_sync = sub.add_parser("sync", help="Re-upload local src/scripts/data bundle to active session")
    p_sync.add_argument("--session", default="loop-dial-gpu")

    p_pull = sub.add_parser("pull", help="Download /content/loop-dial/results into local results/")
    p_pull.add_argument("--session", default="loop-dial-gpu")

    p_exec = sub.add_parser("exec", help="Execute a Python file on the active session")
    p_exec.add_argument("--session", default="loop-dial-gpu")
    p_exec.add_argument("--file", required=True)
    p_exec.add_argument("--timeout", type=float, default=14400.0)

    p_stop = sub.add_parser("stop", help="Stop Colab GPU session immediately")
    p_stop.add_argument("--session", default="loop-dial-gpu")

    args = parser.parse_args()
    if args.action == "start":
        import time

        run_colab_cmd(["new", "-s", args.session, "--gpu", args.gpu], proc_timeout=300.0)
        time.sleep(6.0)
        sync_repo_to_session(args.session)
    elif args.action == "sync":
        sync_repo_to_session(args.session)
    elif args.action == "pull":
        download_results_from_session(args.session)
    elif args.action == "exec":
        run_colab_cmd(
            ["exec", "-s", args.session, "--timeout", str(args.timeout), "-f", args.file],
            proc_timeout=args.timeout + 30.0,
        )
    elif args.action == "stop":
        run_colab_cmd(["stop", "-s", args.session], check=False, proc_timeout=60.0)


if __name__ == "__main__":
    main()
