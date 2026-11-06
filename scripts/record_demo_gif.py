#!/usr/bin/env python3
"""Record `assets/app_demo.gif` directly from a live local `app.py` instance via headless Chrome CDP.

Uses zero extra third-party browser libraries: communicates over `127.0.0.1` Chrome DevTools Protocol
using Python's stdlib `socket` + `urllib.request`, captures crisp frames across slider/filter/reasoning
interactions, and encodes an animated GIF with Pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ok(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.35)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


class LocalCDPClient:
    """Minimal RFC 6455 WebSocket client strictly bound to `ws://127.0.0.1` for Chrome DevTools Protocol."""

    def __init__(self, ws_url: str) -> None:
        parsed = urllib.parse.urlparse(ws_url)
        if parsed.scheme != "ws" or parsed.hostname != "127.0.0.1":
            raise ValueError(f"Refusing non-loopback CDP WebSocket URL: {ws_url}")
        port = int(parsed.port or 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30.0)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(handshake.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("Closed during WebSocket handshake")
            buf += chunk
        header, self._rx_buf = buf.split(b"\r\n\r\n", 1)
        if b"101" not in header.splitlines()[0]:
            raise RuntimeError(f"Unexpected handshake response: {header!r}")
        self._msg_id = 0

    def _recv_exact(self, n: int) -> bytes:
        while len(self._rx_buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("CDP socket closed unexpectedly")
            self._rx_buf += chunk
        out, self._rx_buf = self._rx_buf[:n], self._rx_buf[n:]
        return out

    def _send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        length = len(payload)
        header = bytearray([0x81])  # FIN + text frame
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_frame(self) -> str:
        while True:
            b1, b2 = self._recv_exact(2)
            opcode = b1 & 0x0F
            masked = bool(b2 & 0x80)
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._recv_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._recv_exact(8))
            mask_key = self._recv_exact(4) if masked else b""
            data = self._recv_exact(length)
            if masked:
                data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:
                raise RuntimeError("CDP WebSocket closed by peer")
            if opcode == 0x9:  # ping -> pong
                pong = bytearray([0x8A, 0x80 | len(data)]) + os.urandom(4)
                self.sock.sendall(bytes(pong[:2]) + pong[2:6] + bytes(b ^ pong[2 + (i % 4)] for i, b in enumerate(data)))
                continue
            if opcode == 0x1:
                return data.decode("utf-8")

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._msg_id += 1
        req_id = self._msg_id
        self._send_text(json.dumps({"id": req_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self._recv_frame())
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error on {method}: {msg['error']}")
                return msg.get("result", {})

    def eval_js(self, expression: str) -> Any:
        res = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return res.get("result", {}).get("value")

    def screenshot_png(self) -> Image.Image:
        res = self.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        raw = base64.b64decode(res["data"])
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def record_gif(out_path: Path, chrome_bin: str = CHROME_DEFAULT) -> None:
    if not Path(chrome_bin).is_file():
        raise FileNotFoundError(f"Chrome binary not found at {chrome_bin}")

    app_port = _find_free_port()
    cdp_port = _find_free_port()
    user_data_dir = Path(tempfile.mkdtemp(prefix="loop_dial_chrome_"))

    app_proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "app.py"), "--host", "127.0.0.1", "--port", str(app_port)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chrome_proc: Optional[subprocess.Popen[str]] = None
    cdp: Optional[LocalCDPClient] = None

    try:
        app_url = f"http://127.0.0.1:{app_port}"
        _wait_http_ok(app_url, timeout_s=35.0)

        chrome_proc = subprocess.Popen(
            [
                chrome_bin,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--force-color-profile=srgb",
                "--window-size=1280,960",
                f"--remote-debugging-port={cdp_port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={user_data_dir}",
                app_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        _wait_http_ok(f"http://127.0.0.1:{cdp_port}/json/list", timeout_s=20.0)

        ws_url = ""
        for _ in range(30):
            with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/list", timeout=3.0) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
            page_targets = [t for t in targets if t.get("type") == "page" and "webSocketDebuggerUrl" in t]
            if page_targets:
                ws_url = page_targets[0]["webSocketDebuggerUrl"]
                break
            time.sleep(0.25)
        if not ws_url:
            raise RuntimeError("No page target found in headless Chrome")

        cdp = LocalCDPClient(ws_url)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1280, "height": 1040, "deviceScaleFactor": 1.25, "mobile": False},
        )
        cdp.call(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-color-scheme", "value": "light"}]},
        )
        cdp.call("Page.navigate", {"url": app_url})

        # Wait until `.ld-output-card` is rendered inside Gradio DOM
        for _ in range(40):
            ready = cdp.eval_js("document.querySelectorAll('.ld-output-card').length >= 2")
            if ready:
                break
            time.sleep(0.35)
        time.sleep(0.8)

        frames_and_durations: List[Tuple[Image.Image, int]] = []

        def snap(duration_ms: int) -> None:
            img = cdp.screenshot_png()
            resized = img.resize((1200, 975), Image.Resampling.LANCZOS)
            frames_and_durations.append((resized, duration_ms))

        # Helper JS to set Left slider value in Gradio 5/6 cleanly
        set_left_slider_js = """
        (val) => {
          const numInputs = Array.from(document.querySelectorAll('input[type="number"]'));
          const rangeInputs = Array.from(document.querySelectorAll('input[type="range"]'));
          if (rangeInputs.length > 0) {
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(rangeInputs[0], String(val));
            rangeInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
            rangeInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
          }
          if (numInputs.length > 0) {
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(numInputs[0], String(val));
            numInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
            numInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
          }
          return true;
        }
        """

        # Frame 1: Default 2 middle passes vs Plain Qwen3.5-9B
        snap(1900)

        # Frame 2: Open collapsible step-by-step reasoning <details> and scroll to the side-by-side cards
        cdp.eval_js(
            """
            document.querySelectorAll('details.ld-details').forEach(d => { d.open = true; });
            const firstCard = document.querySelector('.ld-output-card');
            if (firstCard) { firstCard.scrollIntoView({ behavior: 'instant', block: 'start' }); }
            true;
            """
        )
        time.sleep(0.5)
        snap(2200)

        # Collapse reasoning and scroll back to top so dial transitions show full controls + cards
        cdp.eval_js(
            """
            document.querySelectorAll('details.ld-details').forEach(d => { d.open = false; });
            window.scrollTo(0, 0);
            true;
            """
        )
        time.sleep(0.35)

        # Frame 3: Turn Loop Dial down to 1 pass
        cdp.eval_js(f"({set_left_slider_js})(1)")
        time.sleep(0.9)
        snap(1800)

        # Frame 4: Turn Loop Dial up to 3 passes
        cdp.eval_js(f"({set_left_slider_js})(3)")
        time.sleep(0.9)
        snap(1800)

        # Frame 5: Turn Loop Dial up to 4 passes
        cdp.eval_js(f"({set_left_slider_js})(4)")
        time.sleep(0.9)
        snap(1800)

        # Frame 6: Click "2 passes beats Plain Qwen3.5-9B" filter and set Left=2 passes
        cdp.eval_js(
            """
            (() => {
              const labels = Array.from(document.querySelectorAll('label'));
              const target = labels.find(el => el.textContent && el.textContent.includes('2 passes beats Plain Qwen3.5-9B'));
              if (target) target.click();
              return Boolean(target);
            })()
            """
        )
        time.sleep(1.0)
        cdp.eval_js(f"({set_left_slider_js})(2)")
        time.sleep(0.8)
        snap(2100)

        # Frame 7: Switch Benchmark Dataset selector to AIME 2024+2025 (60 olympiad problems · 32k token cap)
        cdp.eval_js(
            """
            (() => {
              const labels = Array.from(document.querySelectorAll('label'));
              const target = labels.find(el => el.textContent && el.textContent.includes('AIME 2024+2025'));
              if (target) target.click();
              return Boolean(target);
            })()
            """
        )
        time.sleep(1.1)
        snap(2400)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        quantized = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for im, _ in frames_and_durations]
        durations = [dur for _, dur in frames_and_durations]
        quantized[0].save(
            out_path,
            save_all=True,
            append_images=quantized[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
        print(f"Saved animated GIF ({len(quantized)} frames, {out_path.stat().st_size / 1024:.1f} KB) -> {out_path}")

    finally:
        if cdp is not None:
            cdp.close()
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
        app_proc.terminate()
        try:
            app_proc.wait(timeout=5)
        except Exception:
            app_proc.kill()
        shutil.rmtree(user_data_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record assets/app_demo.gif from live app.py.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "app_demo.gif")
    parser.add_argument("--chrome", default=CHROME_DEFAULT)
    args = parser.parse_args()
    record_gif(args.out, chrome_bin=args.chrome)


if __name__ == "__main__":
    main()
