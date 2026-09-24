# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Deterministic inference and authenticated external-tool failure fixtures."""

import json
import threading
import time
from http.server import ThreadingHTTPServer

from inference_fixture import MODEL
from inference_fixture import Handler as BaseHandler

STARTED = threading.Event()
COUNTERS = {"http": 0, "authfail": 0, "timeout": 0}
COUNTER_LOCK = threading.Lock()


class Handler(BaseHandler):
    def do_GET(self):
        if self.path == "/started":
            STARTED.set()
            self.reply({"started": True})
        elif self.path == "/stats":
            with COUNTER_LOCK:
                self.reply({"started": STARTED.is_set(), **COUNTERS})
        elif self.path in {"/resource", "/slow"}:
            authorized = self.authorized()
            with COUNTER_LOCK:
                COUNTERS[
                    (
                        "authfail"
                        if not authorized
                        else "timeout" if self.path == "/slow" else "http"
                    )
                ] += 1
            if not authorized:
                return
            if self.path == "/slow":
                time.sleep(2)
            try:
                self.reply({"result": "authenticated-external-tool"})
            except BrokenPipeError:
                return
        else:
            super().do_GET()

    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/v1/chat/completions" or body.get("model") != MODEL:
            self.reply({"error": "Unexpected fixture request"}, 400)
            return
        messages = json.dumps(body["messages"])
        mode = next(
            (
                mode
                for mode in ("cancel", "timeout", "authfail", "http", "egress")
                if "fixture-profile-" + mode in messages
            ),
            None,
        )
        marker = {
            "http": "authenticated-external-tool",
            "authfail": "expected-http-401",
            "timeout": "expected-timeout",
            "egress": "expected-egress-blocked",
        }.get(mode)
        if (
            mode
            and marker
            and marker in messages
            and any(
                row.get("role") in {"tool", "user"}
                and marker in str(row.get("content"))
                for row in body["messages"]
            )
        ):
            content = {"answer": "External profile passed: " + marker}
        elif mode:
            # The inference secret is also the fixture tool credential, within one worker trust boundary.
            code = "import urllib.request,urllib.error,socket,json\nfrom pathlib import Path\nkey=Path('/run/inference-secret').read_text().strip()\n"
            if mode in {"http", "authfail"}:
                code += (
                    "request=urllib.request.Request('http://profile-inference:8099/resource',headers={'Authorization':'Bearer '+key"
                    + ("+'-wrong'" if mode == "authfail" else "")
                    + "})\n"
                )
                code += "try:\n print(urllib.request.urlopen(request,timeout=2).read().decode())\nexcept urllib.error.HTTPError as error:\n print('expected-http-'+str(error.code))\n"
            elif mode == "timeout":
                code += "request=urllib.request.Request('http://profile-inference:8099/slow',headers={'Authorization':'Bearer '+key})\ntry:\n urllib.request.urlopen(request,timeout=.2)\nexcept (TimeoutError,urllib.error.URLError):\n print('expected-timeout')\n"
            elif mode == "egress":
                code += "try:\n socket.create_connection(('1.1.1.1',443),timeout=1).close()\n print('egress-unexpectedly-allowed')\nexcept OSError:\n print('expected-egress-blocked')\n"
            else:
                code += "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(600)'])\nurllib.request.urlopen('http://profile-inference:8099/started',timeout=2).read()\ntime.sleep(600)\n"
            content = {"tool": "run_python", "tool_args": {"code": code}}
        else:
            if "fixture-wait" in messages:
                time.sleep(8)
            content = {"answer": "Container inference fixture passed."}
        raw = json.dumps(content)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, reason in ((raw, None), ("", "stop")):
                chunk = {
                    "id": "profile-fixture",
                    "object": "chat.completion.chunk",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": reason,
                        }
                    ],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self.reply(
            {
                "id": "profile-fixture",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
        )


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8099), Handler).serve_forever()
