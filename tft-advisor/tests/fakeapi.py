"""A tiny fake Anthropic Messages API server for offline tests.

Runs a real HTTP server on 127.0.0.1 so the real ``anthropic`` SDK code path
(request building, beta headers, parsing) is exercised without network.

    with FakeAnthropic() as fake:
        fake.queue_json({"headline": "..."})          # structured output reply
        client = fake.client()                         # anthropic.Anthropic bound to it
        ...
        fake.requests[0]["body"]["fallbacks"] == "default"
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


class FakeAnthropic:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies: list[tuple[int, dict[str, Any]]] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                if self.headers.get("content-encoding") == "gzip":
                    import gzip

                    raw = gzip.decompress(raw)
                body = json.loads(raw or b"{}")
                with fake._lock:
                    fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    status, reply = fake._replies.pop(0) if fake._replies else (200, fake._message_text("ok"))
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("request-id", "req_fake")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ---- lifecycle ---------------------------------------------------------------
    def __enter__(self) -> "FakeAnthropic":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def client(self) -> Any:
        import anthropic

        return anthropic.Anthropic(api_key="test-key", base_url=self.base_url, max_retries=0, timeout=10)

    # ---- scripted replies --------------------------------------------------------
    @staticmethod
    def _message_text(text: str, stop_reason: str = "end_turn", model: str = "claude-opus-5") -> dict[str, Any]:
        return {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 0,
            },
        }

    def queue_text(self, text: str, stop_reason: str = "end_turn") -> None:
        self._replies.append((200, self._message_text(text, stop_reason)))

    def queue_json(self, obj: Any, stop_reason: str = "end_turn") -> None:
        self.queue_text(json.dumps(obj, ensure_ascii=False), stop_reason)

    def queue_refusal(self) -> None:
        msg = self._message_text("", "refusal")
        msg["content"] = []
        msg["stop_details"] = {"type": "refusal", "category": None, "explanation": None}
        self._replies.append((200, msg))

    def queue_error(self, status: int, message: str = "boom", etype: Optional[str] = None) -> None:
        etype = etype or {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error", 429: "rate_limit_error"}.get(status, "api_error")
        self._replies.append((status, {"type": "error", "error": {"type": etype, "message": message}}))
