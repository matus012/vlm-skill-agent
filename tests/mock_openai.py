"""Tiny scriptable OpenAI-compatible server: GET /v1/models, POST /v1/chat/completions."""
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def tool_call_msg(name: str, arguments: Any) -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}


class MockServer:
    """script: list of (status, message-or-None); cycled. Received request bodies kept in .requests.
    responder (optional): request body -> (status, message-or-None), used instead of the script.
    api_key (optional): like `vllm serve --api-key`, /v1/* answers 401 {"error": "Unauthorized"} unless the request
    carries `Authorization: Bearer <api_key>`; other paths (e.g. /health) stay open."""

    def __init__(self, script: list[tuple[int, dict[str, Any] | None]],
                 responder: Callable[[dict[str, Any]], tuple[int, dict[str, Any] | None]] | None = None,
                 api_key: str | None = None) -> None:
        self.script, self.requests, self._i = script, [], 0
        self.responder, self.api_key = responder, api_key
        self.headers: list[dict[str, str]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _denied(self) -> bool:
                if outer.api_key is None or not self.path.startswith("/v1"):
                    return False
                if self.headers.get("Authorization") == f"Bearer {outer.api_key}":
                    return False
                self._send(401, {"error": "Unauthorized"})
                return True

            def do_GET(self) -> None:
                if self.path == "/health":
                    return self._send(200, {})
                if self._denied():
                    return
                self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})

            def do_POST(self) -> None:
                outer.headers.append(dict(self.headers))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self._denied():
                    return
                outer.requests.append(body)
                if outer.responder is not None:
                    code, msg = outer.responder(body)
                else:
                    code, msg = outer.script[outer._i % len(outer.script)]
                    outer._i += 1
                if msg is None:
                    return self._send(code, {"error": {"message": "boom", "type": "server_error"}})
                fin = "tool_calls" if msg.get("tool_calls") else "stop"
                self._send(code, {"id": "x", "object": "chat.completion", "created": 0, "model": "mock",
                                  "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
