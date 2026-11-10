import json
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
MODEL = "Domyn/Domyn-Edge-SFT"  # a real tokenizer and config: BFCL's handler and lm_eval load them
ANSWER = "ANSWER: 42"


@dataclass
class FakeEndpoint:
    """An OpenAI-compatible server answering every request with ``ANSWER``.

    Attributes:
        url: The base URL, ending in ``/v1``.
        requests: The ``(path, model)`` of every completion request, in order.
    """

    url: str
    requests: list[tuple[str, str]] = field(default_factory=list)


def _completion(body: dict, path: str) -> dict:
    prompts = body.get("prompt", [""]) if path == "/v1/completions" else [None]
    if isinstance(prompts, str) or (prompts and isinstance(prompts[0], int)):  # one prompt, as text or token ids
        prompts = [prompts]
    if path == "/v1/completions":
        choices = [{"index": i, "text": ANSWER, "finish_reason": "stop", "logprobs": None} for i in range(len(prompts))]
        kind = "text_completion"
    else:
        message = {"role": "assistant", "content": ANSWER, "reasoning_content": "thinking"}
        choices = [{"index": 0, "message": message, "finish_reason": "stop", "logprobs": None}]
        kind = "chat.completion"
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return {
        "id": "fake",
        "object": kind,
        "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": choices,
        "usage": usage,
    }


@pytest.fixture
def fake_endpoint() -> Iterator[FakeEndpoint]:
    """A local OpenAI-compatible server serving ``MODEL``, up for the test."""
    endpoint = FakeEndpoint(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 32768}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path not in ("/v1/chat/completions", "/v1/completions"):
                self.send_error(404)
                return
            if body.get("stream"):
                self._json(400, {"error": {"message": "the fake endpoint does not stream"}})
                return
            endpoint.requests.append((self.path, body.get("model", "")))
            self._json(200, _completion(body, self.path))

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield endpoint
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def suites_ready() -> None:
    """Skip unless uv and the suite checkouts are there."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    for suite in ("domyn-inspect", "domyn-bfcl"):
        if not (REPO / "suites" / suite / "pyproject.toml").is_file():
            pytest.skip(f"suites/{suite} is not checked out; run `git submodule update --init`")
