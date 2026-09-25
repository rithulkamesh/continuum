"""DurableAgent against an OpenAI-compatible completions server (vLLM / Ollama).

A local stub stands in for the server: it records every request and answers
in the `/v1/completions` wire format, so these tests pin the request shape,
step chaining, and text decoding without a GPU or a model download.
"""

from __future__ import annotations

import json
import threading
import zlib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from continuum import DurableAgent


class _Stub:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.replies: list[str] = []


def _handler(stub: _Stub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stub.requests.append({"path": self.path, **body})
            head = " ".join(body["prompt"].split()[:3])
            digest = zlib.crc32(body["prompt"].encode()) % 10_000
            # Non-ASCII + escapes exercise the shim's JSON string decoding;
            # the digest makes the reply depend on the whole prompt.
            text = f'done: {head} #{digest} \u2713 \U0001f600 "q"\n'
            stub.replies.append(text)
            payload = {
                "object": "text_completion",
                "model": body["model"],
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}},
            }
            raw = json.dumps(payload).encode()  # ensure_ascii -> \\u escapes
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            pass

    return Handler


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stub]:
    stub = _Stub()
    httpd = HTTPServer(("127.0.0.1", 0), _handler(stub))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("VLLM_BASE_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield stub
    httpd.shutdown()


# Three words each: the stub echoes the first three words of the prompt.
PROMPTS = ["summarize the bug", "locate the module", "draft a fix"]


def test_fake_backend_without_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    agent = DurableAgent()
    assert agent.backend == "fake"
    agent.begin(PROMPTS)
    agent.resume_from(agent.run_until_step(0))
    assert all(isinstance(o, list) for o in agent.step_outputs())


def test_vllm_backend_returns_text_and_chains_steps(server: _Stub) -> None:
    agent = DurableAgent()
    assert agent.backend == "vllm"
    agent.begin(PROMPTS)
    agent.resume_from(agent.run_until_step(0))

    outputs = agent.step_outputs()
    assert outputs == server.replies
    assert outputs[0].startswith(f"done: {PROMPTS[0]} #")
    assert outputs[0].endswith(' ✓ \U0001f600 "q"\n')

    assert len(server.requests) == 3
    assert {r["path"] for r in server.requests} == {"/v1/completions"}
    assert {r["model"] for r in server.requests} == {"gemma4"}
    # Step 1 sees only its prompt; later steps see their prompt followed by
    # the previous step's full generated text.
    assert server.requests[0]["prompt"] == PROMPTS[0]
    for i in (1, 2):
        prompt = server.requests[i]["prompt"]
        assert prompt.startswith(PROMPTS[i])
        assert " ".join(outputs[i - 1].split()) in prompt


def test_vllm_fork_changes_every_later_step(server: _Stub) -> None:
    recorder = DurableAgent()
    recorder.begin(PROMPTS, model_id="vllm/custom")
    checkpoint = recorder.run_until_step(0)
    forked = DurableAgent.fork(checkpoint, recorder.prompt_node_ids[1], "rewrite in rust")

    base, alt = DurableAgent(), DurableAgent()
    base.resume_from(checkpoint)
    alt.resume_from(forked)
    a, b = base.step_outputs(), alt.step_outputs()
    assert a[0] == b[0]
    assert a[1] != b[1] and a[2] != b[2]
    assert "rewrite in rust" in b[1]
    assert {r["model"] for r in server.requests} == {"custom"}


def test_prefix_state_survives_checkpoint_and_rewarms(
    server: _Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KV prefix state is exported into the checkpoint, re-imported on resume,
    and (opt-in) the server is re-warmed with a 1-token request per prefix."""
    from continuum.telemetry import CallbackObserver

    monkeypatch.setenv("VLLM_REWARM_ON_IMPORT", "1")
    recorder = DurableAgent()
    recorder.begin(PROMPTS)
    checkpoint = recorder.run_until_step(0)
    assert len(server.requests) == 1

    info = DurableAgent.inspect(checkpoint)
    assert info["checkpoint_bytes"] > 0

    revived = DurableAgent()
    events: list[dict] = []
    revived.set_observer(CallbackObserver(events.append))
    revived.resume_from(checkpoint)
    assert revived.cache_size() > 0

    rewarm = [r for r in server.requests[1:] if r["max_tokens"] == 1]
    assert len(rewarm) >= 1
    assert all(PROMPTS[0].split()[0] in r["prompt"] for r in rewarm)
    token_nodes = [
        e for e in events if e["kind"] == "node_execution" and e["node_kind"] == "TokenOp"
    ]
    assert len(token_nodes) == 2
    # The stub reports 4 cached prompt tokens; the backend surfaces them.
    assert all(e["tokens_saved"] == 4 and e["used_cached_state"] for e in token_nodes)
