"""OpenAI-compatible proxy shim (issue #20).

A stub upstream speaks the OpenAI wire format; the real ``openai`` SDK talks to
the proxy through ``base_url``, exactly as ``OPENAI_BASE_URL`` would set it.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from continuum.embeddings import PrecomputedEmbeddingProvider
from continuum.proxy import (
    ContinuumProxy,
    ProxyConfig,
    cache_bypass_reason,
    main,
    replay_as_stream,
    request_key_parts,
)

openai = pytest.importorskip("openai")


class Upstream:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: int | None = None
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        httpd.daemon_threads = True
        self.httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, status: int, obj: Any, ctype: str = "application/json") -> None:
                raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                up.calls.append({"path": self.path, "auth": self.headers.get("Authorization")})
                self._send(
                    200, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]}
                )

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                up.calls.append(
                    {"path": self.path, "auth": self.headers.get("Authorization"), **body}
                )
                if up.fail_next is not None:
                    status, up.fail_next = up.fail_next, None
                    self._send(status, {"error": {"message": "rate limited", "type": "rate_limit"}})
                    return
                n = len(up.calls)
                if self.path.endswith("/embeddings"):
                    self._send(200, {"data": [{"embedding": [1.0, 0.0]}]})
                    return
                chat = self.path.endswith("/chat/completions")
                if chat:
                    last = body["messages"][-1]["content"]
                    text = f"reply#{n} to {last}"
                else:
                    text = f"completion#{n} of {body['prompt']}"
                if body.get("tools"):
                    choice = {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{n}",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                    self._send(
                        200,
                        {
                            "id": f"up-{n}",
                            "object": "chat.completion",
                            "created": 1,
                            "model": body["model"],
                            "choices": [choice],
                        },
                    )
                    return
                usage = {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 16},
                }
                if not body.get("stream"):
                    choice = (
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                        if chat
                        else {"index": 0, "text": text, "finish_reason": "stop"}
                    )
                    self._send(
                        200,
                        {
                            "id": f"up-{n}",
                            "object": "chat.completion" if chat else "text_completion",
                            "created": 1,
                            "model": body["model"],
                            "choices": [choice],
                            "usage": usage,
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                obj = "chat.completion.chunk" if chat else "text_completion"
                pieces = [text[: len(text) // 2], text[len(text) // 2 :]]
                for i, piece in enumerate(pieces):
                    delta = (
                        {
                            "delta": {"role": "assistant", "content": piece}
                            if i == 0
                            else {"content": piece}
                        }
                        if chat
                        else {"text": piece}
                    )
                    chunk = {
                        "id": f"up-{n}",
                        "object": obj,
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, **delta, "finish_reason": None}],
                    }
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                done = {
                    "id": f"up-{n}",
                    "object": obj,
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            **({"delta": {}} if chat else {"text": ""}),
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
                if (body.get("stream_options") or {}).get("include_usage"):
                    tail = {
                        "id": f"up-{n}",
                        "object": obj,
                        "created": 1,
                        "model": body["model"],
                        "choices": [],
                        "usage": usage,
                    }
                    self.wfile.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        return H


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    up = Upstream()
    yield up
    up.httpd.shutdown()


@pytest.fixture
def proxy(upstream: Upstream) -> Iterator[ContinuumProxy]:
    p = ContinuumProxy(ProxyConfig(upstream=upstream.url), port=0).start()
    yield p
    p.close()


def _client(proxy: ContinuumProxy) -> Any:
    return openai.OpenAI(base_url=proxy.address + "/v1", api_key="sk-test", max_retries=0)


MESSAGES = [
    {"role": "system", "content": "You are a support bot for Acme. Be brief."},
    {"role": "user", "content": "how do I reset my password?"},
]


def _metrics(proxy: ContinuumProxy) -> dict[str, Any]:
    with urllib.request.urlopen(proxy.address + "/continuum/metrics") as r:
        return json.loads(r.read())


def test_chat_repeat_is_served_from_cache(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    first = client.chat.completions.with_raw_response.create(
        model="m", messages=MESSAGES, temperature=0
    )
    second = client.chat.completions.with_raw_response.create(
        model="m", messages=MESSAGES, temperature=0
    )
    assert first.headers["x-continuum-cache"] == "miss"
    assert second.headers["x-continuum-cache"] == "hit"
    assert second.headers["x-continuum-served-by"] == "memo"
    assert int(second.headers["x-continuum-tokens-saved"]) > 0
    a, b = first.parse(), second.parse()
    assert (
        a.choices[0].message.content
        == b.choices[0].message.content
        == "reply#1 to how do I reset my password?"
    )
    assert a.id != b.id
    assert len([c for c in upstream.calls if c["path"] == "/v1/chat/completions"]) == 1
    assert upstream.calls[0]["auth"] == "Bearer sk-test"

    # Different parameters are a different key.
    client.chat.completions.create(model="m", messages=MESSAGES, temperature=0.5)
    assert len(upstream.calls) == 2
    # A new question under the same system prompt misses the memo tier but
    # shares a prefix with the first request.
    other = [MESSAGES[0], {"role": "user", "content": "where is my order?"}]
    client.chat.completions.create(model="m", messages=other, temperature=0)
    assert len(upstream.calls) == 3
    m = _metrics(proxy)
    assert m["requests"] == {"hit": 1, "miss": 3, "bypass": 0, "error": 0}
    assert m["tier_hits"]["memo"] == 1
    assert m["tier_hits"].get("prefix_kv", 0) >= 1  # shared system prompt
    assert m["tokens_saved"] > 0


def test_completions_endpoint(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    a = client.completions.create(model="m", prompt="say hi", max_tokens=5)
    b = client.completions.create(model="m", prompt="say hi", max_tokens=5)
    assert a.choices[0].text == b.choices[0].text == "completion#1 of say hi"
    assert len(upstream.calls) == 1


def test_streaming_miss_then_replayed_hit(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "reply#1 to how do I reset my password?"
    assert upstream.calls[0]["stream"] is True
    assert "stream_options" not in upstream.calls[0]  # never injected

    raw = client.chat.completions.with_raw_response.create(
        model="m", messages=MESSAGES, stream=True
    )
    assert raw.headers["x-continuum-cache"] == "hit"
    replayed = "".join(c.choices[0].delta.content or "" for c in raw.parse() if c.choices)
    assert replayed == text
    # A non-streaming request with the same content is the same key.
    plain = client.chat.completions.create(model="m", messages=MESSAGES)
    assert plain.choices[0].message.content == text
    assert len(upstream.calls) == 1

    usage_stream = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "u"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert [c.usage.prompt_tokens for c in usage_stream if c.usage] == [20]

    comp = "".join(
        c.choices[0].text for c in client.completions.create(model="m", prompt="p", stream=True)
    )
    again = "".join(
        c.choices[0].text for c in client.completions.create(model="m", prompt="p", stream=True)
    )
    assert comp == again == "completion#3 of p"


def test_tool_calls_are_never_cached(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    for _ in range(2):
        raw = client.chat.completions.with_raw_response.create(
            model="m", messages=MESSAGES, tools=tools
        )
        assert raw.headers["x-continuum-cache"] == "bypass"
        assert raw.parse().choices[0].message.tool_calls[0].function.name == "lookup"
    followup = [
        *MESSAGES,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found it"},
    ]
    for _ in range(2):
        client.chat.completions.create(model="m", messages=followup)
    assert len(upstream.calls) == 4
    assert _metrics(proxy)["requests"]["bypass"] == 4


def test_no_cache_header_and_n_bypass(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    for _ in range(2):
        client.chat.completions.create(
            model="m", messages=MESSAGES, extra_headers={"Cache-Control": "no-cache"}
        )
    assert len(upstream.calls) == 2
    assert cache_bypass_reason({"n": 3}, {}) == "n>1"
    assert cache_bypass_reason({"functions": [{}]}, {}) == "tools"
    assert cache_bypass_reason({"messages": [{"role": "user", "content": "x"}]}, {}) is None


def test_upstream_errors_pass_through_and_are_not_cached(
    proxy: ContinuumProxy, upstream: Upstream
) -> None:
    client = _client(proxy)
    upstream.fail_next = 429
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(model="m", messages=MESSAGES)
    ok = client.chat.completions.create(model="m", messages=MESSAGES)
    assert ok.choices[0].message.content.startswith("reply#2")
    assert _metrics(proxy)["requests"]["error"] == 1


def test_passthrough_and_metrics_endpoints(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    assert [m.id for m in client.models.list()] == ["stub-model"]
    emb = client.embeddings.create(model="e", input="x")
    assert emb.data[0].embedding == [1.0, 0.0]
    with urllib.request.urlopen(proxy.address + "/metrics") as r:
        text = r.read().decode()
    assert 'continuum_proxy_requests_total{outcome="bypass"}' in text
    with urllib.request.urlopen(proxy.address + "/health") as r:
        assert json.loads(r.read())["status"] == "ok"
    req = urllib.request.Request(
        proxy.address + "/v1/chat/completions",
        data=b"{not json",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400


def test_unreachable_upstream_is_502() -> None:
    p = ContinuumProxy(ProxyConfig(upstream="http://127.0.0.1:9", timeout=2), port=0).start()
    try:
        client = _client(p)
        with pytest.raises(openai.InternalServerError):
            client.chat.completions.create(model="m", messages=MESSAGES)
        with pytest.raises(openai.InternalServerError):
            client.models.list()
    finally:
        p.close()


def test_semantic_tier_opt_in(upstream: Upstream) -> None:
    with pytest.raises(ValueError, match="embedder"):
        ContinuumProxy(ProxyConfig(upstream=upstream.url, semantic_threshold=0.9), port=0)

    def key(msg: str) -> str:
        return "".join(
            request_key_parts(
                "chat", {"model": "m", "messages": [{"role": "user", "content": msg}]}
            )
        )

    vectors = {key("reset password"): [1.0, 0.0], key("forgot password"): [1.0, 0.0]}
    emb = PrecomputedEmbeddingProvider(vectors, "test")
    p = ContinuumProxy(
        ProxyConfig(upstream=upstream.url, semantic_threshold=0.95, embedder=emb), port=0
    ).start()
    try:
        client = _client(p)
        a = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "reset password"}]
        )
        raw = client.chat.completions.with_raw_response.create(
            model="m", messages=[{"role": "user", "content": "forgot password"}]
        )
        assert raw.headers["x-continuum-served-by"] == "semantic"
        assert raw.parse().choices[0].message.content == a.choices[0].message.content
    finally:
        p.close()


def test_helpers() -> None:
    parts = request_key_parts(
        "chat", {"model": "m", "stream": True, "user": "u", "messages": MESSAGES}
    )
    assert parts[0].startswith("system: You are")
    assert '"stream"' not in parts[-1] and '"user"' not in parts[-1]
    assert request_key_parts("completion", {"prompt": ["a", "b"]})[0] == '["a", "b"]'
    events = list(
        replay_as_stream("completion", {"id": "x", "choices": [{"index": 0, "text": "hi"}]})
    )
    assert events[-1] == b"data: [DONE]\n\n"


def test_streamed_bypass_is_passed_through(proxy: ContinuumProxy, upstream: Upstream) -> None:
    client = _client(proxy)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    # The stub answers tool requests without streaming; ask for a plain stream
    # with Cache-Control instead so the SSE body itself is forwarded.
    chunks = client.chat.completions.create(
        model="m", messages=MESSAGES, stream=True, extra_headers={"Cache-Control": "no-store"}
    )
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices).startswith(
        "reply#1"
    )
    client.chat.completions.create(model="m", messages=MESSAGES, tools=tools)
    assert _metrics(proxy)["requests"]["bypass"] == 2


def test_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--semantic-threshold", "0.9"])
    seen: dict[str, Any] = {}

    def fake_embedder(base: str, model: str, api_key: str | None = None) -> Any:
        seen.update(base=base, model=model)
        return PrecomputedEmbeddingProvider({"x": [1.0]}, "cli")

    monkeypatch.setattr("continuum.embeddings.OpenAICompatibleEmbeddingProvider", fake_embedder)
    monkeypatch.setattr(ContinuumProxy, "serve_forever", lambda self: self.httpd.server_close())
    main(
        [
            "--port",
            "0",
            "--upstream",
            "http://up.example/v1/",
            "--semantic-threshold",
            "0.9",
            "--embed-model",
            "nomic",
        ]
    )
    assert seen == {"base": "http://up.example", "model": "nomic"}
    assert "export OPENAI_BASE_URL=http://127.0.0.1:" in capsys.readouterr().out
