"""OpenAI-compatible proxy that runs requests through Continuum's reuse stack.

Point any OpenAI-format client at it and it gains Continuum's caching with no
code changes::

    python -m continuum.proxy --upstream https://api.openai.com/v1 --port 8787
    export OPENAI_BASE_URL=http://localhost:8787/v1

The OpenAI SDK, LangChain (``ChatOpenAI``), LlamaIndex (``OpenAI``), and plain
HTTP all work, because the proxy speaks the same wire format as the upstream.
Any OpenAI-compatible upstream works: OpenAI, Azure OpenAI's ``/openai/v1``,
vLLM, Ollama (``http://localhost:11434/v1``), LM Studio, ...

**What is reused.** ``POST /v1/chat/completions`` and ``POST /v1/completions``
requests become one Continuum ``TokenOp`` each: the messages (or prompt) and the
sampling parameters are the key, and the upstream call is the backend behind
the memo tier (exact repeats), the prefix-KV tier (shared prefixes, tracked as
metrics), and, if enabled, the semantic tier. A cache hit returns the stored
response without calling the upstream.

**What is never cached.** Requests that carry ``tools`` / ``functions`` or
tool-result messages, requests with ``n > 1``, and requests sent with
``Cache-Control: no-cache`` / ``no-store`` are forwarded untouched. Every other
path (``/v1/models``, embeddings, ...) is passed straight through.

**Streaming.** ``"stream": true`` works both ways: a miss streams the
upstream's server-sent events through as they arrive (and caches the assembled
response), and a hit is replayed as a server-sent-event stream.

**Metrics.** Each response carries ``x-continuum-cache`` (``hit`` / ``miss`` /
``bypass``), ``x-continuum-served-by``, and ``x-continuum-tokens-saved``
headers. ``GET /metrics`` returns Prometheus text and ``GET /continuum/metrics``
returns JSON: per-tier lookups and hits, requests, and tokens saved.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from continuum._native import (
    BackendRegistry,
    EmbeddingProvider,
    KVCacheIndex,
    MemoTable,
    ReuseEvent,
    ReuseEventKind,
    ReuseObserver,
    SemanticCacheIndex,
    Session,
)

__all__ = ["ContinuumProxy", "ProxyConfig", "main"]

DEFAULT_UPSTREAM = "https://api.openai.com/v1"
ENDPOINTS = {"chat/completions": "chat", "completions": "completion"}
# Request fields that change transport, not the answer.
_VOLATILE_FIELDS = ("stream", "stream_options", "user", "metadata", "store", "service_tier")
_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host"}


@dataclass
class ProxyConfig:
    """Proxy settings.

    Attributes:
        upstream: Base URL of the OpenAI-compatible upstream, including ``/v1``.
        api_key: Used when the client sends no ``Authorization`` header.
        memo_entries: Capacity of the exact-match tier.
        semantic_threshold: Enables the semantic tier at this similarity when
            set. Off by default: see ``benchmarks/reports/semantic-false-hits.md``
            before enabling it, and pass a real ``embedder``.
        embedder: Embedding provider for the semantic tier.
        timeout: Upstream timeout in seconds.
    """

    upstream: str = DEFAULT_UPSTREAM
    api_key: str | None = None
    memo_entries: int = 4096
    prefix_entries: int = 8192
    semantic_threshold: float | None = None
    semantic_entries: int = 2048
    embedder: EmbeddingProvider | None = None
    timeout: float = 600.0


class _UpstreamError(Exception):
    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        super().__init__(f"upstream returned HTTP {status}")
        self.status = status
        self.body = body
        self.headers = headers


@dataclass
class _RequestContext:
    """Per-request state shared between the handler and the backend callable."""

    handler: _Handler
    kind: str
    body: dict[str, Any]
    stream: bool
    backend_called: bool = False
    streamed: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class _Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = {"hit": 0, "miss": 0, "bypass": 0, "error": 0}
        self.tier_lookups: dict[str, int] = {}
        self.tier_hits: dict[str, int] = {}
        self.tokens_saved = 0

    def request(self, outcome: str) -> None:
        with self._lock:
            self.requests[outcome] += 1

    def tier(self, tier: str, hit: bool, tokens_saved: int) -> None:
        with self._lock:
            self.tier_lookups[tier] = self.tier_lookups.get(tier, 0) + 1
            if hit:
                self.tier_hits[tier] = self.tier_hits.get(tier, 0) + 1
            self.tokens_saved += tokens_saved

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self.requests),
                "tier_lookups": dict(self.tier_lookups),
                "tier_hits": dict(self.tier_hits),
                "tokens_saved": self.tokens_saved,
            }

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP continuum_proxy_requests_total Proxied completion requests by cache outcome.",
            "# TYPE continuum_proxy_requests_total counter",
        ]
        lines += [f'continuum_proxy_requests_total{{outcome="{k}"}} {v}' for k, v in snap["requests"].items()]
        lines += [
            "# HELP continuum_reuse_lookups_total Reuse-tier lookups.",
            "# TYPE continuum_reuse_lookups_total counter",
        ]
        lines += [f'continuum_reuse_lookups_total{{tier="{k}"}} {v}' for k, v in snap["tier_lookups"].items()]
        lines += [
            "# HELP continuum_reuse_hits_total Reuse-tier lookups that matched.",
            "# TYPE continuum_reuse_hits_total counter",
        ]
        lines += [f'continuum_reuse_hits_total{{tier="{k}"}} {v}' for k, v in snap["tier_hits"].items()]
        lines += [
            "# HELP continuum_reuse_tokens_saved_total Prompt tokens served from reuse tiers.",
            "# TYPE continuum_reuse_tokens_saved_total counter",
            f"continuum_reuse_tokens_saved_total {snap['tokens_saved']}",
        ]
        return "\n".join(lines) + "\n"


class _EventSink(ReuseObserver):
    def __init__(self, proxy: ContinuumProxy) -> None:
        super().__init__()
        self._proxy = proxy

    def on_event(self, event: ReuseEvent) -> None:
        ctx = self._proxy._local.__dict__.get("ctx")
        if event.kind == ReuseEventKind.TierLookup:
            self._proxy.stats.tier(event.tier, event.hit, event.tokens_saved if event.hit else 0)
        if ctx is not None:
            ctx.events.append(
                {
                    "kind": "tier" if event.kind == ReuseEventKind.TierLookup else "node",
                    "tier": event.tier,
                    "hit": event.hit,
                    "served_by": event.served_by,
                    "tokens_saved": event.tokens_saved,
                }
            )


def cache_bypass_reason(body: dict[str, Any], headers: Any) -> str | None:
    """Why a completion request must skip the cache, or None if cacheable."""
    if body.get("tools") or body.get("functions") or body.get("tool_choice") or body.get("function_call"):
        return "tools"
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and (msg.get("role") in ("tool", "function") or msg.get("tool_calls")):
            return "tool-messages"
    if int(body.get("n") or 1) > 1:
        return "n>1"
    cache_control = (headers.get("Cache-Control") or "").lower()
    if "no-cache" in cache_control or "no-store" in cache_control:
        return "cache-control"
    return None


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, list):  # multi-part content: keep text parts, key the rest
        content = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return f"{msg.get('role', '')}: {content if content is not None else ''}"


def request_key_parts(kind: str, body: dict[str, Any]) -> list[str]:
    """The prompt parts a request is keyed on: its content first, so requests
    sharing a system prompt share a prefix, then its canonical parameters."""
    params = {k: v for k, v in body.items() if k not in _VOLATILE_FIELDS and k not in ("messages", "prompt")}
    if kind == "chat":
        parts = [_message_text(m) for m in body.get("messages") or [] if isinstance(m, dict)]
    else:
        prompt = body.get("prompt", "")
        parts = [prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)]
    parts.append(json.dumps({"endpoint": kind, **params}, sort_keys=True, separators=(",", ":")))
    return parts


def _sse(obj: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def replay_as_stream(kind: str, response: dict[str, Any]) -> Iterator[bytes]:
    """Server-sent events equivalent to a stored non-streaming response."""
    base = {"id": response.get("id", ""), "created": response.get("created", int(time.time())),
            "model": response.get("model", "")}
    for choice in response.get("choices", []):
        idx = choice.get("index", 0)
        if kind == "chat":
            msg = choice.get("message") or {}
            obj = "chat.completion.chunk"
            yield _sse({**base, "object": obj, "choices": [
                {"index": idx, "delta": {"role": msg.get("role", "assistant"), "content": ""}, "finish_reason": None}]})
            yield _sse({**base, "object": obj, "choices": [
                {"index": idx, "delta": {"content": msg.get("content") or ""}, "finish_reason": None}]})
            yield _sse({**base, "object": obj, "choices": [
                {"index": idx, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]})
        else:
            yield _sse({**base, "object": "text_completion", "choices": [
                {"index": idx, "text": choice.get("text", ""), "finish_reason": choice.get("finish_reason", "stop")}]})
    yield b"data: [DONE]\n\n"


class _StreamAssembler:
    """Rebuild a non-streaming response from upstream SSE chunks."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.meta: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: dict[str, Any]) -> None:
        for key in ("id", "created", "model", "system_fingerprint"):
            if key in chunk and key not in self.meta:
                self.meta[key] = chunk[key]
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for c in chunk.get("choices", []):
            slot = self.choices.setdefault(c.get("index", 0), {"text": "", "role": "assistant", "finish": None})
            if self.kind == "chat":
                delta = c.get("delta") or {}
                slot["role"] = delta.get("role") or slot["role"]
                slot["text"] += delta.get("content") or ""
            else:
                slot["text"] += c.get("text") or ""
            if c.get("finish_reason"):
                slot["finish"] = c["finish_reason"]

    def response(self) -> dict[str, Any]:
        out: dict[str, Any] = {**self.meta, "object": "chat.completion" if self.kind == "chat" else "text_completion"}
        choices = []
        for idx in sorted(self.choices):
            slot = self.choices[idx]
            if self.kind == "chat":
                choices.append({"index": idx, "message": {"role": slot["role"], "content": slot["text"]},
                                "finish_reason": slot["finish"]})
            else:
                choices.append({"index": idx, "text": slot["text"], "finish_reason": slot["finish"]})
        out["choices"] = choices
        if self.usage is not None:
            out["usage"] = self.usage
        return out


class ContinuumProxy:
    """The proxy server. ``serve_forever()`` blocks; ``start()`` runs it on a thread."""

    def __init__(self, config: ProxyConfig | None = None, host: str = "127.0.0.1", port: int = 8787) -> None:
        self.config = config or ProxyConfig()
        self.config.upstream = self.config.upstream.rstrip("/")
        self.stats = _Stats()
        self.memo = MemoTable(self.config.memo_entries, 0)
        self.prefix_cache = KVCacheIndex(self.config.prefix_entries)
        self.semantic: SemanticCacheIndex | None = None
        if self.config.semantic_threshold is not None:
            if self.config.embedder is None:
                raise ValueError("the semantic tier needs an embedder (ProxyConfig.embedder)")
            self.semantic = SemanticCacheIndex(self.config.semantic_entries, self.config.semantic_threshold)
        self._local = threading.local()
        self._observer = _EventSink(self)
        proxy = self

        class Handler(_Handler):
            owner = proxy

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> ContinuumProxy:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- sessions ------------------------------------------------------------

    def _session(self) -> Session:
        """One session per server thread; every reuse tier is shared across them."""
        session: Session | None = getattr(self._local, "session", None)
        if session is None:
            registry = BackendRegistry()
            registry.register_python("upstream", self._call_upstream, 100)
            session = Session(f"proxy-{threading.get_ident()}", registry, self.prefix_cache)
            session.set_memo_table(self.memo)
            if self.semantic is not None:
                session.set_semantic_cache(self.semantic)
                session.set_embedding_provider(self.config.embedder)
            session.set_observer(self._observer)
            self._local.registry = registry
            self._local.session = session
        return session

    # -- upstream --------------------------------------------------------------

    def upstream_request(self, handler: _Handler, path: str, body: bytes | None, method: str) -> Any:
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_HEADERS}
        headers["Accept-Encoding"] = "identity"
        if "authorization" not in {k.lower() for k in headers} and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(self.config.upstream + path, data=body, headers=headers, method=method)
        try:
            return urllib.request.urlopen(req, timeout=self.config.timeout)
        except urllib.error.HTTPError as err:
            raise _UpstreamError(err.code, err.read(), dict(err.headers.items())) from None

    def _call_upstream(self, req: dict[str, Any]) -> dict[str, Any]:
        ctx: _RequestContext = self._local.ctx
        ctx.backend_called = True
        path = "/chat/completions" if ctx.kind == "chat" else "/completions"
        payload = dict(ctx.body)
        payload["stream"] = ctx.stream  # stream_options pass through as the client sent them
        resp = self.upstream_request(ctx.handler, path, json.dumps(payload).encode(), "POST")
        with resp:
            if not ctx.stream:
                data = json.loads(resp.read())
            else:
                assembler = _StreamAssembler(ctx.kind)
                ctx.handler.begin_stream(cache="miss", served_by="backend")
                ctx.streamed = True
                for raw in resp:
                    ctx.handler.write_chunk(raw)
                    line = raw.strip()
                    if line.startswith(b"data:") and line[5:].strip() not in (b"[DONE]", b""):
                        assembler.feed(json.loads(line[5:]))
                ctx.handler.end_stream()
                data = assembler.response()
        usage = data.get("usage") or {}
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        out: dict[str, Any] = {"output": json.dumps(data, separators=(",", ":"))}
        if cached:
            out["tokens_saved"] = cached
        return out

    # -- request handling ------------------------------------------------------

    def handle_completion(self, handler: _Handler, kind: str) -> None:
        raw = handler.rfile.read(int(handler.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            handler.send_json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            return
        path = "/chat/completions" if kind == "chat" else "/completions"
        if not isinstance(body, dict) or cache_bypass_reason(body, handler.headers) is not None:
            self.stats.request("bypass")
            handler.forward(path, raw, "POST", cache="bypass")
            return

        stream = bool(body.get("stream"))
        temperature = body.get("temperature")
        ctx = _RequestContext(handler=handler, kind=kind, body=body, stream=stream)
        self._local.ctx = ctx
        try:
            output = self._session().generate(
                request_key_parts(kind, body),
                f"{kind}:{body.get('model', '')}",
                int(body.get("max_tokens") or body.get("max_completion_tokens") or 0),
                float(temperature) if isinstance(temperature, (int, float)) else 1.0,
            )
        except _UpstreamError as err:
            self.stats.request("error")
            if not ctx.streamed:
                handler.send_raw(err.status, err.body, err.headers, cache="miss")
            return
        except (urllib.error.URLError, OSError, ValueError) as err:
            self.stats.request("error")
            if not ctx.streamed:
                handler.send_json(502, {"error": {"message": f"continuum proxy: {err}", "type": "upstream_error"}})
            return
        finally:
            self._local.ctx = None
            self._session().reset_metrics()  # per-request records would grow forever

        node = next((e for e in reversed(ctx.events) if e["kind"] == "node"), {})
        served_by = node.get("served_by", "backend")
        saved = int(node.get("tokens_saved", 0))
        outcome = "miss" if ctx.backend_called else "hit"
        self.stats.request(outcome)
        if ctx.streamed:
            return
        response = json.loads(output)
        if outcome == "hit":
            response = copy.deepcopy(response)
            prefix = "chatcmpl-" if kind == "chat" else "cmpl-"
            response["id"] = prefix + "continuum-" + uuid.uuid4().hex[:20]
        extra = {"x-continuum-served-by": served_by, "x-continuum-tokens-saved": str(saved)}
        if stream:
            handler.begin_stream(cache=outcome, served_by=served_by, tokens_saved=saved)
            for chunk in replay_as_stream(kind, response):
                handler.write_chunk(chunk)
            handler.end_stream()
        else:
            handler.send_json(200, response, cache=outcome, extra=extra)


class _Handler(BaseHTTPRequestHandler):
    owner: ContinuumProxy
    protocol_version = "HTTP/1.1"
    server_version = "continuum-proxy"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        if os.environ.get("CONTINUUM_PROXY_ACCESS_LOG"):
            super().log_message(format, *args)

    def _route(self) -> tuple[str, str | None]:
        path = self.path.split("?", 1)[0]
        stripped = path[len("/v1/"):] if path.startswith("/v1/") else path.lstrip("/")
        return stripped, ENDPOINTS.get(stripped)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        route, kind = self._route()
        if kind is not None:
            self.owner.handle_completion(self, kind)
            return
        length = int(self.headers.get("Content-Length") or 0)
        self.forward("/" + route, self.rfile.read(length), "POST", cache="bypass")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            body = self.owner.stats.prometheus().encode()
            self.send_raw(200, body, {"Content-Type": "text/plain; version=0.0.4"})
            return
        if path in ("/continuum/metrics", "/v1/continuum/metrics"):
            self.send_json(200, self.owner.stats.snapshot())
            return
        if path in ("/health", "/healthz"):
            self.send_json(200, {"status": "ok", "upstream": self.owner.config.upstream})
            return
        route, _ = self._route()
        self.forward("/" + route, None, "GET", cache="bypass")

    # -- response helpers ----------------------------------------------------

    def forward(self, path: str, body: bytes | None, method: str, cache: str) -> None:
        try:
            resp = self.owner.upstream_request(self, path, body, method)
        except _UpstreamError as err:
            self.send_raw(err.status, err.body, err.headers, cache=cache)
            return
        except (urllib.error.URLError, OSError) as err:
            self.send_json(502, {"error": {"message": f"continuum proxy: {err}", "type": "upstream_error"}})
            return
        with resp:
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_HEADERS}
            if "text/event-stream" in resp.headers.get("Content-Type", ""):
                self.send_response(resp.status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("x-continuum-cache", cache)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for raw in resp:
                    self.write_chunk(raw)
                self.end_stream()
            else:
                self.send_raw(resp.status, resp.read(), headers, cache=cache)

    def send_raw(self, status: int, body: bytes, headers: dict[str, str] | None = None,
                 cache: str | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            if k.lower() not in _HOP_HEADERS:
                self.send_header(k, v)
        if cache is not None:
            self.send_header("x-continuum-cache", cache)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, obj: dict[str, Any], cache: str | None = None,
                  extra: dict[str, str] | None = None) -> None:
        headers = {"Content-Type": "application/json", **(extra or {})}
        self.send_raw(status, json.dumps(obj).encode(), headers, cache=cache)

    def begin_stream(self, cache: str, served_by: str, tokens_saved: int = 0) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("x-continuum-cache", cache)
        self.send_header("x-continuum-served-by", served_by)
        self.send_header("x-continuum-tokens-saved", str(tokens_saved))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def write_chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m continuum.proxy", description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", default=os.environ.get("CONTINUUM_PROXY_UPSTREAM", DEFAULT_UPSTREAM),
                    help="OpenAI-compatible base URL including /v1 (env CONTINUUM_PROXY_UPSTREAM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--api-key", default=os.environ.get("CONTINUUM_PROXY_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                    help="used when a client sends no Authorization header")
    ap.add_argument("--memo-entries", type=int, default=4096)
    ap.add_argument("--semantic-threshold", type=float, default=None,
                    help="enable the semantic tier (needs --embed-model)")
    ap.add_argument("--embed-model", help="embedding model on the upstream for the semantic tier")
    args = ap.parse_args(argv)

    embedder = None
    if args.semantic_threshold is not None:
        if not args.embed_model:
            ap.error("--semantic-threshold needs --embed-model")
        from continuum.embeddings import OpenAICompatibleEmbeddingProvider

        upstream = args.upstream.rstrip("/")
        base = upstream[: -len("/v1")] if upstream.endswith("/v1") else upstream
        embedder = OpenAICompatibleEmbeddingProvider(base, args.embed_model, api_key=args.api_key)
    config = ProxyConfig(upstream=args.upstream, api_key=args.api_key, memo_entries=args.memo_entries,
                         semantic_threshold=args.semantic_threshold, embedder=embedder)
    proxy = ContinuumProxy(config, host=args.host, port=args.port)
    print(f"continuum proxy on {proxy.address}/v1 -> {config.upstream}", flush=True)
    print(f"  export OPENAI_BASE_URL={proxy.address}/v1", flush=True)
    try:
        proxy.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
