"""Pluggable embedding providers for the semantic tier (issue #9)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from continuum._native import (
    BackendRegistry,
    BruteForceEmbeddingProvider,
    EmbeddingProvider,
    SemanticCacheIndex,
    Session,
)
from continuum.embeddings import (
    CallableEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    PrecomputedEmbeddingProvider,
)

# Two paraphrases share a vector; the unrelated prompt is orthogonal.
VECTORS = {
    "how do I reset my password": [1.0, 0.0, 0.0],
    "I forgot my password, how to reset it": [1.0, 0.0, 0.0],
    "what is the refund policy": [0.0, 1.0, 0.0],
}


def _session(embedder: EmbeddingProvider, sc: SemanticCacheIndex) -> Session:
    reg = BackendRegistry()
    reg.register_fake_llm()
    s = Session("emb", reg)
    s.set_semantic_cache(sc)
    s.set_embedding_provider(embedder)
    return s


def _semantic_hits(s: Session) -> list[bool]:
    return [step.semantic_hit for step in s.metrics().steps]


def test_python_provider_drives_semantic_hits() -> None:
    sc = SemanticCacheIndex(16, 0.95)
    s = _session(PrecomputedEmbeddingProvider(VECTORS, "test/precomputed-v1"), sc)
    first = s.generate(["how do I reset my password"], "fake/m", 8)
    paraphrase = s.generate(["I forgot my password, how to reset it"], "fake/m", 8)
    s.generate(["what is the refund policy"], "fake/m", 8)
    assert _semantic_hits(s) == [False, True, False]
    assert paraphrase == first  # served from the semantic tier


def test_embedder_identity_is_part_of_the_key() -> None:
    sc = SemanticCacheIndex(16, 0.95)
    a = PrecomputedEmbeddingProvider(VECTORS, "embedder-a")
    b = PrecomputedEmbeddingProvider(VECTORS, "embedder-b")  # same vectors, other space
    _session(a, sc).generate(["how do I reset my password"], "fake/m", 8)
    s_b = _session(b, sc)
    s_b.generate(["how do I reset my password"], "fake/m", 8)
    assert _semantic_hits(s_b) == [False]
    s_a = _session(a, sc)
    s_a.generate(["how do I reset my password"], "fake/m", 8)
    assert _semantic_hits(s_a) == [True]


def test_index_filters_on_embedder_id() -> None:
    sc = SemanticCacheIndex(4, 0.9)
    sc.insert([1.0, 0.0], "m", b"x", embedder_id="e1")
    assert sc.lookup([1.0, 0.0], "m", embedder_id="e1")["above_threshold"]
    assert not sc.lookup([1.0, 0.0], "m", embedder_id="e2")["above_threshold"]
    assert not sc.lookup([1.0, 0.0], "m")["above_threshold"]


def test_callable_provider_validates() -> None:
    p = CallableEmbeddingProvider(lambda t: [float(len(t)), 1.0], 2, "len-v1")
    assert p.embed("abc") == [3.0, 1.0]
    assert p.dimension() == 2
    assert p.identity() == "len-v1"
    bad = CallableEmbeddingProvider(lambda t: [1.0], 2, "bad")
    with pytest.raises(ValueError, match="1 dims"):
        bad.embed("x")
    with pytest.raises(ValueError):
        CallableEmbeddingProvider(lambda t: [1.0], 0, "x")
    with pytest.raises(ValueError):
        CallableEmbeddingProvider(lambda t: [1.0], 1, "")


def test_precomputed_provider_fallback_and_errors() -> None:
    p = PrecomputedEmbeddingProvider({"a": [1, 0]}, "pre")
    assert p.embed("a") == [1.0, 0.0]
    assert p.dimension() == 2
    with pytest.raises(KeyError):
        p.embed("missing")
    fb = CallableEmbeddingProvider(lambda t: [0.0, 1.0], 2, "pre")
    assert PrecomputedEmbeddingProvider({"a": [1, 0]}, "pre", fb).embed("zz") == [0.0, 1.0]
    with pytest.raises(ValueError, match="share"):
        PrecomputedEmbeddingProvider({"a": [1, 0]}, "pre", BruteForceEmbeddingProvider(2))
    with pytest.raises(ValueError, match="mixed"):
        PrecomputedEmbeddingProvider({"a": [1], "b": [1, 2]}, "pre")
    with pytest.raises(ValueError):
        PrecomputedEmbeddingProvider({}, "pre")


def test_builtin_provider_identity() -> None:
    assert BruteForceEmbeddingProvider(64).identity() == "continuum/char-ngram-v1:64"


@pytest.fixture
def embed_server() -> Iterator[tuple[str, list[dict]]]:
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
            raw = json.dumps({"data": [{"embedding": [0.5, 0.5, float(len(body["input"]))]}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw.encode())

        def log_message(self, *args: object) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", seen
    httpd.shutdown()


def test_openai_compatible_provider(embed_server: tuple[str, list[dict]]) -> None:
    url, seen = embed_server
    p = OpenAICompatibleEmbeddingProvider(url, "nomic-embed-text", api_key="k")
    assert p.dimension() == 3  # probed once
    assert p.identity() == "openai-compatible/nomic-embed-text"
    assert p.embed("abcd") == [0.5, 0.5, 4.0]
    assert seen[-1]["path"] == "/v1/embeddings"
    assert seen[-1]["model"] == "nomic-embed-text"
    assert seen[-1]["auth"] == "Bearer k"

    fixed = OpenAICompatibleEmbeddingProvider(url, "m", dimension=2, identity="custom")
    assert fixed.identity() == "custom"
    with pytest.raises(ValueError):
        fixed.embed("x")  # server returns 3 dims
