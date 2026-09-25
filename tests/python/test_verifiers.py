"""Semantic-cache hit verification and the WordLlama embedder."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from continuum._native import BackendRegistry, SemanticCacheIndex, Session
from continuum.embeddings import PrecomputedEmbeddingProvider, WordLlamaEmbeddingProvider
from continuum.verifiers import AllOf, HitVerifier, LexicalNearMissVerifier, LLMJudgeVerifier

EXPLAIN = [
    # (cached, new, accept, reason)
    ("how do I reset my password", "how do I reset my username", False, "substitution"),
    (
        "how do I enable two-factor authentication",
        "how do I disable two-factor authentication",
        False,
        "substitution",
    ),
    ("what is the capital of Australia", "what is the capital of Austria", False, "substitution"),
    ("convert 5 euros to dollars", "convert 50 euros to dollars", False, "numbers"),
    ("book a table for two", "book a table for four", False, "numbers"),
    ("turn on notifications", "turn off notifications", False, "polarity"),
    ("when is the next train to Boston", "when is the next train from Boston", False, "polarity"),
    ("show me hotels with a pool", "show me hotels without a pool", False, "polarity"),
    ("is the store open", "isn't the store open", False, "negation"),
    ("convert 10 miles to kilometers", "convert 10 kilometers to miles", False, "direction"),
    ("how to sort a list in python", "sorting a python list", True, "reordered"),
    ("summarize this article", "summary of this article", True, "same"),
    ("how do I reset my password", "I forgot my password and need to reset it", True, "ok"),
    ("what is the refund policy", "what is the refund policy", True, "same"),
]


@pytest.mark.parametrize(("a", "b", "accept", "reason"), EXPLAIN)
def test_lexical_rules(a: str, b: str, accept: bool, reason: str) -> None:
    verdict = LexicalNearMissVerifier.explain(a, b)
    assert verdict == {"accept": accept, "reason": reason}
    assert LexicalNearMissVerifier().verify(a, b, 0.99) is accept


def _index(threshold: float = 0.9) -> SemanticCacheIndex:
    return SemanticCacheIndex(16, threshold)


def test_index_verifies_and_falls_back_to_next_candidate() -> None:
    idx = _index()
    assert idx.verifier_name() == "lexical-near-miss-v1"
    idx.insert([1.0, 0.0], "m", b"username-answer", prompt="how do I reset my username")
    idx.insert([0.97, 0.243], "m", b"password-answer", prompt="how do I reset my password")
    # The closest entry (similarity 1.0) is a one-word edit of the query; the
    # verifier refuses it and the next candidate is served.
    r = idx.lookup([1.0, 0.0], "m", query_prompt="how do I reset my password")
    assert r["above_threshold"]
    assert r["output"] == b"password-answer"  # the username entry was rejected first
    assert r["prompt"] == "how do I reset my password"
    assert r["verifier_rejections"] == 1
    assert idx.verifier_rejections() == 1

    idx.set_verifier(None)
    assert idx.verifier_name() is None
    assert (
        idx.lookup([1.0, 0.0], "m", query_prompt="how do I reset my password")["output"]
        == b"username-answer"
    )
    idx.clear()
    assert idx.verifier_rejections() == 0


def test_unverifiable_candidates_are_served() -> None:
    idx = _index()
    idx.insert([1.0, 0.0], "m", b"x")  # no prompt stored: nothing to verify against
    assert idx.lookup([1.0, 0.0], "m", query_prompt="anything")["above_threshold"]


class Counting(HitVerifier):
    def __init__(self, answer: bool) -> None:
        super().__init__()
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def verify(self, cached_prompt: str, new_prompt: str, similarity: float = 1.0) -> bool:
        self.calls.append((cached_prompt, new_prompt))
        return self.answer

    def name(self) -> str:
        return f"counting-{self.answer}"


def _session(embedder: Any, sc: SemanticCacheIndex) -> Session:
    reg = BackendRegistry()
    reg.register_fake_llm()
    s = Session("verify", reg)
    s.set_semantic_cache(sc)
    s.set_embedding_provider(embedder)
    return s


def test_python_verifier_gates_session_hits_once_per_pair() -> None:
    emb = PrecomputedEmbeddingProvider(
        {"alpha question": [1.0, 0.0], "alpha query": [1.0, 0.0]}, "t"
    )
    for answer in (False, True):
        sc = _index(0.95)
        verifier = Counting(answer)
        sc.set_verifier(verifier)
        s = _session(emb, sc)
        s.generate(["alpha question"], "fake/m", 4)
        s.generate(["alpha query"], "fake/m", 4)
        hits = [step.semantic_hit for step in s.metrics().steps]
        assert hits == [False, answer]
        # The session's metrics pass and the interpreter share one verdict.
        assert verifier.calls == [("alpha question", "alpha query")]


class _Judge:
    def __init__(self) -> None:
        self.reply = "YES"
        self.requests: list[dict[str, Any]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(
                    {"path": self.path, "auth": self.headers.get("Authorization"), **body}
                )
                raw = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": outer.reply}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"


@pytest.fixture
def judge() -> Iterator[_Judge]:
    j = _Judge()
    yield j
    j.httpd.shutdown()


def test_llm_judge(judge: _Judge) -> None:
    v = LLMJudgeVerifier(judge.url, "gemma4", api_key="k")
    assert v.name() == "llm-judge/gemma4"
    assert v.verify("cancel my gym membership", "end my gym membership") is True
    judge.reply = " no."
    assert v.verify("enable 2fa", "disable 2fa") is False
    assert v.calls == 2
    req = judge.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["model"] == "gemma4" and req["temperature"] == 0
    assert "cancel my gym membership" in req["messages"][0]["content"]
    assert req["auth"] == "Bearer k"
    assert LLMJudgeVerifier(judge.url + "/v1", "m")._url == judge.url + "/v1/chat/completions"
    # Unreachable judge: refuse the hit (a miss is safe, a wrong answer is not).
    assert LLMJudgeVerifier("http://127.0.0.1:9", "m", timeout=1).verify("a", "b") is False


def test_llm_judge_in_index(judge: _Judge) -> None:
    idx = _index()
    idx.set_verifier(LLMJudgeVerifier(judge.url, "m"))
    idx.insert([1.0, 0.0], "m", b"ans", prompt="cancel my gym membership")
    assert idx.lookup([1.0, 0.0], "m", query_prompt="end my gym membership")["above_threshold"]
    judge.reply = "NO"
    assert not idx.lookup([1.0, 0.0], "m", query_prompt="freeze my gym membership")[
        "above_threshold"
    ]


def test_all_of() -> None:
    yes, no = Counting(True), Counting(False)
    assert AllOf(yes, yes).verify("a", "b") is True
    assert AllOf(no, yes).verify("a", "b") is False
    assert yes.calls and not [c for c in yes.calls if c != ("a", "b")]
    assert AllOf(yes, no).name() == "all-of(counting-True,counting-False)"
    with pytest.raises(ValueError):
        AllOf()


def test_wordllama_embedder_with_verifier() -> None:
    pytest.importorskip("wordllama")
    emb = WordLlamaEmbeddingProvider()
    assert emb.dimension() == 256 and emb.identity() == "wordllama/l2_supercat:256"
    v = emb.embed("how do I reset my password")
    assert len(v) == 256 and abs(sum(x * x for x in v) - 1.0) < 1e-4

    s = _session(emb, _index(0.7))
    s.generate(["how do I reset my password"], "fake/m", 4)
    s.generate(["I forgot my password and need to reset it"], "fake/m", 4)  # rewording
    s.generate(["how do I reset my username"], "fake/m", 4)  # near-miss
    s.generate(["what is the refund policy"], "fake/m", 4)  # unrelated
    assert [step.semantic_hit for step in s.metrics().steps] == [False, True, False, False]
