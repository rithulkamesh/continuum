from __future__ import annotations

from typing import Any

import pytest
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from continuum.embeddings import PrecomputedEmbeddingProvider
from continuum_langchain import ContinuumCache, ContinuumLLM


class CountingLLM(ContinuumLLM):
    calls: int = 0

    def _call(self, prompt: str, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        self.calls += 1
        return super()._call(prompt, stop, run_manager, **kwargs)


@pytest.fixture(autouse=True)
def _reset_global_cache() -> Any:
    yield
    set_llm_cache(None)


def test_llm_repeat_served_by_cache() -> None:
    cache = ContinuumCache()
    set_llm_cache(cache)
    llm = CountingLLM(max_tokens=8)
    a = llm.invoke("summarize the ticket")
    b = llm.invoke("summarize the ticket")
    llm.invoke("summarize the other ticket")
    assert a == b
    assert llm.calls == 2
    assert cache.stats["memo_hits"] == 1
    # A differently configured model is a different key.
    CountingLLM(max_tokens=9).invoke("summarize the ticket")
    assert cache.stats["misses"] == 3


def test_chat_generations_round_trip_and_tool_calls_skipped() -> None:
    cache = ContinuumCache()
    msg = AIMessage(content="hello", response_metadata={"model": "x"})
    cache.update("p", "llm", [ChatGeneration(message=msg, generation_info={"finish_reason": "stop"})])
    (got,) = cache.lookup("p", "llm") or []
    assert isinstance(got, ChatGeneration)
    assert got.message.content == "hello"
    assert got.generation_info == {"finish_reason": "stop"}

    tool_msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])
    cache.update("q", "llm", [ChatGeneration(message=tool_msg)])
    assert cache.lookup("q", "llm") is None
    cache.update("r", "[('tools', [...])]", [Generation(text="plain")])
    assert cache.lookup("r", "[('tools', [...])]") is None
    assert cache.stats["skipped_tool_calls"] == 2

    cache.clear()
    assert cache.lookup("p", "llm") is None


def test_chat_model_tool_calls_never_cached() -> None:
    cache = ContinuumCache()
    tool_msg = AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "c1"}])
    model = FakeMessagesListChatModel(responses=[tool_msg, tool_msg], cache=cache)
    model.invoke("find x")
    model.invoke("find x")
    assert cache.stats["memo_hits"] == 0
    assert cache.stats["skipped_tool_calls"] == 2


def test_semantic_tier() -> None:
    with pytest.raises(ValueError):
        ContinuumCache(semantic_threshold=0.9)
    emb = PrecomputedEmbeddingProvider({"reset password": [1.0, 0.0], "forgot password": [1.0, 0.0],
                                        "refund policy": [0.0, 1.0]}, "test")
    cache = ContinuumCache(semantic_threshold=0.95, embedder=emb)
    cache.update("reset password", "llm", [Generation(text="use the reset link")])
    assert cache.lookup("forgot password", "llm")[0].text == "use the reset link"  # type: ignore[index]
    assert cache.lookup("refund policy", "llm") is None
    assert cache.lookup("forgot password", "other-llm") is None
    assert cache.stats["semantic_hits"] == 1
    cache.clear()
    assert cache.lookup("forgot password", "llm") is None


def test_llm_stop_and_params() -> None:
    llm = ContinuumLLM(max_tokens=4)
    text = llm.invoke("abc")
    assert len(text.split()) == 4
    first = text.split()[1]
    assert first not in llm.invoke("abc", stop=[first])
    assert llm._identifying_params["max_tokens"] == 4
    assert llm._llm_type == "continuum"
