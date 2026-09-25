"""LangChain ``BaseCache`` backed by Continuum's memo and semantic tiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation

from continuum._native import EmbeddingProvider, MemoKey, MemoTable, SemanticCacheIndex

_TOOL_MARKERS = (
    "'tools'",
    '"tools"',
    "'functions'",
    '"functions"',
    "'tool_choice'",
    '"tool_choice"',
)


def _llm_key(llm_string: str) -> str:
    return hashlib.sha256(llm_string.encode()).hexdigest()


def _has_tool_calls(generations: Sequence[Generation]) -> bool:
    for gen in generations:
        if isinstance(gen, ChatGeneration):
            msg = gen.message
            extra = getattr(msg, "additional_kwargs", {}) or {}
            if (
                getattr(msg, "tool_calls", None)
                or extra.get("tool_calls")
                or extra.get("function_call")
            ):
                return True
    return False


def _dump(generations: Sequence[Generation]) -> bytes:
    out = []
    for gen in generations:
        item: dict[str, Any] = {"text": gen.text, "generation_info": gen.generation_info}
        if isinstance(gen, ChatGeneration):
            item["message"] = message_to_dict(gen.message)
        out.append(item)
    return json.dumps(out).encode()


def _load(raw: bytes) -> list[Generation]:
    gens: list[Generation] = []
    for item in json.loads(raw):
        if "message" in item:
            (message,) = messages_from_dict([item["message"]])
            gens.append(
                ChatGeneration(message=message, generation_info=item.get("generation_info"))
            )
        else:
            gens.append(Generation(text=item["text"], generation_info=item.get("generation_info")))
    return gens


class ContinuumCache(BaseCache):
    """LangChain cache on Continuum's reuse tiers.

    Use it like any LangChain cache::

        from langchain_core.globals import set_llm_cache
        set_llm_cache(ContinuumCache())

    Args:
        memo_entries: Capacity of the exact-match (memo) tier, LRU-evicted.
        semantic_threshold: Enables the semantic tier at this cosine
            similarity. Requires ``embedder``. Off by default: measure your
            embedder's false-hit rate first (``benchmarks/scripts/e9_semantic_false_hits.py``).
        embedder: Embedding provider for the semantic tier; its identity is
            part of the key, so switching embedders never mixes vectors.
            ``continuum.embeddings.WordLlamaEmbeddingProvider`` with
            ``semantic_threshold=0.7`` is the measured starting point.
        verifier: Hit verifier for semantic candidates; defaults to the
            engine's ``LexicalNearMissVerifier``. Pass
            ``continuum.verifiers.LLMJudgeVerifier(...)`` to also catch
            topically-related-but-different questions.
        namespace: Cache namespace, isolating tenants that share tables.
        memo: Share an existing ``MemoTable`` (e.g. with a Continuum ``Session``).

    Responses containing tool calls, and calls from models bound to tools,
    are never cached.
    """

    def __init__(
        self,
        memo_entries: int = 4096,
        semantic_threshold: float | None = None,
        embedder: EmbeddingProvider | None = None,
        namespace: str = "",
        memo: MemoTable | None = None,
        semantic_entries: int = 2048,
        verifier: Any = None,
    ) -> None:
        self.memo = memo if memo is not None else MemoTable(memo_entries, 0)
        self.semantic: SemanticCacheIndex | None = None
        if semantic_threshold is not None:
            if embedder is None:
                raise ValueError("the semantic tier needs an embedder")
            self.semantic = SemanticCacheIndex(semantic_entries, semantic_threshold)
            if verifier is not None:
                self.semantic.set_verifier(verifier)
        self.embedder = embedder
        self.namespace = namespace
        self.stats = {"memo_hits": 0, "semantic_hits": 0, "misses": 0, "skipped_tool_calls": 0}

    def _key(self, prompt: str, llm_string: str) -> MemoKey:
        return MemoKey("LangChainCache", _llm_key(llm_string), prompt.encode(), self.namespace)

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit = self.memo.lookup(self._key(prompt, llm_string))
        if hit is not None:
            self.stats["memo_hits"] += 1
            return _load(hit["output_bytes"])
        if self.semantic is not None and self.embedder is not None:
            r = self.semantic.lookup(
                self.embedder.embed(prompt),
                _llm_key(llm_string),
                self.namespace,
                self.embedder.identity(),
                prompt,  # checked by the hit verifier
            )
            if r["above_threshold"]:
                self.stats["semantic_hits"] += 1
                return _load(r["output"])
        self.stats["misses"] += 1
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        if _has_tool_calls(return_val) or any(m in llm_string for m in _TOOL_MARKERS):
            self.stats["skipped_tool_calls"] += 1
            return
        data = _dump(return_val)
        self.memo.insert(self._key(prompt, llm_string), data, self.memo.version())
        if self.semantic is not None and self.embedder is not None:
            self.semantic.insert(
                self.embedder.embed(prompt),
                _llm_key(llm_string),
                data,
                self.namespace,
                self.embedder.identity(),
                prompt,
            )

    def clear(self, **kwargs: Any) -> None:
        self.memo.clear()
        if self.semantic is not None:
            self.semantic.clear()
