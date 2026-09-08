"""04 - Support-bot deflection with the semantic cache tier.

A real support queue is full of the *same question asked five different ways*:
"how do I reset my password", "forgot my password, need to reset it", "reset my
forgotten password please". An exact-match cache never fires on those. The
semantic tier embeds each incoming ticket and serves an answer from cache when a
past ticket is close enough in meaning (cosine >= threshold).

This drives ``SemanticCacheIndex`` + ``BruteForceEmbeddingProvider`` directly,
the way a gateway in front of your LLM would. The bundled embedder is lexical
(good enough to show the mechanism); a real deployment swaps in a sentence
embedding model without touching the cache logic.

    PYTHONPATH=python python examples/04_semantic_support_cache.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import (  # noqa: E402
    BruteForceEmbeddingProvider,
    SemanticCacheIndex,
)

BAR = "=" * 64
MODEL = "support-bot/v1"
THRESHOLD = 0.80

# A day's worth of tickets. Clusters are the same intent, reworded; the last
# two are genuinely new and must miss.
QUEUE = [
    "How do I reset my password?",
    "How do I request a refund for an order?",
    "I forgot my password, how do I reset it?",
    "Need to reset my forgotten password please",
    "Can I get a refund on my recent order?",
    "I want to request a refund for an order I placed",
    "Where are your data centers located?",
    "how to reset my password",
    "What regions are your data centers located in?",
    "How do I change my billing address?",
]


def canned_answer(ticket: str) -> bytes:
    """Stand-in for the expensive LLM call we are trying to avoid."""
    return f"[generated answer for: {ticket!r}]".encode()


def main() -> None:
    print(BAR)
    print(" Continuum - Semantic Cache (support-bot deflection)")
    print(BAR)
    print(f"threshold cosine >= {THRESHOLD:.2f}   embedder = BruteForce (lexical)")
    print("-" * 64)

    embedder = BruteForceEmbeddingProvider(dim=64)
    cache = SemanticCacheIndex(max_entries=2048, similarity_threshold=THRESHOLD)

    llm_calls = 0
    deflected = 0
    for i, ticket in enumerate(QUEUE, 1):
        emb = embedder.embed(ticket)
        hit = cache.lookup(emb, MODEL)
        if hit["above_threshold"]:
            deflected += 1
            print(f"{i:2d}. HIT  sim={hit['similarity']:.3f}  served from cache   | {ticket}")
        else:
            llm_calls += 1
            cache.insert(emb, MODEL, canned_answer(ticket))
            print(f"{i:2d}. MISS sim={hit['similarity']:.3f}  called the model    | {ticket}")

    print("-" * 64)
    total = len(QUEUE)
    print(f"tickets={total}  model_calls={llm_calls}  deflected={deflected}")
    print(f"cache entries armed: {cache.size()}")
    if total:
        print(f"deflection rate: {deflected / total * 100:.0f}% of the queue answered for free")
    print(BAR)
    print(" semantic support cache: OK")
    print(BAR)


if __name__ == "__main__":
    main()
