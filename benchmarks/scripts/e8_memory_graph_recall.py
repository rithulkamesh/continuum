"""E8: memory-graph recall tier in isolation (issue #12).

Offline and deterministic: no backend is called. The tier is exercised
directly through ``MemoryGraphStore`` so nothing else in the reuse stack can
influence the numbers. Three measurements:

1. Recall quality. A labeled log of earlier turns across several topics, then
   follow-up queries whose relevant turns are known. Reports precision@k and
   recall@k per similarity threshold for the bundled n-gram embedder.
2. Lookup cost. ``retrieve_similar`` is a linear scan; latency is measured at
   store sizes up to the default capacity (8192 nodes).
3. Footprint. ``estimated_bytes()`` per node.

Run:

    PYTHONPATH=python python benchmarks/scripts/e8_memory_graph_recall.py

Writes benchmarks/data/e8_memory_graph_recall.json.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from continuum._native import BruteForceEmbeddingProvider, MemoryGraphStore

DATA = Path(__file__).resolve().parents[1] / "data" / "e8_memory_graph_recall.json"
EMBED_DIM = 64
TOP_K = 3
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
SIZES = [128, 1024, 8192]
LATENCY_QUERIES = 200

# Earlier conversation turns, grouped by topic.
HISTORY = {
    "billing": [
        "why was my credit card charged twice this month",
        "how do I update the credit card on my billing account",
        "can I get an invoice for last month's billing charge",
    ],
    "password": [
        "how do I reset my account password",
        "the password reset email never arrived",
        "is there a rule for password length and symbols",
    ],
    "shipping": [
        "where is my package, the shipping tracking number is not working",
        "can I change the shipping address after the order shipped",
        "how long does international shipping take",
    ],
    "refund": [
        "what is the refund policy for opened items",
        "how long does a refund take to reach my bank",
        "can I get a refund instead of store credit",
    ],
}

# Follow-ups and the topic whose turns are relevant to them.
QUERIES = [
    ("I was charged twice on my credit card", "billing"),
    ("send me the invoice for my billing account", "billing"),
    ("I still have not received the password reset email", "password"),
    ("what are the password requirements", "password"),
    ("my shipping tracking number shows nothing", "shipping"),
    ("update the shipping address on my order", "shipping"),
    ("how long until my refund reaches the bank", "refund"),
    ("is the refund policy different for opened items", "refund"),
]


def _labeled_store() -> tuple[BruteForceEmbeddingProvider, MemoryGraphStore, dict[int, str]]:
    embedder = BruteForceEmbeddingProvider(EMBED_DIM)
    store = MemoryGraphStore(1024)
    topic_of: dict[int, str] = {}
    for topic, turns in HISTORY.items():
        for turn in turns:
            topic_of[store.add_node(turn, embedder.embed(turn))] = topic
    return embedder, store, topic_of


def separation() -> dict:
    """Best relevant vs best irrelevant similarity per query (threshold-free)."""
    embedder, store, topic_of = _labeled_store()
    total = sum(len(v) for v in HISTORY.values())
    relevant, irrelevant, wins = [], [], 0
    for text, topic in QUERIES:
        hits = store.retrieve_similar(embedder.embed(text), total, -1.0)
        best_rel = max(h["similarity"] for h in hits if topic_of[h["id"]] == topic)
        best_irr = max(h["similarity"] for h in hits if topic_of[h["id"]] != topic)
        relevant.append(best_rel)
        irrelevant.append(best_irr)
        wins += best_rel > best_irr
    return {
        "mean_best_relevant_similarity": round(statistics.fmean(relevant), 3),
        "mean_best_irrelevant_similarity": round(statistics.fmean(irrelevant), 3),
        "queries_where_top1_is_relevant": wins,
        "queries": len(QUERIES),
    }


def quality() -> list[dict]:
    embedder, store, topic_of = _labeled_store()
    rows = []
    for threshold in THRESHOLDS:
        precisions, recalls, fired = [], [], 0
        for text, topic in QUERIES:
            hits = store.retrieve_similar(embedder.embed(text), TOP_K, threshold)
            relevant_total = len(HISTORY[topic])
            relevant_hit = sum(1 for h in hits if topic_of[h["id"]] == topic)
            if hits:
                fired += 1
                precisions.append(relevant_hit / len(hits))
            recalls.append(relevant_hit / min(TOP_K, relevant_total))
        rows.append(
            {
                "threshold": threshold,
                "fired": fired,
                "queries": len(QUERIES),
                "precision_at_k": round(statistics.fmean(precisions), 3) if precisions else None,
                "recall_at_k": round(statistics.fmean(recalls), 3),
            }
        )
    return rows


def cost() -> list[dict]:
    embedder = BruteForceEmbeddingProvider(EMBED_DIM)
    rows = []
    for size in SIZES:
        store = MemoryGraphStore(size)
        for i in range(size):
            text = f"turn {i}: user asked about item {i % 97} and order {i % 13}"
            store.add_node(text, embedder.embed(text))
        query = embedder.embed("user asked about item 42 and order 3")
        samples = []
        for _ in range(LATENCY_QUERIES):
            t0 = time.perf_counter()
            store.retrieve_similar(query, TOP_K, 0.7)
            samples.append((time.perf_counter() - t0) * 1e6)
        samples.sort()
        rows.append(
            {
                "nodes": store.size(),
                "p50_us": round(statistics.median(samples), 1),
                "p95_us": round(samples[int(0.95 * (len(samples) - 1))], 1),
                "bytes": store.estimated_bytes(),
                "bytes_per_node": round(store.estimated_bytes() / store.size(), 1),
            }
        )
    return rows


def main() -> None:
    result = {
        "experiment": "E8: memory-graph recall in isolation",
        "embedder": BruteForceEmbeddingProvider(EMBED_DIM).identity(),
        "top_k": TOP_K,
        "quality": quality(),
        "separation": separation(),
        "cost": cost(),
    }
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(result, indent=2) + "\n")

    print("E8: memory-graph recall (isolated)")
    print(f"embedder={result['embedder']} top_k={TOP_K}")
    print("threshold  fired  precision@k  recall@k")
    for r in result["quality"]:
        p = "-" if r["precision_at_k"] is None else f"{r['precision_at_k']:.3f}"
        print(
            f"{r['threshold']:>9}  {r['fired']:>2}/{r['queries']}  {p:>11}  {r['recall_at_k']:>8.3f}"
        )
    sep = result["separation"]
    print(
        f"top-1 relevant in {sep['queries_where_top1_is_relevant']}/{sep['queries']} queries; "
        f"mean best sim relevant={sep['mean_best_relevant_similarity']} "
        f"irrelevant={sep['mean_best_irrelevant_similarity']}"
    )
    print("nodes   p50_us   p95_us   bytes/node")
    for r in result["cost"]:
        print(f"{r['nodes']:>5}  {r['p50_us']:>7}  {r['p95_us']:>7}  {r['bytes_per_node']:>10}")
    print(f"wrote {DATA.relative_to(Path.cwd()) if DATA.is_relative_to(Path.cwd()) else DATA}")


if __name__ == "__main__":
    main()
