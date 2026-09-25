# E8 — Memory-Graph Recall, Isolated

**Date**: 2026-09-25
**Backend**: none. The tier is driven directly through `MemoryGraphStore`, so
no other reuse tier and no network time can affect the numbers.
**Embedder**: `continuum/char-ngram-v1:64` (the bundled `BruteForceEmbeddingProvider`)
**Host**: 4 vCPU Intel Xeon @ 2.80 GHz, Linux, CPython 3.12
**Runner**: [`benchmarks/scripts/e8_memory_graph_recall.py`](../scripts/e8_memory_graph_recall.py)
**Raw data**: [`benchmarks/data/e8_memory_graph_recall.json`](../data/e8_memory_graph_recall.json)

## What this tier does

The memory graph logs every `TokenOp` prompt as a node and, before each new
generation, retrieves the most similar earlier prompts in the same namespace
(top 5, cosine >= 0.7). Recall is **observational today**: the related nodes
are logged, but they are not injected into the request and no tokens are
saved. So this benchmark measures recall quality and cost, not token savings.
See `docs/design/cache.md` ("Memory-graph recall tier").

## 1. Recall quality

12 earlier turns across 4 topics (billing, password, shipping, refund), then
8 follow-up queries whose relevant turns are known (3 per topic). `top_k = 3`.

| Threshold | Queries with recall | Precision@3 | Recall@3 |
|---|---|---|---|
| 0.5 | 8/8 | 0.583 | 0.583 |
| 0.6 | 8/8 | 0.583 | 0.583 |
| 0.7 (runtime default) | 8/8 | 0.583 | 0.583 |
| 0.8 | 8/8 | 0.583 | 0.583 |
| 0.9 | 8/8 | 0.833 | 0.417 |

Threshold-free separation: the top-1 recalled turn was on-topic for **8/8**
queries, but the mean best similarity was 0.944 for relevant turns and 0.890
for irrelevant ones. The char-n-gram embedder rates almost any two English
sentences above 0.8, so the runtime's 0.7 cutoff filters nothing, and top-3
recall is barely better than half on-topic. Recall becomes precise only near
0.9, at the cost of recall.

**Takeaway**: ranking works (top-1 is right); the threshold does not. With the
bundled embedder, treat recall as "nearest few turns", not "relevant turns".
A semantic embedder (`continuum.embeddings`) is needed before recall could
safely feed context into a request.

## 2. Lookup cost

`retrieve_similar` is a linear scan over every node, followed by a sort of
the survivors. 200 lookups per size, 64-dim vectors:

| Nodes | p50 | p95 |
|---|---|---|
| 128 | 45 µs | 100 µs |
| 1,024 | 353 µs | 498 µs |
| 8,192 (default capacity) | 3.4 ms | 4.1 ms |

Cost is linear in store size, as expected. At full default capacity a recall
adds ~3.4 ms per generation, small next to a remote LLM call but not free for
a local model. Lower `max_nodes`, or an ANN index, if that matters.

## 3. Footprint

`estimated_bytes()` reports ~414 bytes per node at 64 dimensions (content
string, 256 bytes of embedding, ids, and struct overhead). A full default
store (8,192 nodes) is ~3.4 MB. Eviction is FIFO by insertion; see
"Eviction and memory bounds" in `docs/design/cache.md`.

## Reproduce

```bash
PYTHONPATH=python python benchmarks/scripts/e8_memory_graph_recall.py
```

Quality numbers are deterministic. Latency varies with the host.
