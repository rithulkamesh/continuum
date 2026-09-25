# KV Cache Semantics

Continuum cache keys are built from:

- backend/model identity
- decode parameters (`op_name`, `temperature`, `max_tokens`)
- canonicalized input token sequence
- optional **cache namespace** (tenant / deployment isolation)

## Namespace isolation

`Session::cache_namespace` (default empty) is threaded into every reuse tier —
memo, semantic, prefix KV, layer KV, and memory-graph recall. Identical
prompts under different namespaces never hit each other. Empty namespace
preserves the historical single-tenant key space.

Set it from Python with `session.cache_namespace = "tenant-a"`.

## Prefix Normalization

Interpreter canonicalizes textual/token inputs before cache lookup. Prefix length is computed against this canonical sequence so cache matching stays stable across formatting differences.

## State Reuse vs Token Reuse

- Token reuse: runtime detects longest matching token prefix.
- State reuse: runtime passes a backend-owned `BackendState` handle from the matched entry.

Correctness requires both to align. If backend state was derived from a different canonical prefix length, reuse can become semantically invalid.

## Azure vs vLLM

- Azure path currently approximates prefix savings by sending suffix-only requests on cache hit and tracking `tokens_sent`/`tokens_saved`.
- vLLM path (also Ollama and any `/v1/completions` server when `VLLM_BASE_URL`
  is set) always sends the full prompt. The server's own prefix cache (vLLM
  automatic prefix caching) skips recomputing the shared prefix, and its
  `cached_tokens` is reported as `tokens_saved`. The backend state handle
  records which prefix the server holds warm. It is portable, so it survives a
  checkpoint / resume, and `VLLM_REWARM_ON_IMPORT=1` re-warms the server after
  a restart. See `benchmarks/reports/vllm-prefix-reuse.md`.

Both paths emit the same runtime metrics so benchmark comparisons stay backend-agnostic.

## Semantic tier

The semantic tier (`SemanticCacheIndex`) returns a cached output when a new
prompt is *close enough* to an earlier one, catching paraphrases the exact
memo tier misses.

**Key.** An entry matches a lookup only when all of these are equal: model id,
cache namespace, and **embedder identity**. Among matching entries the highest
cosine similarity wins, and it is served only if it is at least
`similarity_threshold` (default `0.85`).

**Where vectors come from.** A session embeds the concatenated string inputs
of each `TokenOp` with its `EmbeddingProvider` (`Session.set_embedding_provider`).
The provider is an interface, `embed(text)`, `dimension()`, `identity()`,
implementable in C++ or by subclassing `continuum._native.EmbeddingProvider` in
Python. `continuum.embeddings` ships three:

| Provider | Use |
|----------|-----|
| `CallableEmbeddingProvider(fn, dimension, identity)` | any local model, e.g. `SentenceTransformer(...).encode` |
| `OpenAICompatibleEmbeddingProvider(base_url, model)` | a hosted `/v1/embeddings` endpoint (OpenAI, vLLM, Ollama) |
| `PrecomputedEmbeddingProvider(vectors, identity)` | vectors computed ahead of time, for reproducible runs and evals |

The built-in `BruteForceEmbeddingProvider(dim)` (identity
`continuum/char-ngram-v1:<dim>`) hashes character 1–3-grams. It is
deterministic and dependency-free, but lexical: it scores shared *spelling*,
not shared meaning. Use a real model for production paraphrase matching.

```python
from continuum.embeddings import CallableEmbeddingProvider
from sentence_transformers import SentenceTransformer

st = SentenceTransformer("all-MiniLM-L6-v2")
session.set_embedding_provider(CallableEmbeddingProvider(
    lambda text: st.encode(text, normalize_embeddings=True).tolist(),
    dimension=384,
    identity="sentence-transformers/all-MiniLM-L6-v2",
))
```

**Why identity is in the key.** Cosine similarity between vectors from two
different embedders is meaningless, and two embedders can even share a
dimension. Storing `identity()` with each entry means changing the embedder
(or its version) starts a fresh key space instead of producing silent false
hits. Change the identity string whenever the vectors would change.

**Reproducibility.** A run is reproducible when the embedder is deterministic
and its identity pins the exact model and preprocessing. For evaluations, embed
the dataset once and replay it through `PrecomputedEmbeddingProvider`.

**Threshold.** How often a near-miss is served wrongly depends on the embedder
and the threshold; `benchmarks/reports/semantic-false-hits.md` measures that
trade-off.

## Memory-graph recall tier

`MemoryGraphStore` is a log of earlier prompts that the runtime searches for
related context before each generation.

**What it stores.** After a `TokenOp` runs on the backend, the interpreter adds
one `MemoryNode` holding the concatenated string inputs (`content`), their
embedding from the session's `EmbeddingProvider`, the node type (`Prompt`),
and the session's cache namespace. Nodes get monotonically increasing ids.
Nothing is stored when the step was served by the memo or semantic tier, when
the step has no string inputs, or when no embedder is attached.

**How recall triggers.** On every `TokenOp` with both a memory graph and an
embedder attached, before calling the backend, the interpreter embeds the
prompt and calls `retrieve_similar(query, max_results=5, min_similarity=0.7,
namespace)`: a linear scan returning up to five nodes from the same namespace
with cosine similarity >= 0.7, best first.

**What recall does with the result.** Today, it logs it
(`memory_recall ... related=N top_sim=...`). Recalled nodes are *not* added to
the request, so the tier saves no tokens yet; it is an observable signal and
the hook where context injection would go. Isolated measurements of recall
quality and cost are in `benchmarks/reports/memory-graph-recall.md`: with the
bundled n-gram embedder the top-1 hit is on topic, but the 0.7 cutoff filters
almost nothing, so a semantic embedder is a prerequisite for injecting
recalled context.

**Invalidation.**

- Namespace: recall never crosses cache namespaces.
- Capacity: FIFO eviction by insertion (see below). Reads never refresh a node.
- `clear()` drops every node and resets ids.
- There is no model- or version-based invalidation: nodes are prompts, not
  model outputs, so they stay valid when the model changes. Changing the
  embedder *does* matter: vectors from different embedders are not
  comparable, so clear the store when you switch embedders.

## Eviction and memory bounds

Every tier is bounded. Capacities are set at construction; a capacity of `0`
stores nothing. Each tier reports `size()`, its capacity, and an
`estimated_bytes()` figure, and `Session.cache_stats()` returns all three for
every tier attached to a session:

```python
session.cache_stats()
# {"prefix_kv": {"entries": 42, "capacity": 8192, "bytes": 13440},
#  "memo": {...}, "semantic": {...}, "layer_kv": {...}, "memory_graph": {...}}
```

| Tier | Class | Bound | Policy | What refreshes recency |
|------|-------|-------|--------|------------------------|
| Prefix KV | `KVCacheIndex` | `max_entries` (per-depth trie entries) | LRU | `insert`, `longest_prefix` hit |
| Memo | `MemoTable` | `max_entries` | LRU; stale versions dropped on lookup | `insert`, `lookup` hit |
| Semantic | `SemanticCacheIndex` | `max_entries` | LRU | `insert`, above-threshold `lookup` |
| Layer KV | `LayerKVCacheIndex` | `max_entries` **and** `max_bytes` | LRU until both bounds hold | `insert`, `find_deepest` hit |
| Memory graph | `MemoryGraphStore` | `max_nodes` | FIFO by insertion | never (recall is a scan) |
| Prefetch | `FutureCache` | `max_entries`, `ttl` | expired first, then FIFO | never |

Byte accounting is approximate: it counts owned payloads (keys, cached
outputs, embeddings, content strings) plus the fixed size of each entry
struct. It does **not** count backend-owned state behind a
`BackendState` handle (for example a real KV tensor); the layer tier instead
uses the caller-supplied `LayerCheckpoint::estimated_bytes` and enforces
`max_bytes` against that sum.

The memory graph is FIFO rather than LRU on purpose: it is an append-only
conversation log, and recall reads every node, so "recently read" carries no
signal. Oldest-first keeps recent turns.

The contract is pinned by `tests/python/test_cache_eviction.py`, which fills
each tier past capacity and checks which entries survive.

## On-disk format versions

| Blob | Magic | Versions | Notes |
|------|-------|----------|-------|
| Checkpoint | `CPT1` | 1–3 | v1 = no KV snapshot; v2 = snapshot without namespace; v3 = per-entry `cache_namespace`. Current writers emit v3. |
| KV metadata | `CPKV` | 1–2 | v1 = no namespace; v2 = per-entry namespace. Current writers emit v2. |

Unknown magic or version is refused with a clear error. Known older versions
still load (empty namespace). `migrate_checkpoint(bytes)` re-serializes a
readable blob at the current wire format.
