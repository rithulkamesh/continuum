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
- vLLM path is designed for real KV reuse semantics: the same runtime prefix hit mechanism forwards backend state, and vLLM can avoid recomputing the shared prefix work.

Both paths emit the same runtime metrics so benchmark comparisons stay backend-agnostic.

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
