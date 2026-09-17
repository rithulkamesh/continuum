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

## On-disk format versions

| Blob | Magic | Versions | Notes |
|------|-------|----------|-------|
| Checkpoint | `CPT1` | 1–3 | v1 = no KV snapshot; v2 = snapshot without namespace; v3 = per-entry `cache_namespace`. Current writers emit v3. |
| KV metadata | `CPKV` | 1–2 | v1 = no namespace; v2 = per-entry namespace. Current writers emit v2. |

Unknown magic or version is refused with a clear error. Known older versions
still load (empty namespace). `migrate_checkpoint(bytes)` re-serializes a
readable blob at the current wire format.
