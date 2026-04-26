# KV Cache Semantics

Continuum cache keys are built from:

- backend/model identity
- decode parameters (`op_name`, `temperature`, `max_tokens`)
- canonicalized input token sequence

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
