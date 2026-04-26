# Runtime Execution Model

`runtime::Interpreter` executes nodes in topological order. Input node values are injected first; all other values are computed by stepping each node against already-materialized dependencies.

## Node Execution Paths

- `TensorOp`: direct backend execution (typically `libtorch`).
- `TokenOp`: canonicalize input tokens, query cache for longest reusable prefix, then invoke backend with optional prefix state.
- `PromptOp`: pass-through or empty-string fallback.
- `ToolOp`: backend-dispatched operation.

## Backend Contract

Backends implement `run_with_cache(...)` and return:

- output value
- resulting backend state handle (for future prefix reuse)
- reuse/compute metrics (`reused_prefix_len`, `compute_steps`, `tokens_sent`, `tokens_saved`)
- whether cached state was consumed

Interpreter responsibility is orchestration and cache index maintenance. Backend responsibility is remote/local model execution and backend-specific state encoding.
