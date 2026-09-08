# Runtime Execution Model

`runtime::Interpreter` executes nodes in scheduler order. The scheduler defaults
to topological order: input node values are injected first, then every other
value is computed by stepping each node against its already-materialized
dependencies.

## Node execution paths

- `TensorOp`: direct backend execution, typically `libtorch`.
- `TokenOp`: canonicalize input tokens, query the reuse stack for the longest
  reusable prefix, then invoke the backend with optional prefix state.
- `PromptOp`: pass-through, with an empty-string fallback.
- `ToolOp`: backend-dispatched operation. Never served from cache.

## Backend contract

Backends implement `run_with_cache(...)` and return:

- the output value
- the resulting backend state handle, for future prefix reuse
- reuse and compute metrics (`reused_prefix_len`, `compute_steps`,
  `tokens_sent`, `tokens_saved`)
- whether cached state was consumed

The interpreter is responsible for orchestration and cache-index maintenance.
The backend is responsible for remote or local model execution and for
backend-specific state encoding.

## Checkpointing

Checkpointing serializes the graph plus runtime state (every computed value and
portable KV state) so a run can resume in a fresh process or fork from a past
step. See the durable-agent examples under `examples/`.
