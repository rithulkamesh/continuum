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

## Observability

Attach a `ReuseObserver` (`Session::set_observer`, `DurableAgent.set_observer`)
and the interpreter emits a `ReuseEvent` for:

- every **tier lookup** on a `TokenOp`: `memo`, `semantic`, `prefix_kv`,
  `layer_kv`, `memory_graph`, with `hit`, `tokens_saved`, `match_len`,
  `similarity`, and start / end timestamps;
- every **executed node**: kind, backend, model, `served_by` (`memo`,
  `semantic`, `backend`, `passthrough`), and the backend's reuse metrics
  (`tokens_sent`, `tokens_saved`, `reused_prefix_len`, `used_cached_state`,
  `compute_steps`).

A node's tier events arrive before its node event. With no observer attached
(the default) nothing is emitted, so there is no cost.

`continuum.telemetry.OpenTelemetryObserver` exports the events to
OpenTelemetry: a `continuum.node` span per node with one
`continuum.reuse.<tier>` child span per lookup, plus these metrics:

| Metric | Type | Attributes |
|--------|------|------------|
| `continuum.reuse.lookups` | counter | `continuum.reuse.tier`, `continuum.model_id` |
| `continuum.reuse.hits` | counter | same |
| `continuum.reuse.tokens_saved` | counter | same |
| `continuum.reuse.lookup.duration` | histogram (ms) | same |
| `continuum.node.executions` | counter | `continuum.node.kind`, `continuum.served_by` |

Enable it per session with `continuum.telemetry.instrument(session)`, or set
`CONTINUUM_OTEL=1` to have every `DurableAgent` instrument itself. It uses the
global OpenTelemetry providers unless given others; install with
`pip install "continuum-ai[otel]"`.

## Checkpointing

Checkpointing serializes the graph plus runtime state (every computed value and
portable KV state) so a run can resume in a fresh process or fork from a past
step. See the durable-agent examples under `examples/`.
