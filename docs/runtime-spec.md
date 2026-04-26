# Runtime Spec v0.1

- Eager interpreter executes nodes in scheduler order.
- Scheduler defaults to topological order in v0.1.
- KV cache indexes `(model_id, token_prefix)` for reuse.
- Checkpointing serializes graph/runtime state for resume.
