Backends and the IR
===================

One graph for tokens and tensors
--------------------------------

LLM calls and tensor ops are both operators in the same IR, scheduled by the
same cache-aware interpreter. A pipeline that calls a hosted model, runs a
local reranker in libtorch, and calls the model again is one graph, not three
systems stitched together.

Capability dispatch
-------------------

Each backend declares which capabilities it provides:

- **tensor**: runs ``TensorOp`` nodes,
- **token**: runs ``TokenOp`` nodes,
- **cache**: can export and re-import KV state for prefix reuse.

The scheduler routes each node to a backend that can run it. When a graph spans
two backends (a hosted token step feeding a local tensor step, say) the
interpreter converts values across the boundary explicitly rather than
guessing.

Available backends
------------------

=================  ==========================  ================================================
Backend            Kind                        Notes
=================  ==========================  ================================================
Azure OpenAI       hosted token                Prefix savings approximated by suffix-only requests.
OpenAI             hosted token                Same path as Azure.
Anthropic          hosted token                Same path as Azure.
vLLM               self-hosted token           Real KV reuse: forwarded state, prefix not recomputed.
libtorch           in-process tensor           CPU or GPU tensor execution.
MLX                in-process tensor           Apple silicon.
FakeLLM            deterministic token         For tests and examples. No network, reproducible.
=================  ==========================  ================================================

``FakeLLM`` is what the examples and the Python test suite use, which is why
every number they print is reproducible and CI-checkable.

Where Continuum sits
--------------------

Continuum runs below your framework, not beside it. LangChain or LangGraph
code, raw SDK calls, and plain Python all route through the same runtime.
Provider prompt caching gives you prefix discounts inside one provider;
Continuum adds provider-agnostic reuse, the memo and semantic tiers, and cache
state you own inside a checkpoint. LangGraph or Temporal checkpointing persists
workflow state; a Continuum checkpoint also carries the KV cache, so a resume
starts warm.

See `docs/comparison.md
<https://github.com/rithulkamesh/continuum/blob/master/docs/comparison.md>`_
for the full comparison.
