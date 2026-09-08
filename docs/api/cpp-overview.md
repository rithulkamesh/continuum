# Continuum {#mainpage}

The AI runtime that never computes the same thing twice, and never loses its
place.

Continuum is a C++ execution engine that runs LLM calls and tensor ops as
operators in one dataflow graph. Work it has already done is reused at the
runtime level, and a running job can be checkpointed to bytes, resumed in
another process, or forked from any past step.

This manual documents the C++ headers under `include/continuum`. The Python
surface is documented separately at <https://ct.rithul.dev/python/>.

## The shape of the system

A program is a **graph** of typed nodes. An **interpreter** executes the graph
in scheduler order, dispatching each node to a **backend**. Every token
generation first passes through the **reuse stack**, which answers from one of
five caches when it can instead of calling the backend. Serializing the graph
together with every value and the portable KV state produces a **checkpoint**.

```
program  ->  Graph (CIR)  ->  Interpreter  ->  reuse stack  ->  Backend
                                    |
                                    +-- serialize --> checkpoint (bytes)
```

## Namespaces

| Namespace            | Contents |
|----------------------|----------|
| `continuum::ir`      | `Graph`, `Node`, `NodeKind`, type lattice, payloads, CIR serialization. |
| `continuum::runtime` | `Interpreter`, `Session`, the reuse-tier indexes, checkpoint serialization, `DurableAgent`. |
| `continuum::backend` | `Backend` contract, `BackendRegistry`, and the concrete backends (Azure, OpenAI, Anthropic, vLLM, libtorch, MLX, FakeLLM). |

## Key types

- `ir::Graph` : the dataflow program. `serialize()` emits the CIR binary
  envelope; `deserialize()` reads it back.
- `ir::Node` / `ir::NodeKind` : one step. `TensorOp`, `TokenOp`, `PromptOp`,
  `ToolOp`, `ControlOp`. `ToolOp` is never served from cache.
- `runtime::Interpreter` : executes nodes against materialized dependencies.
  For a `TokenOp` it canonicalizes tokens, queries the reuse stack for the
  longest reusable prefix, and calls the backend with the reusable state.
- `runtime::Session` : a reuse-aware context that holds the cache index and a
  `ReusePolicy` (`always`, `never`, or a prefix-length threshold) across many
  `run` calls.
- `runtime::DurableAgent` : a step-sequenced run. `run_until_step` returns a
  checkpoint; `resume_from` continues it in a fresh instance; `fork` branches
  from a past node with one value replaced.
- `backend::Backend` : implements `run_with_cache(...)`, returning the output,
  an updated `BackendState` handle for future prefix reuse, and reuse metrics
  (`reused_prefix_len`, `compute_steps`, `tokens_sent`, `tokens_saved`).

## The reuse stack

Every `TokenOp` checks these tiers in cost order and stops at the first hit:

1. **Memo** : the exact same call, seen before. O(1), no tokens.
2. **Semantic** : same intent, different wording, matched by embedding.
3. **Prefix KV** : a shared prompt prefix is already tokenized; send the suffix.
4. **Layer KV** : warm attention state carried forward; no prefill.
5. **Memory graph** : relevant context from a prior run.

A cache key is built from backend and model identity, decode parameters
(`op_name`, `temperature`, `max_tokens`), and the canonicalized token sequence.
A prefix hit reuses tokens *and* a backend state handle, and is valid only when
both were derived from the same canonical prefix length.

## The ownership boundary

The cache is runtime-owned, but a reuse decision depends on backend state
handles. That coupling is deliberate: the runtime decides *when* a prefix is
reusable, the backend decides *how* that state is represented.

## Design notes

The pages below go deeper on each subsystem:

- Architecture overview
- Runtime execution model
- KV cache semantics
- CIR: the canonical IR
- The backend ABI boundary
