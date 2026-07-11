# Continuum

<p align="center">
  <a href="https://github.com/rithulkamesh/continuum/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/rithulkamesh/continuum/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://pypi.org/project/continuum-ai/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/continuum-ai.svg">
  </a>
  <a href="https://pypi.org/project/continuum-ai/">
    <img alt="Python >=3.10" src="https://img.shields.io/pypi/pyversions/continuum-ai.svg">
  </a>
  <a href="./LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://ct.rithul.dev/python/">
    <img alt="Python Docs" src="https://img.shields.io/badge/docs-python-blue">
  </a>
  <a href="https://ct.rithul.dev/cpp/">
    <img alt="C++ Docs" src="https://img.shields.io/badge/docs-c%2B%2B-informational">
  </a>
</p>
<p align="center">
  <a href="https://www.producthunt.com/products/continuum-4?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-continuum-5" target="_blank" rel="noopener noreferrer">
    <img alt="Continuum - A runtime that reuses computation across AI workflows | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1133105&amp;theme=light&amp;t=1777274020727">
  </a>
</p>

**The AI runtime that never computes the same thing twice — and never loses its place.**

Agent workflows burn money recomputing what they already know: the same system
prompt tokenized ten thousand times, the same subtask answered again, an
hour-long run lost to one crash at step 19. Continuum is a C++ execution
engine that treats LLM calls and tensor ops as operators in one dataflow
graph — redundant work is cached *at the runtime level*, and a running
workflow can be **checkpointed to bytes, resumed in another process, or forked
from any past step**.

```mermaid
flowchart LR
    P([prompt]) --> M{memo}
    M -->|exact hit · 0 ms| R([result])
    M --> S{semantic}
    S -->|paraphrase hit · 0 ms| R
    S --> T{trie prefix KV}
    T -->|"shared prefix · ~99% fewer tokens sent"| B
    T --> L{layer KV}
    L -->|warm decode state| B[backend call]
    B --> R
```

- **92.5% token reduction** on a mixed 20-step agent workload against live Azure OpenAI
- **Zero-cost exact repeats** — memoized calls skip the backend entirely
- **Durable execution** — checkpoint / crash / resume / time-travel fork, deterministic replay
- **One graph for tokens and tensors** — Azure, OpenAI, Anthropic, vLLM, libtorch, and MLX behind one IR

## What You Can Build With It

- **Agents that survive anything** — deploys, crashes, spot-instance eviction:
  checkpoint mid-run, resume on another machine with the KV cache still warm.
- **Cheaper agent fleets** — hundreds of sessions sharing one system prompt
  send it once; the trie prefix cache serves the rest.
- **Eval and CI loops** — re-running near-identical prompt suites hits the
  memo and prefix tiers instead of your API budget.
- **Time-travel debugging** — rewind a finished run, edit step 7, replay the
  alternate timeline; completed steps come from the checkpoint, never recomputed.
- **Hybrid pipelines** — hosted LLM calls and local tensor ops (libtorch, MLX)
  as operators in the same scheduled graph.

## How It Fits With What You Already Use

Continuum sits *below* your framework, not beside it — LangChain/LangGraph
code, raw SDK calls, or plain Python all route through the same runtime.

| You may already use | What it gives you | What Continuum adds |
|---|---|---|
| Provider prompt caching (OpenAI / Anthropic) | Prefix discounts inside one provider, TTL-bound | Provider-agnostic reuse, plus memo/semantic tiers and cache state you own |
| GPTCache / LangChain cache | App-layer response cache | Runtime-level reuse with defined invalidation semantics — tool calls never served stale |
| LangGraph checkpointing / Temporal | Workflow state persistence and retries | Checkpoints that carry the *KV cache* too — resume warm, deterministic replay, fork any past step |
| vLLM prefix caching | KV reuse on GPUs you operate | The same idea extended across hosted APIs, portable inside checkpoints |

## Quick Start

```bash
python -m pip install continuum-ai
```

Kill an agent mid-run and finish it in a different process:

```python
from continuum._native import DurableAgent

agent = DurableAgent()
agent.begin(["research the topic", "draft the report", "publish it"])
ckpt = agent.run_until_step(1)        # bytes: graph + every value + KV cache state

# ... process dies here ...

revived = DurableAgent()              # brand-new runtime
outputs = revived.resume_from(ckpt)   # completes steps 3+ without redoing 1-2
```

Rewind a finished run, edit one step, and replay the alternate timeline:

```python
forked = DurableAgent.fork(ckpt, node_id, "write a haiku instead")
alternate = DurableAgent().resume_from(forked)
```

See every reuse tier fire in one deterministic run:

```bash
PYTHONPATH=python python examples/05_continuum_reuse_stack.py   # --trace for per-tier firing
PYTHONPATH=python python examples/06_durable_agent.py           # checkpoint / crash / resume
PYTHONPATH=python python examples/07_time_travel_fork.py        # rewind, edit, replay
```

## Measured Results

Isolated per-tier benchmarks against a live Azure OpenAI backend (gpt-5-mini),
one reuse mechanism enabled at a time ([`benchmarks/v11/`](benchmarks/v11/)):

| Mechanism | Workload | Result |
|---|---|---|
| Trie prefix KV cache | 10 calls, 3,000-char shared prefix | **~99% token reduction** (9/9 hits, ~30 tokens sent per call) |
| Memo table | 5 exact-repeat tool calls | **5/5 backend calls skipped**, 0 ms |
| Mixed 20-step agent workflow | prefix + repeats + paraphrases + cold queries | **92.5% token reduction**, 4/20 backend calls eliminated |
| Cross-session cold start | persist cache metadata, restart, reload | **≥80% hit rate** on first warm run |
| No-reuse worst case | 4 unrelated queries | ~0.5% overhead-free passthrough, no errors |

Latency on prefix hits drops ~31% (5.4 s → 3.7 s median) — the API round-trip
dominates once 99% of prompt tokens are skipped; token cost is where reuse pays.
The bundled n-gram embedding provider is a placeholder: semantic-tier results
require a real embedding model and are excluded from the headline numbers.

<p align="center">
  <img alt="Continuum v1.1 benchmark dashboard" src="plots/v11/summary_dashboard.png" width="720">
</p>

Raw data, per-experiment reports, and the scripts that produced every number
live in [`benchmarks/v11/`](benchmarks/v11/); plots in [`plots/v11/`](plots/v11/).
Deterministic, CI-checked versions of every mechanism run offline via the
FakeLLM backend (`examples/05`–`07`, `tests/python/`).

## Why a Runtime, Not a Wrapper

Caching bolted onto an SDK can't know what is safe to reuse. Continuum sits
below the program, where reuse has defined semantics:

- **Correct invalidation** — tool calls are never served from cache
  (side-effecting), memoized results are version-bumped on resume, and cached
  KV state is reused only when its tokens are verifiably a prefix of the query.
- **Policy-gated** — every tier respects a per-session `ReusePolicy`
  (`always` / `never` / prefix-length threshold): one switch, no stale reads.
- **Portable state** — backends that can export their state handles carry the
  KV cache *inside the checkpoint*, so a resumed process starts warm, not cold.
- **Capability dispatch** — backends declare tensor/token/cache capabilities;
  the scheduler routes each node, converting tensors across backends explicitly.

```mermaid
flowchart TB
    subgraph app["your code"]
        A[LangChain / SDK calls / plain Python]
    end
    subgraph rt["Continuum runtime (C++)"]
        IR[dataflow IR] --> SCHED[capability-aware scheduler]
        SCHED --> REUSE[five-tier reuse stack]
        SCHED --> CKPT[(checkpoints<br/>graph + values + KV state)]
    end
    subgraph backends["backends"]
        B1[Azure / OpenAI / Anthropic]
        B2[vLLM]
        B3[libtorch / MLX]
    end
    A --> IR
    REUSE --> B1 & B2 & B3
```

## What Is Implemented

- C++ execution engine with IR interpreter and serializable checkpoints
- Five-tier reuse stack: trie prefix KV cache, memo table, semantic cache, layer KV warm-start, memory graph recall
- Durable execution: checkpoint a running workflow to bytes, resume in a fresh process (KV cache included), or fork from a past step with an edited value
- Session API with per-tier reuse policies and cross-session cache persistence
- Backends: Azure OpenAI, OpenAI, Anthropic, vLLM shim, libtorch, MLX, deterministic FakeLLM for CI

## Current Status

- v1 release hardening in progress
- CIR schema lock with serialization conformance (`schema/cir.fbs`)
- Linux and macOS CI matrix with coverage gates and fuzz workflow
- PyPI packaging under `continuum-ai` (import path remains `continuum`)

## Documentation

- Python API docs: <https://ct.rithul.dev/python/>
- C++ API docs: <https://ct.rithul.dev/cpp/>

Build docs locally:

```bash
# Python docs
python -m venv .venv-docs
. .venv-docs/bin/activate
pip install sphinx furo breathe
PYTHONPATH=python sphinx-build -b html docs/api/python docs/api/python/_build

# C++ docs
doxygen Doxyfile
```

Local outputs:

- `docs/api/python/_build/index.html`
- `docs/api/cpp/html/index.html`

## Community

- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Code of Conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Support guide: [`SUPPORT.md`](SUPPORT.md)
- Governance: [`GOVERNANCE.md`](GOVERNANCE.md)

Quick contributor setup:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
pytest
```

[![Star History Chart](https://api.star-history.com/chart?repos=rithulkamesh/continuum&type=date&legend=top-left)](https://www.star-history.com/?repos=rithulkamesh%2Fcontinuum&type=date&legend=top-left)

## Citation

If Continuum helps your work, cite it as:

```bibtex
@software{continuum2026,
  title        = {Continuum: Unified Runtime for Token and Tensor Programs},
  author       = {Kamesh, Rithul and Contributors},
  year         = {2026},
  url          = {https://github.com/rithulkamesh/continuum},
  version      = {1.0.0}
}
```
