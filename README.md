# Continuum

Continuum is a unified execution runtime for LLM and ML programs.  
It is not an API wrapper and not an orchestration-only tool.  
The core system executes a shared intermediate representation (IR) that spans token generation and tensor computation in one runtime.

## Core Idea

Continuum uses one IR to represent both token and tensor operations, then executes that graph through a single interpreter.  
KV caching is treated as a program-level concern rather than a backend-specific add-on.  
Backends receive reusable state handles through a common contract, so cache-aware execution can remain backend-agnostic.  
This allows the same execution model to drive cloud LLM calls, local LLM backends, and tensor workloads.

## What Works Today

- C++ execution engine with IR + interpreter
- KV cache index with canonical prefix normalization
- Azure backend (real network execution)
- libtorch backend (tensor/training execution)

## Example

See `examples/01_research_agent.py` for a paired benchmark workflow that exercises cache-aware token generation across backends.

## Benchmarking Approach

Benchmarks are run as paired trials (uncached vs cached on identical input), with warmup discarded and robust statistics reported (median/p50/p95).  
Primary signal is token reduction (`tokens_saved / (tokens_sent + tokens_saved)`), with latency ratio tracked as secondary due to provider/network noise.

## Status

- Research prototype
- vLLM integration in progress
- M1 partially validated: Azure token reduction is measurable; latency results remain noisy under network/provider variance

[![Star History Chart](https://api.star-history.com/chart?repos=rithulkamesh/continuum&type=date&legend=top-left)](https://www.star-history.com/?repos=rithulkamesh%2Fcontinuum&type=date&legend=top-left)


## Install

```bash
python -m pip install continuum
```

## Reproducible Example Validation

```bash
PYTHONPATH=python python scripts/benchmarks/run_examples.py | python scripts/benchmarks/validate_outputs.py
```
