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

Continuum is a unified execution runtime for LLM and ML programs.
It is not just an API wrapper and not just orchestration glue.
Continuum executes a shared intermediate representation (IR) that spans token generation and tensor computation inside one runtime.

## Why Continuum

- **One IR, two worlds**: token ops and tensor ops in a single executable graph.
- **Backend-agnostic caching**: reusable backend state handles enable cross-call prefix reuse without backend-specific app code.
- **Capability-driven dispatch**: runtime routes ops by declared backend capability, not brittle string checks.
- **Explicit interoperability**: cross-backend tensors are tagged and converted explicitly, never silently mixed.
- **Native-ready architecture**: C++ core with ABI boundary prep for future dynamic backend loading.

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
- MLX backend (native tensor op path for Apple workflows)

## Example

See `examples/01_research_agent.py` for a paired benchmark workflow that exercises cache-aware token generation across backends.

## Benchmarking Approach

Benchmarks are run as paired trials (uncached vs cached on identical input), with warmup discarded and robust statistics reported (median/p50/p95).
Primary signal is token reduction (`tokens_saved / (tokens_sent + tokens_saved)`), with latency ratio tracked as secondary due to provider/network noise.

## Status

- v1 release hardening in progress
- Capability-driven backend dispatch implemented (tensor/token/cache)
- MLX + libtorch tensor interoperability implemented with explicit conversion rules
- CIR schema lock added (`schema/cir.fbs`) with serialization conformance tests
- Python + C++ API docs pipelines wired (Sphinx + Doxygen + GitHub Pages workflow)
- Packaging migrated to `continuum-ai` (import path `continuum`) with PyPI publish workflow
- CI matrix active on Linux + macOS with coverage gates and fuzz workflow

[![Star History Chart](https://api.star-history.com/chart?repos=rithulkamesh/continuum&type=date&legend=top-left)](https://www.star-history.com/?repos=rithulkamesh%2Fcontinuum&type=date&legend=top-left)


## Install

```bash
python -m pip install continuum-ai
```

Import remains:

```python
import continuum
```

## Reproducible Example Validation

```bash
PYTHONPATH=python python scripts/benchmarks/run_examples.py | python scripts/benchmarks/validate_outputs.py
```

## Pre-commit Hooks

Set up local quality gates (`ruff`, formatting, YAML/whitespace checks):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Note: generated docs/build outputs are excluded by default in `.pre-commit-config.yaml`.

## API Docs

Build Python docs locally:

```bash
python -m venv .venv-docs
. .venv-docs/bin/activate
pip install sphinx furo breathe
PYTHONPATH=python sphinx-build -b html docs/api/python docs/api/python/_build
```

Then open:

- `docs/api/python/_build/index.html`
- GitHub Pages: `https://rithulkamesh.github.io/continuum/python/`

Build C++ docs locally:

```bash
doxygen Doxyfile
```

Then open:

- `docs/api/cpp/html/index.html`
- GitHub Pages: `https://rithulkamesh.github.io/continuum/cpp/`

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
