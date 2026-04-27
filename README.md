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

Continuum is a unified runtime for LLM and ML programs.
It executes token generation and tensor computation through a shared intermediate representation (IR), so caching, dispatch, and interoperability are handled by one system instead of ad-hoc glue.

## Quick Start

Install:

```bash
python -m pip install continuum-ai
```

Use:

```python
import continuum
```

Run a reproducible benchmark validation:

```bash
PYTHONPATH=python python scripts/benchmarks/run_examples.py | python scripts/benchmarks/validate_outputs.py
```

Try the main example:

- `examples/01_research_agent.py`

## Why Continuum

- **One runtime model**: token ops and tensor ops run in one executable graph.
- **Backend-agnostic cache reuse**: reusable state handles enable cross-call prefix reuse without backend-specific app code.
- **Capability-driven dispatch**: backends are selected by declared capabilities (tensor, token, cache).
- **Explicit tensor interoperability**: cross-backend conversions are explicit and type-tagged.
- **Native-first core**: C++ engine and ABI-focused design for long-term extensibility.

## What Is Implemented

- C++ execution engine with IR interpreter
- KV cache index with canonical prefix normalization
- Azure backend for real network execution
- libtorch backend for tensor/training execution
- MLX backend for Apple-native tensor paths

## Current Status

- v1 release hardening in progress
- CIR schema lock with serialization conformance (`schema/cir.fbs`)
- Linux and macOS CI matrix with coverage gates and fuzz workflow
- PyPI packaging under `continuum-ai` (import path remains `continuum`)

## Documentation

- Python API docs: `https://ct.rithul.dev/python/`
- C++ API docs: `https://ct.rithul.dev/cpp/`

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

- Contributing guide: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Support guide: `SUPPORT.md`

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
