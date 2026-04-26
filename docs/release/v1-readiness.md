# Continuum v1 Readiness Report

## Build and Test Evidence

- C++: `cmake --build build-tests -j4 && ctest --output-on-failure` passed (`28/28`).
- Python: `PYTHONPATH=python python -m pytest tests/python -q` passed with coverage gate.
- Coverage: Python total coverage `96.92%` (`--cov-fail-under=95` satisfied).

## CIR Schema Validation

- Canonical schema added at `schema/cir.fbs`.
- Runtime binary envelope documented in `docs/ir-spec.md`.
- C++ schema-layout validation test added (`GraphTest.SerializedBinaryMatchesCirSchemaLayout`).

## API Documentation

- Sphinx Python docs scaffold: `docs/api/python/`.
- Doxygen config: `Doxyfile`.
- CI docs job builds Sphinx and Doxygen artifacts.

## Backend ABI Preparation

- C ABI boundary introduced: `include/continuum/backend/backend_abi.h`.
- C++ adapter bridge: `src/backend/backend_abi_adapter.cpp`.
- ABI notes documented in `docs/abi.md`.

## Reproducible Examples

- Golden-output tests for deterministic examples in `tests/python/test_examples_golden.py`.
- Benchmark runner + validator scripts:
  - `scripts/benchmarks/run_examples.py`
  - `scripts/benchmarks/validate_outputs.py`
- Local reproducibility check: `PYTHONPATH=python python scripts/benchmarks/run_examples.py | python scripts/benchmarks/validate_outputs.py` passed.

## CI / Platform Matrix

- CI now includes Linux + macOS matrix for Python job.
- Nightly fuzz workflow added: `.github/workflows/fuzz.yml`.

## Release Artifacts

- Package version set to `1.0.0` in `pyproject.toml`.
- Changelog added: `CHANGELOG.md`.
- Release workflow added: `.github/workflows/release.yml`.

## Issue Risk Summary

- P0 runtime regressions observed in local verification: none.
- Known environment caveat: local Doxygen binary unavailable unless installed; CI job installs it explicitly.

## Operational v1 Criteria (External)

The following items require non-repository evidence and are tracked as release gate inputs:

- Three publicly named external users.
- Issue tracker state: zero P0 issues open at release cut.
