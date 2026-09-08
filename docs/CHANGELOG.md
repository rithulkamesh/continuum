# Changelog

## Unreleased

- Raised `CMAKE_CXX_STANDARD` to 20; current libtorch headers require a C++20
  compiler (CI had been failing to build since a torch upgrade).
- Restructured the repo: `docs/design/`, `benchmarks/{scripts,data,plots,reports}/`,
  `examples/` + `examples/milestones/`, community files under `.github/`,
  landing page under `web/`. Paper split to a separate private repo.
- Removed dead Python scaffolding (`backends/` package, fake DSL helpers);
  `continuum._native` now binds the extension directly.
- Renamed `src/runtime/cache.*` to `kv_prefix_cache.*`.
- Added `.clang-format`, `.clang-tidy`, `.editorconfig`, and a C++ standards
  skill. See `docs/design/code-standards-audit.md`.

## 1.0.0 - 2026-04-26

- Locked canonical CIR schema in `schema/cir.fbs` and added schema-layout validation tests.
- Added capability-driven backend routing, tensor interoperability tagging, and explicit cross-backend conversions.
- Introduced backend ABI preparation layer (`backend_abi.h` + C++ adapter bridge).
- Added reproducibility hardening for examples with golden-output tests and benchmark scripts.
- Added Python coverage enforcement (`>=95%`) and nightly fuzz workflow.
- Added Sphinx (Python) and Doxygen (C++) docs pipelines with CI integration.
- Added release preparation artifacts and v1 readiness report.
