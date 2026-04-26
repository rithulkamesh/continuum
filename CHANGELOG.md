# Changelog

## 1.0.0 - 2026-04-26

- Locked canonical CIR schema in `schema/cir.fbs` and added schema-layout validation tests.
- Added capability-driven backend routing, tensor interoperability tagging, and explicit cross-backend conversions.
- Introduced backend ABI preparation layer (`backend_abi.h` + C++ adapter bridge).
- Added reproducibility hardening for examples with golden-output tests and benchmark scripts.
- Added Python coverage enforcement (`>=95%`) and nightly fuzz workflow.
- Added Sphinx (Python) and Doxygen (C++) docs pipelines with CI integration.
- Added release preparation artifacts and v1 readiness report.
