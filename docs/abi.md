# ABI Contract v0.1

- pybind11 module `_continuum` exposes `ir`, `runtime`, and `backend` submodules.
- Python/C++ boundary converts at edge; no deep Python object propagation in C++.
- Long-running C++ calls should release GIL in future hot-path implementations.

## Backend ABI Preparation (v1 groundwork)

Continuum now defines a C-compatible backend ABI boundary in
`include/continuum/backend/backend_abi.h`.

- ABI version constant: `CONTINUUM_BACKEND_ABI_VERSION`
- C structs for capabilities, node metadata, values, cache state, and run results
- Backend function table (`continuum_backend_vtable_t`) suitable for dynamic loading later

The current runtime remains C++ native. For forward compatibility, a bridge adapter is available:

- `src/backend/backend_abi_adapter.cpp`
- `continuum::backend::MakeBackendFromAbi(...)`

This allows static linking today while isolating a stable C ABI seam for future plugin loading.
