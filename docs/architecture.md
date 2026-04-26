# Continuum Architecture

Continuum is a unified execution runtime for token, tensor, and tool-shaped program steps. The core is in C++ for execution speed and deterministic runtime behavior, while Python provides ergonomic entrypoints and benchmarking hooks.

Execution flow is:

1. Build or load IR (`Graph`, `Node`, payloads).
2. Execute via `runtime::Interpreter`.
3. Dispatch each step to a backend (`libtorch`, `azure`, `vllm`, etc.).
4. Feed token workloads through `KVCacheIndex` for prefix-aware reuse.

The cache is runtime-owned, but reuse depends on backend state handles. That coupling is intentional: runtime decides *when* a prefix is reusable, backend decides *how* the state is represented.

Python bindings expose this C++ stack with thin wrappers. The Python layer does not reimplement scheduling or cache logic; it calls into the same C++ execution path used by tests.
