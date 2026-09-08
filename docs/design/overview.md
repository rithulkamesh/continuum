# Architecture Overview

Continuum is a unified execution runtime for token, tensor, and tool-shaped
program steps. The core is C++ for execution speed and deterministic behavior.
Python provides ergonomic entrypoints and benchmarking hooks.

## Execution flow

1. Build or load IR (`Graph`, `Node`, payloads). See [ir.md](ir.md).
2. Execute via `runtime::Interpreter`. See [runtime.md](runtime.md).
3. Dispatch each step to a backend (`libtorch`, `azure`, `vllm`, and so on).
4. Feed token workloads through the reuse stack for prefix-aware reuse. See
   [cache.md](cache.md).

## Ownership boundary

The cache is runtime-owned, but reuse depends on backend state handles. That
coupling is intentional: the runtime decides *when* a prefix is reusable, and
the backend decides *how* the state is represented. See [abi.md](abi.md).

Python bindings expose this C++ stack with thin wrappers. The Python layer does
not reimplement scheduling or cache logic. It calls into the same C++ execution
path used by the tests.
