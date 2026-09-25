# ABI Contract

- pybind11 module `_continuum` exposes `ir`, `runtime`, and `backend` submodules.
- Python/C++ boundary converts at edge; no deep Python object propagation in C++.
- Calls that can block on a backend (`DurableAgent.run_until_step` /
  `resume_from`, `Session.generate`) release the GIL while they execute.

## Backend C ABI

`include/continuum/backend/backend_abi.h` is a plain-C header that describes a
backend as a function table, `continuum_backend_vtable_t`: capabilities, node
metadata, values (string or token ids), opaque state handles, and run results.

| Version | Vtable fields |
|---------|---------------|
| 1 | `capabilities`, `tensor_backend_type`, `run_with_cache` |
| 2 (current) | v1 + `destroy`, `export_state`, `import_state` (all optional) |

`CONTINUUM_BACKEND_ABI_VERSION` is the host's version; the host accepts any
vtable from `CONTINUUM_BACKEND_ABI_MIN_VERSION` (1) up to it. New fields are
only ever appended, and a v1 vtable's missing fields are never read.

Memory rules: pointers the host passes in are borrowed for the call. Output
strings / token arrays a backend returns must stay valid until the next call
on the same instance; the host copies them immediately. `destroy`, when set,
is called exactly once when the host drops the backend.

In-process, `continuum::backend::MakeBackendFromAbi(vtable)` wraps a vtable
as a `Backend`.

## Backend plugins

A backend can live outside the engine as a shared library built against only
`backend_abi.h`. It exports one versioned entry point:

```c
int continuum_backend_plugin_init_v2(uint32_t host_abi_version,
                                     continuum_backend_vtable_t* out);
```

(`CONTINUUM_BACKEND_PLUGIN_INIT_SYMBOL`, type `continuum_backend_plugin_init_fn`.)
The host zero-fills `*out`, passes its ABI version, and expects the plugin to
fill in `abi_version`, `instance`, and the function pointers, returning 0.

`continuum::backend::LoadBackendPlugin(path)` does the loading: `dlopen`
(`LoadLibrary` on Windows), symbol lookup, init, a `run_with_cache` presence
check, and the ABI version check. Any failure throws with the reason. The
library stays mapped until the backend is destroyed, and `destroy` runs
before it is unmapped.

Registering a plugin:

```python
from continuum._native import BackendRegistry

reg = BackendRegistry()
reg.load_plugin("echo", "/opt/plugins/libcontinuum_echo_backend.so", priority=100)
```

or without code, via the environment:

```bash
# ';'-separated name[@priority]=path entries
export CONTINUUM_BACKEND_PLUGINS="echo@100=/opt/plugins/libcontinuum_echo_backend.so"
```

```python
reg.load_plugins_from_env()          # C++: BackendRegistry::load_plugins_from_env()
```

The registry routes each node to the highest-priority backend whose
capabilities cover it, so a plugin with a higher priority than the built-ins
takes over that node kind.

A complete example, ~150 lines of C, lives in
[`examples/plugins/echo_backend/`](../../examples/plugins/echo_backend/). The
C++ test suite builds it from the public header alone and loads it.

## Backend conformance

`continuum::backend::RunBackendConformance(backend, name)` (in
`include/continuum/backend/conformance.hpp`) is **the bar a new backend must
clear**. It drives the backend through the checks below, applying only the
ones its declared capabilities call for, and returns a pass / fail / skip
report. Run it from the shell against a built-in, a plugin path, or a
`CONTINUUM_BACKEND_PLUGINS` name:

```bash
python -m continuum.backend_check /opt/plugins/libcontinuum_echo_backend.so
python -m continuum.backend_check fake
python -m continuum.backend_check vllm --nondeterministic   # sampling / remote
```

or from Python with `continuum._native.check_backend(target)`.

With `L` the prompt length the kit sends and `P` its cached-prefix length:

| Check | Applies when | Contract |
|-------|--------------|----------|
| `capabilities.declared` | always | supports tensor and/or token |
| `capabilities.cache_implies_token` | always | `supports_cache` requires `supports_token` |
| `token.cold_run` | token | no throw; output is a string or token ids |
| `token.cold_metrics` | token | without prefix state: `used_cached_state = false`, `reused_prefix_len = 0`, `tokens_saved = 0`, `tokens_sent >= 0`, `compute_steps >= 0` |
| `token.deterministic` | token, unless disabled | identical cold inputs at temperature 0 give identical output |
| `cache.state_handle` | cache | a run returns a non-null `resulting_state` |
| `cache.warm.*` | cache | given that state and `remaining_tokens = L - P`: `used_cached_state = true`; `0 <= reused_prefix_len <= L`; `0 < tokens_saved <= L` |
| `cache.warm_output_matches_cold` | cache, deterministic | reusing a prefix does not change the output |
| `state.export` / `import` / `roundtrip` | cache, and `export_state` returns bytes | `import_state` accepts exported bytes; `export(import(b)) == b` |
| `state.imported_warm.*` | as above | a warm run from an imported state meets the `cache.warm.*` contract (this is what checkpoint resume relies on) |
| `tensor.<op>` / `tensor.no_reuse_claimed` | tensor | the identity op (configurable) returns its input unchanged and claims no reuse |

State portability is optional: a backend whose `export_state` returns nothing
skips the `state.*` checks, and checkpoints then resume it with a cold cache.

`fake_llm`, `libtorch`, the offline `vllm` shim, and the example echo plugin
pass; see `tests/cpp/test_backend_plugins.cpp` and
`tests/python/test_backend_plugins.py`.
