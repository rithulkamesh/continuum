# Code Standards Audit

Baseline review of `src/`, `include/`, `bindings/`, and `python/` against the
C++ standards in `.claude/skills/cpp-standards/SKILL.md` and against the Python
tooling. Configs added in the same pass: `.clang-format`, `.clang-tidy`,
`.editorconfig`.

Status legend: **fixed** in this pass, **config** now enforced by a checked-in
config, **open** needs a follow-up change.

## C++

| # | Finding | Severity | Status |
|---|---|---|---|
| C1 | **No Doxygen anywhere.** 0 of 27 public headers carry `@brief`/`@param`/`@return`, though `Doxyfile` runs in CI and publishes to `ct.rithul.dev/cpp`. | high | open |
| C2 | **Owning raw pointers across the backend-state seam.** `src/backend/azure_openai.cpp`, `src/backend/vllm_shim.cpp` do `new PrefixStateData{...}` and store it in `BackendState.handle` (a `void*`). Ownership and deletion are manual. `cppcoreguidelines-owning-memory` will flag this. Proper fix: give `BackendState` a typed owner (`std::shared_ptr<void>` with deleter, or a small variant) and thread it through `backend_abi.h` and checkpoint serialization. | high | config, open |
| C3 | **`CURL*` lifetime is manual.** `azure_openai.cpp` acquires `curl_easy_init()` / `curl_slist_append` and frees them by hand. An exception between init and cleanup leaks. Wrap in an RAII holder (`std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>` and a `curl_slist` guard). | medium | open |
| C4 | **`pybind` factory lambdas return `new T(...)`.** `bind_runtime.cpp` (`Session`, `FutureCache`, `BruteForceEmbeddingProvider`). pybind then owns them, so it is not a leak, but `std::make_unique` + `py::return_value_policy` or the pointer-holder form reads better and satisfies the check. | low | config |
| C5 | **`reinterpret_cast<void*>` round-trips for state handles.** Inherent to the C ABI seam, but should be isolated to one adapter file with a documented contract rather than repeated per backend. | medium | open |
| C6 | **No `const` on locals that never change** in several runtime hot paths. `misc-const-correctness` in the new `.clang-tidy` will surface the list. | low | config |
| C7 | **Missing special-member declarations.** Classes that own resources (`KVCacheIndex`, `LayerKVCacheIndex`, `Session`) should state the rule of five explicitly or `= default` / `= delete`. `cppcoreguidelines-special-member-functions` covers it. | medium | config |
| C8 | **`std::memcpy` for type punning** in `ir/graph.cpp`, `runtime/checkpoint.cpp`, `runtime/kv_prefix_cache.cpp`. This is acceptable for serialization and is the portable idiom pre-C++20 `bit_cast`; left as is; `std::bit_cast` is now available (C++20) and is the preferred follow-up. | info | open |
| C9 | **`.clang-format` / `.clang-tidy` were absent.** The tree is already close to Google style with 2-space indent and 100-column lines; the new `.clang-format` encodes exactly that so it is close to a no-op on existing files. | medium | fixed |
| C10 | **CI was red on `master` since 2026-07-11**, unrelated to any repo change: a newer libtorch release requires a C++20 compiler while `CMakeLists.txt` pinned `CMAKE_CXX_STANDARD 17` (`#error C++20 or later compatible compiler is required to use PyTorch`). Bumped to C++20. | high | fixed |

### C++ naming

The project uses `PascalCase` for free functions and file-local helpers,
`snake_case` for locals and for methods on bound classes, `PascalCase` for
types. This is internally consistent, so `.clang-tidy` does **not** enable
`readability-identifier-naming` (no single profile matches, and a rename pass
would be pure churn). New code should match the file it lives in.

## Python

| # | Finding | Severity | Status |
|---|---|---|---|
| P1 | **`ruff` lint relied on implicit defaults, which drifted.** A newer ruff widened its default rule set (I/RUF/UP), breaking CI. Pinned `select = ["E4", "E7", "E9", "F"]` (historical default) and `ruff>=0.15,<0.16`. Adopting the wider set (`I`, `B`, `SIM`, `UP`) is still the goal, once the toolchain runs locally. | medium | fixed (config) |
| P2 | **`mypy` is loose.** `ignore_missing_imports = true` is justified for the native extension, but `warn_redundant_casts`, `warn_unused_ignores`, and `strict_equality` should be added once validated. | low | open |
| P3 | **`_native.py` masked binary/Python drift.** ~40 symbols pulled via `getattr(_c.runtime, name, None)` then replaced with `_missing_runtime_fn` shims, plus a full Python reimplementation of `benchmark_deterministic_m1`. A stale `.so` failed deep in a benchmark instead of at import. Rewritten to bind every name directly with an explicit `__all__`; a stale build now fails loudly at import. | medium | fixed |
| P4 | **Fake DSL helpers in the public namespace.** `continuum.retrieve/classify/extract_expr/format/format_prompt/critique_prompt/refine` and `LM` were hardcoded stubs, tested only by asserting the stub returns its constant. Removed. Public surface is now `Optimizer`, `Param`, `program`, `tool`, `nn`. | medium | fixed |
| P5 | **Empty `backends/` package.** `python/continuum/backends/__init__.py` (0 bytes, nothing else). Removed. | low | fixed |
| P6 | **`pyproject` license mismatch.** File and badge say MIT; classifier said "Apache Software License". Fixed, and `authors` set to the real author. | low | fixed |
| P7 | **`tests/bench/` outside `testpaths`.** `test_cache_bench.py` never ran under `pytest` and asserted wall-clock timing. Moved to `tests/python/test_graph_builder.py` and rewritten to assert behavior. | low | fixed |

## Follow-up work

1. Document public headers (C1). Start with `ir/`, then `runtime/session.hpp`
   and `runtime/interpreter.hpp`.
2. Introduce a typed owner for backend state and collapse the `void*` handling
   into one adapter (C2, C5).
3. RAII wrappers for `libcurl` handles (C3).
4. Wire `ruff`/`mypy`/`pytest` into a dev container or documented setup, then
   tighten `ruff` and `mypy` (P1, P2) and run `clang-tidy` in CI.
