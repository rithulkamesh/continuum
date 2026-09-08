# Repo instructions

## Commit and PR hygiene (hard rules)

- Never add `Co-Authored-By:` trailers for AI assistants.
- Never add `Claude-Session:`, `Generated with`, or any other
  tool-attribution line to a commit message or PR description.
- Never commit assistant or editor state. Not `.claude/settings.local.json`,
  `CLAUDE.local.md`, session transcripts, scratch plans, `.cursor/`, or
  `.aider*`. Only `.claude/skills/` and this file are tracked on purpose.
- Keep generated artifacts out of git (docs builds, `build/`, `dist/`).
- Branch off `master` for changes; do not commit or push unless asked.

## Coding standards

- C++: modern C++17. Follow `.clang-format` and `.clang-tidy`, and the
  checklist in `.claude/skills/cpp-standards/SKILL.md` (RAII, no raw
  `new`/`delete`, `const` by default, smart pointers for ownership, raw
  pointers only for non-owning observation, `enum class` over raw ints,
  Doxygen on public declarations). Existing free functions use `PascalCase`
  and locals use `snake_case`; match the file you are in.
- Python: `ruff` and `mypy` must pass. Public API surface is
  `continuum` (frontend) and `continuum._native` (the compiled engine).

## Layout

- `src/`, `include/continuum/` mirror each other: engine core.
- `bindings/pybind/` splits by domain: `bind_ir`, `bind_runtime`, `bind_backend`.
- `python/continuum/` is the Python frontend; `_native.py` re-exports the extension.
- `examples/` holds the three product demos; `examples/milestones/` holds
  parity and training demos.
- `benchmarks/` holds `scripts/`, `data/`, `plots/`, `reports/`.
- `docs/` guides at the top level, design notes under `docs/design/`.
- `web/` is the GitHub Pages landing page.

## Build and test

```bash
scripts/build.sh          # cmake configure + build into build/
pytest                    # Python suite (tests/python)
ctest --test-dir build    # C++ suite, when configured with -DCONTINUUM_BUILD_TESTS=ON
pre-commit run --all-files
```
