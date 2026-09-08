# Continuum dev tasks. Run `just` with no arguments to list them.
#
# Compile parallelism cap. The Python extension links libtorch with LTO, and an
# unbounded parallel build can exhaust memory on a 16 GB machine. Raise it if
# you have the headroom: `just jobs=6 sync`.
jobs := "2"

# Show the recipe list.
default:
    @just --list

# --- environment --------------------------------------------------------

# Create/refresh the venv with all extras and build the native extension.
sync:
    CMAKE_BUILD_PARALLEL_LEVEL={{jobs}} uv sync --all-extras

# Rebuild the native extension after C++ changes (uv does not auto-rebuild).
rebuild:
    CMAKE_BUILD_PARALLEL_LEVEL={{jobs}} uv sync --all-extras --reinstall-package continuum-ai

# --- build & test -----------------------------------------------------

# Configure and build the C++ core plus tests into build/ (no Python).
build:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCONTINUUM_BUILD_TESTS=ON
    cmake --build build -j {{jobs}}

# Python test suite.
test:
    uv run --extra test pytest

# C++ test suite. Run `just build` first.
test-cpp:
    ctest --test-dir build --output-on-failure

# Ruff + mypy.
lint:
    uv run --extra dev ruff check python tests
    uv run --extra dev mypy python/continuum

# Autoformat and autofix.
fmt:
    uv run --extra dev ruff format python tests
    uv run --extra dev ruff check --fix python tests

# pre-commit across the whole tree.
precommit:
    uv run --extra dev pre-commit run --all-files

# Deterministic reproducibility check for the example scripts.
bench:
    bash scripts/bench.sh

# --- docs -------------------------------------------------------------

# Build both doc sets.
docs: docs-py docs-cpp

# Python API docs -> docs/api/python/_build (warnings are errors).
docs-py:
    PYTHONPATH=python uv run --extra docs sphinx-build -b html -W --keep-going docs/api/python docs/api/python/_build

# C++ API docs -> docs/api/cpp/html. Vendors doxygen-awesome-css on first run.
docs-cpp:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d docs/doxygen-awesome-css ]; then
      git clone --depth 1 --branch v2.3.4 \
        https://github.com/jothepro/doxygen-awesome-css.git docs/doxygen-awesome-css
    fi
    doxygen docs/Doxyfile

# Build the Python docs and serve them at http://localhost:8000.
docs-serve: docs-py
    python3 -m http.server 8000 -d docs/api/python/_build

# Serve the landing page at http://localhost:8001.
web:
    python3 -m http.server 8001 -d web

# --- housekeeping ---------------------------------------------------

# Remove the venv, C++ build, and all generated doc output.
clean:
    rm -rf build .venv docs/api/python/_build docs/api/cpp docs/api/python/_static/logo.svg

# Remove only the generated doc output.
clean-docs:
    rm -rf docs/api/python/_build docs/api/cpp docs/api/python/_static/logo.svg
