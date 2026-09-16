#!/usr/bin/env bash
# Build continuum_core+tests with --coverage, run ctest, emit gcovr HTML/XML.
# Usage: scripts/cpp-coverage.sh [build-dir] [fail-under]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build-coverage}"
FAIL_UNDER="${2:-}"
if [[ -z "$FAIL_UNDER" && -f "$ROOT/.cpp-cov-fail-under" ]]; then
  FAIL_UNDER="$(tr -d '[:space:]' < "$ROOT/.cpp-cov-fail-under")"
fi
FAIL_UNDER="${FAIL_UNDER:-0}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null; then
  PYTHON="$(command -v python3)"
else
  echo "python3 required" >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCONTINUUM_BUILD_TESTS=ON \
  -DCONTINUUM_BUILD_PYTHON=OFF \
  -DCONTINUUM_COVERAGE=ON \
  -DCONTINUUM_WERROR=OFF \
  -DPython3_EXECUTABLE="$PYTHON"
cmake --build "$BUILD_DIR" -j
ctest --test-dir "$BUILD_DIR" --output-on-failure

REPORT_DIR="$BUILD_DIR/coverage"
mkdir -p "$REPORT_DIR"

if command -v uv >/dev/null; then
  uv pip install --python "$PYTHON" 'gcovr>=7.2'
else
  "$PYTHON" -m pip install --quiet 'gcovr>=7.2'
fi
GCOVR="$("$PYTHON" -c 'import shutil,sys; print(shutil.which("gcovr") or "")')"
if [[ -z "$GCOVR" ]]; then
  GCOVR="$ROOT/.venv/bin/gcovr"
fi

"$GCOVR" --root "$ROOT" \
  --object-directory "$BUILD_DIR" \
  --exclude '.*/_deps/.*' \
  --exclude '.*/googletest/.*' \
  --gcov-ignore-errors=all \
  --html-details "$REPORT_DIR/index.html" \
  --xml "$REPORT_DIR/coverage.xml" \
  --print-summary \
  --fail-under-line "$FAIL_UNDER"

echo "HTML report: $REPORT_DIR/index.html"
