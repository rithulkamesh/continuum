#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCONTINUUM_BUILD_TESTS=ON \
  -DCONTINUUM_BUILD_PYTHON=OFF
cmake --build build -j
