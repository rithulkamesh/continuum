#!/usr/bin/env bash
# Run the deterministic example reproducibility check.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=python python benchmarks/scripts/run_examples.py \
  | python benchmarks/scripts/validate_outputs.py
