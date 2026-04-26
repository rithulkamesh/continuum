#!/usr/bin/env python3
"""Validate benchmark/example outputs against expected invariants."""

from __future__ import annotations

import json
import sys


def main() -> None:
    data = json.load(sys.stdin)
    failures: list[str] = []
    for row in data.get("results", []):
        if row.get("exit_code") != 0:
            failures.append(f"{row.get('example')} exit_code={row.get('exit_code')}")
            continue
        out = row.get("stdout", "")
        example = row.get("example")
        if example == "02_deterministic_m1.py":
            if "cache_hit_rate=80.00%" not in out:
                failures.append("deterministic_m1 cache hit ratio mismatch")
        if example == "03_transformer_from_scratch.py":
            if "transformer output [0.0, 0.0]" not in out:
                failures.append("transformer output mismatch")
        if example == "04_m2_benchmark_validation.py":
            if "M2 BENCHMARK VALIDATION" not in out:
                failures.append("m2 benchmark missing header")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("All benchmark validations passed.")


if __name__ == "__main__":
    main()
