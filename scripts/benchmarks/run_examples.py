#!/usr/bin/env python3
"""Run reproducible example/benchmark programs and emit JSON summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _run(example: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / example)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "example": example,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    targets = [
        "02_deterministic_m1.py",
        "03_transformer_from_scratch.py",
        "04_m2_benchmark_validation.py",
    ]
    rows = [_run(t) for t in targets]
    print(json.dumps({"results": rows}, indent=2))
    bad = [r for r in rows if r["exit_code"] != 0]
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
