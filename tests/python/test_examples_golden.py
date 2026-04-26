from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
GOLDEN = ROOT / "tests" / "data" / "golden"


def _run(example: str) -> str:
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
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_transformer_from_scratch_matches_golden() -> None:
    out = _run("03_transformer_from_scratch.py")
    golden = (GOLDEN / "03_transformer_from_scratch.txt").read_text().strip()
    assert out == golden


def test_deterministic_m1_meets_acceptance() -> None:
    out = _run("02_deterministic_m1.py")
    assert "cache_hit_rate=80.00%" in out
    assert "acceptance cache_hit>=80%=True latency_reduction>=20%=True" in out


def test_m2_benchmark_is_reproducible() -> None:
    out = _run("04_m2_benchmark_validation.py")
    assert "M2 BENCHMARK VALIDATION" in out
    assert "dataset_size_per_seed: 200" in out
    assert "train_test_split: 80/20" in out
