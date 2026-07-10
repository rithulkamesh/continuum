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


def test_durable_agent_example_resumes_after_crash() -> None:
    out = _run("06_durable_agent.py")
    assert "fresh runtime resumed: workflow completed, 10 node outputs" in out
    assert "KV cache restored across the process boundary:" in out
    assert "determinism check: two resumes produced identical outputs" in out
    assert "durable agent: OK" in out


def test_time_travel_fork_example_diverges_only_at_edit() -> None:
    out = _run("07_time_travel_fork.py")
    assert "divergence: 2/8 node outputs differ" in out
    assert "replay check: completed steps identical, only the edit diverged" in out
    assert "time-travel fork: OK" in out


def test_reuse_stack_example_shows_all_five_tiers() -> None:
    out = _run("05_continuum_reuse_stack.py")
    assert "Continuum - Full Reuse Stack" in out
    # All five reuse tiers reported in the scoreboard.
    for tier in ("Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"):
        assert tier in out
    # Trie warm reuse and semantic catch are deterministic on the fake backend.
    assert "84.9% tokens saved" in out
    assert "100% hit rate" in out
    # Layer KV + memory graph wiring exercised end to end.
    assert "1 checkpoint(s) reused" in out
    assert "recall fired" in out
