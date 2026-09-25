"""The E10 vLLM benchmark's plumbing, against its simulated server."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "scripts" / "e10_vllm_prefix_reuse.py"


def test_e10_self_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    module = runpy.run_path(str(SCRIPT))
    result = module["main"](["--self-test", "--trials", "3", "--prefix-chars", "800"])
    assert result["simulated"] is True
    assert result["cold"]["server_cached_tokens"] == 0
    assert result["warm"]["server_cached_tokens"] > 0
    assert result["warm"]["p50_ms"] < result["cold"]["p50_ms"]
    with pytest.raises(SystemExit):
        module["main"]([])
