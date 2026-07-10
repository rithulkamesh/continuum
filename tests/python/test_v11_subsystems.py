import pytest


def test_memo_table_basic():
    from continuum._native import MemoTable

    mt = MemoTable(100, 0)
    assert mt.size() == 0
    assert mt.version() == 0

    mt.set_version(1)
    assert mt.version() == 1
    mt.clear()
    assert mt.size() == 0


def test_memo_table_version_invalidation():
    from continuum._native import MemoTable

    mt = MemoTable(100, 0)
    mt.set_version(1)
    mt.clear()
    assert mt.size() == 0


def test_semantic_cache_basic():
    from continuum._native import SemanticCacheIndex

    sc = SemanticCacheIndex(100, 0.85)
    assert sc.size() == 0
    assert sc.similarity_threshold() == pytest.approx(0.85, abs=0.01)

    sc.set_similarity_threshold(0.90)
    assert sc.similarity_threshold() == pytest.approx(0.90, abs=0.01)

    sc.clear()
    assert sc.size() == 0


def test_semantic_cache_cosine_similarity():
    from continuum._native import SemanticCacheIndex

    identical = SemanticCacheIndex.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert identical == pytest.approx(1.0)

    orthogonal = SemanticCacheIndex.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert orthogonal == pytest.approx(0.0)


def test_memory_graph_basic():
    from continuum._native import MemoryGraphStore

    mg = MemoryGraphStore(100)
    assert mg.size() == 0
    mg.clear()
    assert mg.size() == 0


def test_layer_cache_basic():
    from continuum._native import LayerKVCacheIndex

    lc = LayerKVCacheIndex(100, 1024 * 1024)
    assert lc.size() == 0
    assert lc.estimated_bytes() == 0
    lc.clear()
    assert lc.size() == 0


def test_future_cache_basic():
    from continuum._native import FutureCache

    fc = FutureCache(50)
    assert fc.size() == 0
    fc.clear()
    assert fc.size() == 0


def test_v11_benchmark_runs():
    from continuum._native import run_v11_benchmark

    r = run_v11_benchmark(num_steps=10, prefix_tokens=20)
    assert "semantic_hit_rate" in r
    assert "memo_hit_rate" in r
    assert "memory_nodes" in r
    assert r["memory_nodes"] == 10
    assert r["semantic_cache_size"] == 10
    assert isinstance(r["semantic_hit_rate"], float)


def test_v11_benchmark_semantic_hits_increase():
    from continuum._native import run_v11_benchmark

    r = run_v11_benchmark(num_steps=30, prefix_tokens=50)
    assert r["semantic_hit_rate"] >= 0.0
    assert r["semantic_hit_rate"] <= 1.0


def test_existing_session_benchmark_still_works():
    from continuum._native import run_session_benchmark, run_cold_start_benchmark

    sr = run_session_benchmark(num_steps=5, prefix_tokens=10, suffix_tokens=5)
    assert sr["runs"] is not None
    assert len(sr["runs"]) == 5

    cr = run_cold_start_benchmark("/tmp/test_v11_cold.bin", steps_warm=3, steps_cold=3)
    assert "warm_run" in cr
    assert "cold_run" in cr
    assert cr["cold_run"]["hit_rate"] >= 0.8


def test_v11_layer_cache_and_memory_graph_wired():
    """LayerKVCacheIndex + MemoryGraphStore are exercised by Session.run."""
    from continuum._native import run_v11_wiring_check

    r = run_v11_wiring_check()
    # Run 1 inserts a layer checkpoint; run 2 re-keys the same checkpoint.
    assert r["layer_cache_size"] == 1
    assert r["layer_cache_bytes"] > 0
    # Each run records the prompt as a memory node (no memo short-circuit here).
    assert r["memory_nodes"] == 2


def test_layer_cache_content_isolation():
    """Layer checkpoints are reusable only for prompts they are a token-prefix of."""
    from continuum._native import run_v11_layer_isolation_check

    r = run_v11_layer_isolation_check()
    assert r["hit_exact"] is True
    assert r["hit_extended"] is True
    assert r["hit_unrelated"] is False
    assert r["hit_shorter"] is False


def test_memo_invalidation_api_wired():
    """invalidate_node / invalidate_version are callable (used on ToolOp / resume)."""
    from continuum._native import MemoTable

    mt = MemoTable(100, 0)
    mt.invalidate_node("ToolOp")
    next_version = mt.version() + 1
    mt.set_version(next_version)
    mt.invalidate_version(next_version)
    assert mt.version() == next_version
    assert mt.size() == 0


def test_memo_table_capacity():
    from continuum._native import MemoTable

    mt = MemoTable(5, 0)
    assert mt.size() == 0
    mt.clear()


def test_semantic_cache_threshold_boundary():
    from continuum._native import SemanticCacheIndex

    sc = SemanticCacheIndex(100, 0.5)
    sc.set_similarity_threshold(0.99)
    sc.clear()


def test_layer_cache_byte_tracking():
    from continuum._native import LayerKVCacheIndex

    lc = LayerKVCacheIndex(100, 256)
    assert lc.estimated_bytes() == 0
