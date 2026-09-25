"""Eviction contract and byte accounting for every reuse tier (issue #11).

Each test fills a tier past capacity and asserts which entries survive, per
the policy documented in docs/design/cache.md ("Eviction and memory bounds").
"""

from __future__ import annotations

import time

from continuum._native import (
    BackendRegistry,
    FutureCache,
    LayerKVCacheIndex,
    MemoKey,
    MemoryGraphStore,
    MemoTable,
    SemanticCacheIndex,
    Session,
)


def _key(i: int) -> MemoKey:
    return MemoKey("ToolOp", f"h{i}", bytes([i]))


def _onehot(i: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# --- memo: LRU ---------------------------------------------------------------


def test_memo_lru_evicts_least_recently_used():
    mt = MemoTable(3, 0)
    for i in range(3):
        mt.insert(_key(i), b"out", 0)
    assert mt.lookup(_key(0)) is not None  # refresh 0; 1 is now the LRU
    mt.insert(_key(3), b"out", 0)
    assert mt.size() == 3
    assert mt.lookup(_key(1)) is None
    for i in (0, 2, 3):
        assert mt.lookup(_key(i)) is not None


def test_memo_zero_capacity_stores_nothing():
    mt = MemoTable(0, 0)
    mt.insert(_key(0), b"out", 0)
    assert mt.size() == 0


def test_memo_bytes_track_contents():
    mt = MemoTable(10, 0)
    assert mt.estimated_bytes() == 0
    mt.insert(_key(0), b"x" * 1000, 0)
    one = mt.estimated_bytes()
    assert one >= 1000
    mt.insert(_key(1), b"x" * 1000, 0)
    assert mt.estimated_bytes() > one
    mt.clear()
    assert mt.estimated_bytes() == 0
    assert mt.max_entries() == 10


# --- semantic: LRU -------------------------------------------------------------


def test_semantic_lru_hit_refreshes_entry():
    sc = SemanticCacheIndex(3, 0.99)
    for i in range(3):
        sc.insert(_onehot(i), "m", bytes([i]))
    assert sc.lookup(_onehot(0), "m")["above_threshold"]  # refresh 0
    sc.insert(_onehot(3), "m", b"\x03")
    assert sc.size() == 3
    assert not sc.lookup(_onehot(1), "m")["above_threshold"]
    for i in (0, 2, 3):
        r = sc.lookup(_onehot(i), "m")
        assert r["above_threshold"]
        assert r["output"] == bytes([i])


def test_semantic_zero_capacity_and_bytes():
    assert SemanticCacheIndex(0, 0.9).size() == 0
    sc0 = SemanticCacheIndex(0, 0.9)
    sc0.insert(_onehot(0), "m", b"x")
    assert sc0.size() == 0

    sc = SemanticCacheIndex(4, 0.9)
    sc.insert(_onehot(0, 64), "m", b"x" * 100)
    assert sc.estimated_bytes() >= 64 * 4 + 100
    assert sc.max_entries() == 4


def test_semantic_namespace_isolated():
    sc = SemanticCacheIndex(4, 0.9)
    sc.insert(_onehot(0), "m", b"a", cache_namespace="t1")
    assert sc.lookup(_onehot(0), "m", cache_namespace="t1")["above_threshold"]
    assert not sc.lookup(_onehot(0), "m", cache_namespace="t2")["above_threshold"]


# --- memory graph: FIFO -------------------------------------------------------


def test_memory_graph_fifo_ignores_reads():
    mg = MemoryGraphStore(3)
    ids = [mg.add_node(f"n{i}", _onehot(i)) for i in range(3)]
    # A read does not protect a node under FIFO.
    assert mg.retrieve_similar(_onehot(0), 5, 0.9)
    mg.add_node("n3", _onehot(3))
    assert mg.size() == 3
    assert mg.get_node(ids[0]) is None
    assert mg.get_node(ids[1])["content"] == "n1"


def test_memory_graph_zero_capacity_and_bytes():
    mg0 = MemoryGraphStore(0)
    mg0.add_node("x", _onehot(0))
    assert mg0.size() == 0

    mg = MemoryGraphStore(10)
    assert mg.estimated_bytes() == 0
    mg.add_node("x" * 500, _onehot(0, 32), cache_namespace="ns")
    assert mg.estimated_bytes() >= 500 + 32 * 4
    assert mg.max_nodes() == 10
    assert mg.retrieve_similar(_onehot(0, 32), cache_namespace="other") == []


# --- layer KV: LRU under entry + byte bounds --------------------------------------


def test_layer_cache_entry_bound_lru():
    lc = LayerKVCacheIndex(2, 1 << 30)
    lc.insert("m", [1, 2], 1, 10)
    lc.insert("m", [3, 4], 1, 10)
    assert lc.find_deepest("m", [1, 2, 9], 4)["found"]  # refresh [1,2]
    lc.insert("m", [5, 6], 1, 10)
    assert lc.size() == 2
    assert not lc.find_deepest("m", [3, 4], 4)["found"]
    assert lc.find_deepest("m", [1, 2], 4)["found"]
    assert lc.find_deepest("m", [5, 6], 4)["found"]


def test_layer_cache_byte_bound():
    lc = LayerKVCacheIndex(100, 250)
    for i in range(5):
        lc.insert("m", [i], 1, 100)
    assert lc.estimated_bytes() <= 250
    assert lc.size() == 2
    assert lc.max_bytes() == 250
    assert lc.max_entries() == 100


# --- future cache: TTL then FIFO ------------------------------------------------


def test_future_cache_fifo_and_ttl():
    fc = FutureCache(2, 10_000)
    fc.put("a", [1])
    fc.put("b", [2])
    fc.put("c", [3])
    assert fc.size() == 2
    assert not fc.has("a")
    assert fc.estimated_bytes() > 0
    assert fc.max_entries() == 2

    short = FutureCache(4, 1)
    short.put("x", [1])
    time.sleep(0.01)
    assert short.get("x") is None
    short.put("y", [2])  # purges the expired entry
    assert short.size() == 1


# --- session-level accounting ----------------------------------------------


def test_session_cache_stats_reports_attached_tiers():
    s = Session("stats", BackendRegistry(), 16)
    stats = s.cache_stats()
    assert set(stats) == {"prefix_kv"}
    assert stats["prefix_kv"]["capacity"] == 16

    mt = MemoTable(5, 0)
    sc = SemanticCacheIndex(6, 0.9)
    lc = LayerKVCacheIndex(7, 1024)
    mg = MemoryGraphStore(8)
    s.set_memo_table(mt)
    s.set_semantic_cache(sc)
    s.set_layer_cache(lc)
    s.set_memory_graph(mg)
    mt.insert(_key(0), b"x" * 50, 0)

    stats = s.cache_stats()
    assert set(stats) == {"prefix_kv", "memo", "semantic", "layer_kv", "memory_graph"}
    assert stats["memo"] == {"entries": 1, "capacity": 5, "bytes": mt.estimated_bytes()}
    assert stats["semantic"]["capacity"] == 6
    assert stats["layer_kv"]["capacity"] == 7
    assert stats["memory_graph"]["capacity"] == 8
