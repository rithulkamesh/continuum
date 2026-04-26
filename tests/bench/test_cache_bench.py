import time

from continuum._native import GraphBuilder


def test_cache_hitrate_smoke():
    t0 = time.time()
    g = GraphBuilder()
    for _ in range(100):
        g.add(1, {"model_id": "anthropic/demo"}, [1, 2, 3], None, None)
    elapsed = time.time() - t0
    assert elapsed < 1.0
