from __future__ import annotations

from continuum._native import GraphBuilder, NodeKind


def test_graph_builder_accumulates_nodes() -> None:
    g = GraphBuilder()
    for _ in range(100):
        node = g.add(NodeKind.TokenOp, {"model_id": "anthropic/demo"}, [1, 2, 3], None, None)
        assert len(node) == 5
    summary = g.finalize().run()
    assert summary["nodes"] == 100
