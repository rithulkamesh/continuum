from __future__ import annotations

from hypothesis import given
from hypothesis import settings
import hypothesis.strategies as st
import pytest

from continuum._native import Graph


@settings(max_examples=200, deadline=None)
@given(st.binary(min_size=0, max_size=512))
def test_graph_deserialize_fuzz_does_not_crash(data: bytes) -> None:
    if not hasattr(Graph, "deserialize"):
        pytest.skip("Graph.deserialize not available in loaded native module")
    try:
        Graph.deserialize(data)
    except RuntimeError:
        # Expected for malformed payloads.
        pass
