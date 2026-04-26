from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from continuum._native import GraphBuilder, NodeKind, eager_step

_active_builder: Any | None = None


@contextmanager
def trace():
    global _active_builder
    builder = GraphBuilder()
    prev, _active_builder = _active_builder, builder
    try:
        yield builder
    finally:
        _active_builder = prev


def emit(kind: Any, payload, inputs, out_type=None, effect=None):
    if _active_builder is None:
        return eager_step(kind, payload, inputs, out_type, effect)
    return _active_builder.add(kind, payload, inputs, out_type, effect)
