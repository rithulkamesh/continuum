from contextlib import contextmanager

from continuum._native import GraphBuilder, NodeKind, eager_step

_active_builder: GraphBuilder | None = None


@contextmanager
def trace():
    global _active_builder
    builder = GraphBuilder()
    prev, _active_builder = _active_builder, builder
    try:
        yield builder
    finally:
        _active_builder = prev


def emit(kind: NodeKind, payload, inputs, out_type=None, effect=None):
    if _active_builder is None:
        return eager_step(kind, payload, inputs, out_type, effect)
    return _active_builder.add(kind, payload, inputs, out_type, effect)
