"""Public Python API surface for Continuum.

The execution engine lives in the compiled extension, reached through
:mod:`continuum._native`. This module exposes the ergonomic Python frontend:
the :func:`program` tracer, tunable :class:`Param` values, and the
:class:`Optimizer`.
"""

from importlib import metadata as _metadata

from . import nn
from .frontend.optimizer import Optimizer
from .frontend.param import Param


def program(fn):
    """Decorate a Python function as a Continuum program; traced to CIR on first call."""
    from .programs.program import program as _program

    return _program(fn)


def tool(fn):
    """Mark a Python callable as a tool. Currently a pass-through marker."""
    return fn


try:
    __version__ = _metadata.version("continuum-ai")
except _metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0+local"

__all__ = ["Optimizer", "Param", "program", "tool", "nn", "__version__"]
