"""Public Python API surface for Continuum."""

from . import nn
from .frontend.optimizer import Optimizer
from .frontend.param import Param


def program(fn):
    """Decorate a Python function as a Continuum program."""
    from .programs.program import program as _program

    return _program(fn)


def tool(fn):
    """Mark a Python callable as a tool (v0.1 no-op marker)."""
    return fn


def retrieve(question: str, k: int = 4):
    return [f"doc:{question}:{i}" for i in range(k)]


def classify(question: str, labels):
    return labels[0] if "math" in question.lower() else labels[-1]


def extract_expr(question: str) -> str:
    return question


def format(value) -> str:
    return str(value)


def format_prompt(*parts):
    return " ".join(str(p) for p in parts)


def critique_prompt(draft):
    return f"critique: {draft}"


def refine(draft, critique):
    return f"{draft} | {critique}"


class LM:
    """Minimal language-model callable facade used by examples."""

    def __init__(self, model_id: str, adapter=None):
        self.model_id = model_id
        self.adapter = adapter

    def __call__(self, prompt):
        return f"{self.model_id}:{prompt}"


__all__ = ["Optimizer", "Param", "program", "tool", "LM", "nn"]
