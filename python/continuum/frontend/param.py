from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class Param:
    """Typed program parameter tracked by the Continuum optimizer."""

    kind: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def tensor(shape=None, dtype="f32", initial=None, **metadata) -> "Param":
        merged = {"shape": tuple(shape) if shape is not None else None, "dtype": dtype}
        merged.update(metadata)
        return Param(kind="tensor", value=initial, metadata=merged)

    @staticmethod
    def text(initial: str, **metadata) -> "Param":
        return Param(kind="text", value=initial, metadata=metadata)

    @staticmethod
    def fewshot(k: int = 3) -> "Param":
        return Param(kind="text", value="", metadata={"fewshot": k})

    @staticmethod
    def lora(rank: int = 8) -> "Param":
        return Param(kind="tensor", value=None, metadata={"adapter": "lora", "rank": rank})

    @staticmethod
    def discrete(initial: Any = 0, *, choices=None, **metadata) -> "Param":
        merged = dict(metadata)
        if choices is not None:
            merged["choices"] = list(choices)
        return Param(kind="discrete", value=initial, metadata=merged)

    @staticmethod
    def continuous(initial: float = 0.0, *, min_value=None, max_value=None, **metadata) -> "Param":
        merged = dict(metadata)
        if min_value is not None:
            merged["min"] = float(min_value)
        if max_value is not None:
            merged["max"] = float(max_value)
        return Param(kind="continuous", value=float(initial), metadata=merged)
