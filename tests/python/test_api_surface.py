from __future__ import annotations

import continuum as ct
from continuum._native import NodeKind
import continuum._native as native
from continuum.frontend.optimizer import Optimizer
from continuum.frontend.param import Param
from continuum.frontend.trace import emit
from continuum.frontend.trace import trace
from continuum.nn.module import Module
from continuum.programs.program import program


def test_public_helpers_and_lm() -> None:
    assert ct.retrieve("q", 2) == ["doc:q:0", "doc:q:1"]
    assert ct.classify("math please", ["math", "no_math"]) == "math"
    assert ct.extract_expr("2+2") == "2+2"
    assert ct.format(2.5) == "2.5"
    assert ct.format_prompt("a", "b") == "a b"
    assert ct.critique_prompt("draft") == "critique: draft"
    assert ct.refine("d", "c") == "d | c"
    lm = ct.LM("fake/model")
    assert lm("hello") == "fake/model:hello"
    marker = ct.tool(lambda x: x)
    assert marker("ok") == "ok"


def test_param_variants_cover_all_branches() -> None:
    assert Param.tensor((2, 2), dtype="f32").metadata["shape"] == (2, 2)
    assert Param.text("x").value == "x"
    assert Param.fewshot(2).metadata["fewshot"] == 2
    assert Param.lora(4).metadata["rank"] == 4
    assert Param.discrete("a", choices=["a", "b"]).metadata["choices"] == ["a", "b"]
    cont = Param.continuous(0.5, min_value=0.1, max_value=0.9)
    assert cont.metadata["min"] == 0.1 and cont.metadata["max"] == 0.9


def test_trace_emit_eager_path() -> None:
    out = emit(NodeKind.TensorOp, None, [[1.0, 2.0]])
    assert isinstance(out, list)


def test_trace_emit_builder_path() -> None:
    with trace():
        out = emit(NodeKind.TensorOp, None, [[1.0, 2.0]])
    assert out is not None


def test_module_forward_contract() -> None:
    m = Module()
    try:
        m.forward()
        assert False
    except RuntimeError:
        assert True


def test_module_parameter_and_submodule_registration() -> None:
    parent = Module()
    parent.p = Param.tensor((1,))
    parent.child = Module()
    assert len(list(parent.parameters())) == 1


def test_program_compile_and_cir_accessors() -> None:
    @program
    def f(x):
        return x

    compiled = f.compile("mlx")
    assert compiled["target"] == "mlx"
    assert f.cir() is None


def test_optimizer_branch_coverage() -> None:
    class Prog:
        def __init__(self):
            self.ps = [
                Param.tensor((1,), initial=1.0),
                Param.text("hello", tokens=["x"], trials=1),
                Param.discrete("a", choices=["a", "b"]),
                Param.continuous(0.1, min_value=0.0, max_value=1.0, trials=1),
            ]

        def __call__(self, x):
            return x

        def parameters(self):
            return iter(self.ps)

    p = Prog()
    opt = Optimizer(p, metric=lambda y, x: 1.0, seed=42)
    opt.step([1])

    # Hit non-numeric tensor branch and empty token pool branch.
    p.ps[0].value = None
    p.ps[1].value = ""
    p.ps[1].metadata["tokens"] = []
    opt.step([1])


def test_native_runtime_fallback_helpers() -> None:
    fn = native._missing_runtime_fn("x")
    try:
        fn()
        assert False
    except RuntimeError:
        assert True
    result = native.benchmark_deterministic_m1(cost_per_token_ms=2.0)
    assert "cache_hit_rate" in result
