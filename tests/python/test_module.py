from continuum.frontend.param import Param
from continuum.nn.module import Module
from continuum.nn.module import Linear


class M(Module):
    def __init__(self):
        super().__init__()
        self.p = Param.tensor((1,))

    def forward(self, x):
        return x


def test_module_parameter_discovery():
    m = M()
    assert len(list(m.parameters())) == 1


def test_linear_forward_produces_output_shape():
    layer = Linear(3, 2)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert len(out) == 2
