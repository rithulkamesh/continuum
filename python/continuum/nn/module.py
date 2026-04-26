from continuum.frontend.param import Param


class Module:
    """Minimal nn.Module-like base class for parameter discovery."""

    def __init__(self):
        self._params = {}
        self._submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Param):
            self._params[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._params.values():
            yield p
        for m in self._submodules.values():
            yield from m.parameters()

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"{self.__class__.__name__}.forward() must be implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """Tiny reference linear layer used by examples and tests."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Param.tensor((out_features, in_features))
        self.bias = Param.tensor((out_features,))
        self.weight_values = [[0.0 for _ in range(self.in_features)] for _ in range(self.out_features)]
        self.bias_values = [0.0 for _ in range(self.out_features)]

    def forward(self, x):
        vec = list(x)
        out = []
        for o in range(self.out_features):
            acc = self.bias_values[o]
            for i in range(self.in_features):
                acc += self.weight_values[o][i] * float(vec[i])
            out.append(acc)
        return out
