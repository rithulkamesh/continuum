from continuum.nn.module import Module
from continuum.nn.module import Linear


class TinyTransformer(Module):
    def __init__(self):
        super().__init__()
        self.proj_in = Linear(4, 8)
        self.proj_out = Linear(8, 2)

    def forward(self, x):
        hidden = self.proj_in(x)
        return self.proj_out(hidden)


if __name__ == "__main__":
    model = TinyTransformer()
    y = model([0.1, 0.2, 0.3, 0.4])
    print("transformer output", y)
