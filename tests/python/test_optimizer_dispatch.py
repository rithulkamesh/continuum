from continuum.frontend.optimizer import Optimizer
from continuum.frontend.param import Param


class Program:
    def __init__(self):
        self.tensor = Param(kind="tensor", value=0.0, metadata={"lr": 0.4})
        self.text = Param.text("bad", tokens=["bad", "good"])
        self.discrete = Param.discrete(0, choices=[0, 1, 2])
        self.continuous = Param.continuous(0.0, min_value=0.0, max_value=2.0)
        self._params = [self.tensor, self.text, self.discrete, self.continuous]

    def parameters(self):
        return self._params

    def __call__(self, x):
        text_bonus = 2.0 if "good" in self.text.value.split() else 0.0
        discrete_bonus = float(self.discrete.value)
        return (
            x
            + float(self.tensor.value)
            + text_bonus
            + discrete_bonus
            + float(self.continuous.value)
        )


class Dataset:
    def batches(self):
        for _ in range(6):
            yield [0.0, 1.0]


def metric(output, x):
    target = x + 5.0
    return -((output - target) ** 2)


def mean_score(program, batch):
    scores = [metric(program(x), x) for x in batch]
    return sum(scores) / len(scores)


def test_optimizer_updates_all_param_kinds_and_improves_metric():
    program = Program()
    batch = [0.0, 1.0]
    before = mean_score(program, batch)
    before_tensor = program.tensor.value
    before_text = program.text.value
    before_discrete = program.discrete.value
    before_continuous = program.continuous.value

    opt = Optimizer(program, metric=metric, seed=7)
    opt.fit(Dataset(), epochs=1)

    after = mean_score(program, batch)
    assert program.tensor.value != before_tensor
    assert program.text.value != before_text
    assert program.discrete.value != before_discrete
    assert program.continuous.value != before_continuous
    assert after > before
