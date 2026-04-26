import continuum as ct
from continuum.benchmarks import MiniQADataset
from continuum.benchmarks import QABenchmarkProgram
from continuum.benchmarks import build_hotpotqa_mini
from continuum.benchmarks import em_metric
from continuum.benchmarks import evaluate
from continuum.benchmarks import split_train_test
from continuum.frontend.optimizer import Optimizer
import statistics


def make_program():
    return QABenchmarkProgram(
        instruction=ct.Param.text(
            "Use context-first extraction.",
            tokens=["question-first", "context-first", "extract"],
            trials=4,
        ),
        fewshot=ct.Param.text(
            "",
            tokens=["match country", "Q:", "A:"],
            trials=4,
        ),
        reasoning_style=ct.Param.discrete("chain", choices=["chain", "direct"]),
        temperature=ct.Param.continuous(0.8, min_value=0.0, max_value=1.0, trials=8),
    )


def test_m2_benchmark_multi_seed_improves_with_reasonable_variance():
    seeds = [11, 17, 23, 31, 47]
    baseline_scores = []
    optimized_scores = []

    for seed in seeds:
        rows = build_hotpotqa_mini(samples=200, seed=seed)
        train_rows, test_rows = split_train_test(rows, train_ratio=0.8, seed=seed)

        baseline_program = make_program()
        baseline_scores.append(evaluate(baseline_program, test_rows))

        program = make_program()
        dataset = MiniQADataset(train_rows, batch_size=20)
        opt = Optimizer(program, metric=em_metric, seed=seed)
        opt.fit(dataset, epochs=5)
        optimized_scores.append(evaluate(program, test_rows))

    baseline_mean = statistics.mean(baseline_scores)
    optimized_mean = statistics.mean(optimized_scores)
    optimized_std = statistics.pstdev(optimized_scores)

    assert optimized_mean > baseline_mean
    assert optimized_mean < 1.0
    assert optimized_std < 0.2
