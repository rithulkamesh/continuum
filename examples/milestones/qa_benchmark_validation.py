import continuum as ct
from continuum.benchmarks import MiniQADataset
from continuum.benchmarks import QABenchmarkProgram
from continuum.benchmarks import build_hotpotqa_mini
from continuum.benchmarks import em_metric
from continuum.benchmarks import evaluate
from continuum.benchmarks import split_train_test
from continuum.frontend.optimizer import Optimizer
import statistics


def main():
    seeds = [11, 17, 23, 31, 47]
    baseline_test_scores = []
    optimized_test_scores = []
    train_scores = []
    test_scores = []

    for seed in seeds:
        rows = build_hotpotqa_mini(samples=200, seed=seed)
        train_rows, test_rows = split_train_test(rows, train_ratio=0.8, seed=seed)
        dataset = MiniQADataset(train_rows, batch_size=20)

        baseline_program = QABenchmarkProgram(
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
        baseline_test = evaluate(baseline_program, test_rows)

        program = QABenchmarkProgram(
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

        optimizer = Optimizer(program, metric=em_metric, lr_tensor=0.1, lr_text=1.0, seed=seed)
        optimizer.fit(dataset, epochs=5)
        optimized_train = evaluate(program, train_rows)
        optimized_test = evaluate(program, test_rows)

        baseline_test_scores.append(baseline_test)
        optimized_test_scores.append(optimized_test)
        train_scores.append(optimized_train)
        test_scores.append(optimized_test)

    baseline_test_mean = statistics.mean(baseline_test_scores)
    optimized_test_mean = statistics.mean(optimized_test_scores)
    baseline_test_std = statistics.pstdev(baseline_test_scores)
    optimized_test_std = statistics.pstdev(optimized_test_scores)
    train_mean = statistics.mean(train_scores)
    test_mean = statistics.mean(test_scores)
    improvement = (
        ((optimized_test_mean - baseline_test_mean) / baseline_test_mean * 100.0)
        if baseline_test_mean > 0.0
        else optimized_test_mean * 100.0
    )

    print("M2 BENCHMARK VALIDATION")
    print("seeds:", seeds)
    print("dataset_size_per_seed: 200")
    print("train_test_split: 80/20")
    print(f"baseline_test_mean: {baseline_test_mean:.4f}")
    print(f"baseline_test_std: {baseline_test_std:.4f}")
    print(f"optimized_test_mean: {optimized_test_mean:.4f}")
    print(f"optimized_test_std: {optimized_test_std:.4f}")
    print(f"optimized_train_mean: {train_mean:.4f}")
    print(f"optimized_test_mean_repeat: {test_mean:.4f}")
    print(f"improvement_percent: {improvement:.2f}%")


if __name__ == "__main__":
    main()
