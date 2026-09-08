"""10 - Tune a prompt program's text and sampling params against data.

Continuum's Python frontend treats a prompt program like a model: its
instruction wording and its numeric knobs are :class:`Param` values, and
:class:`Optimizer` searches them against a metric on a training split
(DSPy / TextGrad style).

This is a self-contained, dependency-free demo. The "LLM" is a deterministic
lexical scorer so the run is reproducible; the point is the
``Param`` / ``parameters()`` / ``Optimizer.fit`` API, not the toy classifier.
``examples/milestones/qa_benchmark_validation.py`` runs the same machinery over
a HotpotQA-style set with five seeds.

    PYTHONPATH=python python examples/10_prompt_optimization.py
"""

from __future__ import annotations

import continuum as ct

BAR = "=" * 64

POS = {"great", "love", "excellent", "good", "happy", "amazing", "best", "nice"}
NEG = {"bad", "hate", "terrible", "awful", "poor", "worst", "sad", "broken"}

# id, text, hidden label. Label rule: positivity >= 0.55 -> "positive".
# The 1-pos/1-neg rows sit at positivity 0.5, so a 0.50 threshold misreads them.
ROWS = [
    (0, "great and excellent and love it", "positive"),
    (1, "good nice happy amazing", "positive"),
    (2, "terrible awful and broken", "negative"),
    (3, "bad poor worst experience", "negative"),
    (4, "good but bad", "negative"),
    (5, "love it yet broken", "negative"),
    (6, "nice though poor", "negative"),
    (7, "happy but awful", "negative"),
    (8, "great great great bad", "positive"),
    (9, "excellent good nice terrible", "positive"),
    (10, "awful awful good", "negative"),
    (11, "best love happy good", "positive"),
    (12, "worst sad broken bad", "negative"),
    (13, "amazing good but poor", "positive"),
    (14, "hate broken awful sad", "negative"),
    (15, "good good good good bad", "positive"),
]


def positivity(text: str) -> float:
    words = text.lower().split()
    p = sum(w in POS for w in words)
    n = sum(w in NEG for w in words)
    return 0.5 if p + n == 0 else p / (p + n)


class SentimentProgram(ct.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Text param: starts without a format-carrying token.
        self.instruction = ct.Param.text(
            "classify the review",
            tokens=["label", "sentiment", "only", "positive", "negative"],
            trials=6,
        )
        # Continuous param: the decision threshold, tuned by random search.
        self.threshold = ct.Param.continuous(0.50, min_value=0.0, max_value=1.0, trials=16)

    def forward(self, row: dict) -> str:
        score = positivity(row["text"])
        pred = "positive" if score >= float(self.threshold.value) else "negative"
        instr = str(self.instruction.value).lower()
        # Without a format cue the "model" gets confused on some rows.
        if "label" not in instr and "sentiment" not in instr and row["id"] % 3 == 0:
            pred = "negative" if pred == "positive" else "positive"
        return pred


def accuracy(pred: str, row: dict) -> float:
    return 1.0 if pred == row["label"] else 0.0


class Dataset:
    def __init__(self, rows: list[dict], batch_size: int) -> None:
        self.rows = rows
        self.batch_size = batch_size

    def batches(self):
        for i in range(0, len(self.rows), self.batch_size):
            yield self.rows[i : i + self.batch_size]


def evaluate(program: SentimentProgram, rows: list[dict]) -> float:
    return sum(accuracy(program(r), r) for r in rows) / len(rows)


def main() -> None:
    print(BAR)
    print(" Continuum - Prompt Program Optimization")
    print(BAR)

    rows = [{"id": i, "text": t, "label": y} for (i, t, y) in ROWS]
    train, test = rows[:12], rows[12:]

    program = SentimentProgram()
    print(f"start : instruction={program.instruction.value!r}  "
          f"threshold={program.threshold.value:.2f}")
    print(f"        train acc={evaluate(program, train):.2f}  test acc={evaluate(program, test):.2f}")
    print("-" * 64)

    opt = ct.Optimizer(program, metric=accuracy, lr_text=1.0, seed=7)
    opt.fit(Dataset(train, batch_size=6), epochs=6)

    print(f"tuned : instruction={program.instruction.value!r}  "
          f"threshold={program.threshold.value:.2f}")
    print(f"        train acc={evaluate(program, train):.2f}  test acc={evaluate(program, test):.2f}")
    print("-" * 64)
    print(f"params optimized: {[p.kind for p in program.parameters()]}")
    print(BAR)
    print(" prompt optimization: OK")
    print(BAR)


if __name__ == "__main__":
    main()
