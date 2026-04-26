import random


class TensorOpt:
    """Local optimizer for numeric tensor-like scalar parameters."""

    def __init__(self, lr: float, rng: random.Random):
        self.lr = float(lr)
        self.rng = rng

    def update(self, p, evaluate):
        if not isinstance(p.value, (int, float)):
            return False
        current = float(p.value)
        best = evaluate()
        candidates = [current + self.lr, current - self.lr]
        for c in candidates:
            p.value = c
            score = evaluate()
            if score > best:
                best = score
                current = c
        p.value = current
        return True


class TextGradOpt:
    """Text mutation optimizer used for prompt-style parameters."""

    def __init__(self, lr: float, rng: random.Random):
        self.lr = float(lr)
        self.rng = rng

    def _mutate(self, text, tokens):
        words = text.split()
        if not words:
            words = [self.rng.choice(tokens)]
            return " ".join(words)
        do_add = self.rng.random() < 0.5 or len(words) == 1
        if do_add:
            idx = self.rng.randrange(0, len(words) + 1)
            tok = self.rng.choice(tokens)
            words.insert(idx, tok)
        else:
            idx = self.rng.randrange(0, len(words))
            words.pop(idx)
        return " ".join(words).strip()

    def update(self, p, evaluate):
        current = str(p.value)
        best_text = current
        best_score = evaluate()
        token_pool = p.metadata.get("tokens")
        if not token_pool:
            token_pool = [w for w in current.split() if w]
        if not token_pool:
            token_pool = ["good", "better", "best"]
        trials = max(1, int(p.metadata.get("trials", max(1, self.lr * 4))))
        for _ in range(trials):
            candidate = self._mutate(best_text, token_pool)
            p.value = candidate
            score = evaluate()
            if score > best_score:
                best_score = score
                best_text = candidate
        p.value = best_text
        return True


class GEPAOpt:
    """Discrete choice optimizer for categorical parameters."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def update(self, p, evaluate):
        choices = p.metadata.get("choices")
        if not choices:
            return False
        best_value = p.value
        best_score = evaluate()
        for candidate in choices:
            p.value = candidate
            score = evaluate()
            if score > best_score:
                best_score = score
                best_value = candidate
        p.value = best_value
        return True


class BayesOpt:
    """Simple random-search optimizer over bounded continuous values."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def update(self, p, evaluate):
        if not isinstance(p.value, (int, float)):
            return False
        current = float(p.value)
        low = float(p.metadata.get("min", current - 1.0))
        high = float(p.metadata.get("max", current + 1.0))
        trials = int(p.metadata.get("trials", 8))
        best_value = current
        best_score = evaluate()
        for _ in range(max(1, trials)):
            candidate = self.rng.uniform(low, high)
            p.value = candidate
            score = evaluate()
            if score > best_score:
                best_score = score
                best_value = candidate
        p.value = best_value
        return True


class Optimizer:
    """Unified optimizer facade for all Continuum parameter kinds."""

    def __init__(self, program, metric, *, lr_tensor=1e-1, lr_text=1.0, seed=0):
        self.program = program
        self.metric = metric
        self.rng = random.Random(seed)
        self._tensor = TensorOpt(lr=lr_tensor, rng=self.rng)
        self._text = TextGradOpt(lr=lr_text, rng=self.rng)
        self._discrete = GEPAOpt(rng=self.rng)
        self._continuous = BayesOpt(rng=self.rng)

    def _score(self, batch):
        scores = [self.metric(self.program(x), x) for x in batch]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def step(self, batch):
        params = list(self.program.parameters()) if hasattr(self.program, "parameters") else []
        for p in params:
            evaluate = lambda: self._score(batch)
            if p.kind == "tensor":
                self._tensor.update(p, evaluate)
            elif p.kind == "text":
                self._text.update(p, evaluate)
            elif p.kind == "discrete":
                self._discrete.update(p, evaluate)
            elif p.kind == "continuous":
                self._continuous.update(p, evaluate)

    def fit(self, dataset, epochs=1):
        for _ in range(epochs):
            for batch in dataset.batches():
                self.step(batch)
