"""False-hit evaluation for the semantic cache tier (issue #10).

A semantic cache that serves a wrong answer is worse than no cache. This
harness measures that risk for a given embedder over a labeled set of prompt
pairs and a sweep of similarity thresholds.

Each pair ``(a, b)`` is labeled:

- ``paraphrase``: serving ``a``'s cached answer for ``b`` is correct.
- ``near_miss``: lexically close, but ``b`` needs a different answer.
- ``unrelated``: a different topic entirely.

Two views are reported per threshold:

- **Pair level.** A pair "hits" when ``cosine(a, b) >= threshold``. Paraphrase
  hits are true positives; near-miss and unrelated hits are false hits.
- **Cache level.** Every distinct ``a`` is inserted into one real
  :class:`SemanticCacheIndex`, then every ``b`` is looked up, so a query can be
  served by *any* cached prompt. A served prompt is judged by the labels:
  correct if it is the query itself or a labeled paraphrase of it, wrong if
  the pair is labeled near-miss or unrelated, and *unjudged* otherwise
  (listed in ``unjudged`` for inspection, never silently counted).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from continuum._native import EmbeddingProvider, HitVerifier, SemanticCacheIndex

LABELS = ("paraphrase", "near_miss", "unrelated")
DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14)) + (0.97, 0.99)


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    label: str


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    true_hits: int
    missed_paraphrases: int
    false_hits: int
    near_miss_false_hits: int
    unrelated_false_hits: int
    precision: float | None
    recall: float
    false_hit_rate: float
    near_miss_false_hit_rate: float
    unrelated_false_hit_rate: float
    cache_served: int
    cache_wrong_answers: int
    cache_unjudged: int
    cache_wrong_answer_rate: float | None
    unjudged: list[list[str]]


def load_pairs(path: str | Path) -> list[Pair]:
    """Read a dataset file of the form ``{"pairs": [{"a", "b", "label"}, ...]}``."""
    raw = json.loads(Path(path).read_text())
    pairs = [Pair(p["a"], p["b"], p["label"]) for p in raw["pairs"]]
    bad = {p.label for p in pairs} - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels: {sorted(bad)}")
    return pairs


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate(
    pairs: Sequence[Pair],
    embedder: EmbeddingProvider,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    verifier: HitVerifier | None = None,
) -> dict[str, Any]:
    """Sweep ``thresholds`` and return pair- and cache-level metrics.

    With a ``verifier``, a pair only hits if it also passes verification, and
    the cache-level index runs the same verifier on its candidates (as the
    runtime does); without one, verification is disabled.
    """
    cache: dict[str, list[float]] = {}

    def vec(text: str) -> list[float]:
        if text not in cache:
            cache[text] = list(embedder.embed(text))
        return cache[text]

    sims = [SemanticCacheIndex.cosine_similarity(vec(p.a), vec(p.b)) for p in pairs]
    verified = [
        verifier.verify(p.a, p.b, s) if verifier is not None else True for p, s in zip(pairs, sims)
    ]
    counts = {label: sum(p.label == label for p in pairs) for label in LABELS}
    negatives = counts["near_miss"] + counts["unrelated"]
    anchors = list(dict.fromkeys(p.a for p in pairs))
    identity = embedder.identity()
    labels: dict[tuple[str, str], str] = {}
    for p in pairs:
        labels[(p.a, p.b)] = labels[(p.b, p.a)] = p.label

    rows: list[ThresholdResult] = []
    for t in thresholds:
        hit = [s >= t and ok for s, ok in zip(sims, verified)]
        tp = sum(h for h, p in zip(hit, pairs) if p.label == "paraphrase")
        fp_near = sum(h for h, p in zip(hit, pairs) if p.label == "near_miss")
        fp_unrel = sum(h for h, p in zip(hit, pairs) if p.label == "unrelated")
        fp = fp_near + fp_unrel

        index = SemanticCacheIndex(len(anchors), float(t))
        index.set_verifier(verifier)
        for i, anchor in enumerate(anchors):
            index.insert(
                vec(anchor), "eval", i.to_bytes(4, "little"), embedder_id=identity, prompt=anchor
            )
        served = wrong = 0
        unjudged: list[list[str]] = []
        for p in pairs:
            r = index.lookup(vec(p.b), "eval", embedder_id=identity, query_prompt=p.b)
            if not r["above_threshold"]:
                continue
            served += 1
            answered_by = anchors[int.from_bytes(r["output"], "little")]
            label = "paraphrase" if answered_by == p.b else labels.get((p.b, answered_by))
            if label is None:
                unjudged.append([p.b, answered_by])
            elif label != "paraphrase":
                wrong += 1
        judged = served - len(unjudged)

        rows.append(
            ThresholdResult(
                threshold=float(t),
                true_hits=tp,
                missed_paraphrases=counts["paraphrase"] - tp,
                false_hits=fp,
                near_miss_false_hits=fp_near,
                unrelated_false_hits=fp_unrel,
                precision=(tp / (tp + fp)) if (tp + fp) else None,
                recall=_rate(tp, counts["paraphrase"]),
                false_hit_rate=_rate(fp, negatives),
                near_miss_false_hit_rate=_rate(fp_near, counts["near_miss"]),
                unrelated_false_hit_rate=_rate(fp_unrel, counts["unrelated"]),
                cache_served=served,
                cache_wrong_answers=wrong,
                cache_unjudged=len(unjudged),
                cache_wrong_answer_rate=(wrong / judged) if judged else None,
                unjudged=unjudged,
            )
        )

    by_label = {label: [s for s, p in zip(sims, pairs) if p.label == label] for label in LABELS}
    return {
        "embedder": identity,
        "verifier": verifier.name() if verifier is not None else None,
        "pairs": counts,
        "similarity": {
            label: {
                "min": round(min(v), 4),
                "mean": round(sum(v) / len(v), 4),
                "max": round(max(v), 4),
            }
            for label, v in by_label.items()
            if v
        },
        "thresholds": [asdict(r) for r in rows],
    }


def safest_threshold(result: dict[str, Any], max_false_hit_rate: float = 0.0) -> float | None:
    """Lowest threshold whose pair-level false-hit rate is within the budget."""
    for row in result["thresholds"]:
        if row["false_hit_rate"] <= max_false_hit_rate:
            return float(row["threshold"])
    return None


def summarize(result: dict[str, Any], default_threshold: float = 0.85) -> dict[str, Any]:
    """Headline numbers: the lowest zero-false-hit threshold and its recall,
    plus recall and false-hit rate at ``default_threshold``."""
    zero = safest_threshold(result, 0.0)
    row0 = next((r for r in result["thresholds"] if r["threshold"] == zero), None)
    rowd = min(result["thresholds"], key=lambda r: abs(r["threshold"] - default_threshold))
    return {
        "zero_false_hit_threshold": zero,
        "recall_at_zero_false_hits": row0["recall"] if row0 else None,
        "default_threshold": rowd["threshold"],
        "recall_at_default": rowd["recall"],
        "false_hit_rate_at_default": rowd["false_hit_rate"],
        "near_miss_false_hit_rate_at_default": rowd["near_miss_false_hit_rate"],
    }
