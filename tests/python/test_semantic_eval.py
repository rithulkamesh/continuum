"""Metric arithmetic of the semantic false-hit harness (issue #10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum.benchmarks.semantic_eval import Pair, evaluate, load_pairs, safest_threshold
from continuum.embeddings import PrecomputedEmbeddingProvider

ROOT = Path(__file__).resolve().parents[2]

VECS = {
    "reset password": [1.0, 0.0, 0.0],
    "forgot password": [0.96, 0.28, 0.0],  # cos 0.96 with "reset password"
    "reset username": [0.8, 0.6, 0.0],  # cos 0.80: a near miss
    "weather today": [0.0, 0.0, 1.0],  # cos 0.0: unrelated
}
PAIRS = [
    Pair("reset password", "forgot password", "paraphrase"),
    Pair("reset password", "reset username", "near_miss"),
    Pair("reset password", "weather today", "unrelated"),
]


def _row(result: dict, threshold: float) -> dict:
    return next(r for r in result["thresholds"] if r["threshold"] == threshold)


def test_metrics_per_threshold() -> None:
    emb = PrecomputedEmbeddingProvider(VECS, "test")
    result = evaluate(PAIRS, emb, thresholds=[0.5, 0.9, 0.99])
    assert result["embedder"] == "test"
    assert result["pairs"] == {"paraphrase": 1, "near_miss": 1, "unrelated": 1}

    low = _row(result, 0.5)
    assert (low["true_hits"], low["false_hits"]) == (1, 1)
    assert low["near_miss_false_hit_rate"] == 1.0
    assert low["unrelated_false_hit_rate"] == 0.0
    assert low["precision"] == 0.5
    # Cache level: the paraphrase and the near miss are both answered by the
    # single anchor; only the paraphrase answer is right.
    assert (low["cache_served"], low["cache_wrong_answers"], low["cache_unjudged"]) == (2, 1, 0)

    mid = _row(result, 0.9)
    assert (mid["true_hits"], mid["false_hits"], mid["recall"]) == (1, 0, 1.0)
    assert mid["cache_wrong_answer_rate"] == 0.0

    high = _row(result, 0.99)
    assert high["recall"] == 0.0 and high["precision"] is None
    assert high["cache_wrong_answer_rate"] is None

    assert safest_threshold(result) == 0.9
    assert safest_threshold(evaluate(PAIRS, emb, thresholds=[0.5])) is None


def test_bundled_dataset_is_balanced_and_loads() -> None:
    pairs = load_pairs(ROOT / "benchmarks" / "data" / "semantic_pairs.json")
    labels = [p.label for p in pairs]
    assert {labels.count(x) for x in ("paraphrase", "near_miss", "unrelated")} == {25}


def test_unknown_label_rejected(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"pairs": [{"a": "x", "b": "y", "label": "maybe"}]}))
    with pytest.raises(ValueError, match="unknown labels"):
        load_pairs(f)


def test_verifier_and_unjudged_serves() -> None:
    from continuum.benchmarks.semantic_eval import summarize
    from continuum.verifiers import LexicalNearMissVerifier

    vecs = {**VECS, "I lost my password and must reset it": [0.99, 0.14, 0.0]}
    pairs = [*PAIRS, Pair("weather today", "I lost my password and must reset it", "unrelated")]
    emb = PrecomputedEmbeddingProvider(vecs, "test")
    plain = evaluate(pairs, emb, thresholds=[0.5])
    checked = evaluate(pairs, emb, thresholds=[0.5], verifier=LexicalNearMissVerifier())
    assert checked["verifier"] == "lexical-near-miss-v1" and plain["verifier"] is None
    # "reset username" is a one-word edit of "reset password": refused.
    assert plain["thresholds"][0]["near_miss_false_hit_rate"] == 1.0
    assert checked["thresholds"][0]["near_miss_false_hit_rate"] == 0.0
    row = checked["thresholds"][0]
    # The rewording is served by "reset password", a pair with no label.
    assert row["cache_unjudged"] == 1 and row["unjudged"] == [
        ["I lost my password and must reset it", "reset password"]
    ]
    s = summarize(checked, default_threshold=0.5)
    # "forgot password" is a one-word swap of "reset password": the lexical
    # verifier refuses synonym swaps too, so this paraphrase is lost.
    assert s["zero_false_hit_threshold"] == 0.5 and s["recall_at_zero_false_hits"] == 0.0
    assert summarize(plain, default_threshold=0.5)["zero_false_hit_threshold"] is None


def test_validation_and_test_sets_are_balanced() -> None:
    for name in ("semantic_pairs_validation.json", "semantic_pairs_test.json"):
        labels = [p.label for p in load_pairs(ROOT / "benchmarks" / "data" / name)]
        assert {labels.count(x) for x in ("paraphrase", "near_miss", "unrelated")} == {20}
