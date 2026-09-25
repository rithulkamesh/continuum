# E9 — Semantic Cache False-Hit Rate

**Date**: 2026-09-25
**Embedder**: `continuum/char-ngram-v1:64` (the bundled `BruteForceEmbeddingProvider`)
**Dataset**: [`benchmarks/data/semantic_pairs.json`](../data/semantic_pairs.json), 75 hand-labeled pairs
**Runner**: [`benchmarks/scripts/e9_semantic_false_hits.py`](../scripts/e9_semantic_false_hits.py),
method in `continuum.benchmarks.semantic_eval`
**Raw data**: [`benchmarks/data/e9_semantic_false_hits.json`](../data/e9_semantic_false_hits.json)

## Why

The semantic tier returns a cached answer when a new prompt is similar enough
to an old one. When "similar enough" is wrong, the user gets a confident answer
to a different question. That is worse than a cache miss, so the false-hit rate
has to be measured before the tier is trusted.

## Dataset

Each pair `(a, b)` asks: is it correct to answer `b` with `a`'s cached answer?

| Label | Count | Meaning | Example |
|---|---|---|---|
| paraphrase | 25 | same question, different wording (a hit is correct) | "how do I reset my password" / "I forgot my password, how can I reset it" |
| near_miss | 25 | lexically close, different answer (a hit is a false hit) | "how do I enable two-factor authentication" / "how do I **disable** two-factor authentication" |
| unrelated | 25 | different topic (a hit is a false hit) | "where is my order" / "explain quantum entanglement simply" |

## Method

- **Pair level**: a pair hits when `cosine(embed(a), embed(b)) >= threshold`.
  Precision = paraphrase hits / all hits; recall = paraphrase hits / 25;
  false-hit rate = (near-miss + unrelated hits) / 50.
- **Cache level**: all distinct `a` prompts go into one real
  `SemanticCacheIndex`, then every `b` is looked up. A served answer is wrong
  unless the pair is a paraphrase *and* the index returned that pair's own `a`.
  This is what a user of the cache would experience.

## Results (bundled n-gram embedder)

Similarity by label:

| Label | min | mean | max |
|---|---|---|---|
| paraphrase | 0.706 | 0.863 | 0.944 |
| near_miss | 0.866 | **0.954** | 0.997 |
| unrelated | 0.630 | 0.765 | 0.845 |

| Threshold | Precision | Recall | False-hit rate | Near-miss FHR | Unrelated FHR | Cache wrong-answer rate |
|---|---|---|---|---|---|---|
| 0.70 | 0.357 | 1.000 | 0.900 | 1.000 | 0.800 | 0.827 |
| 0.75 | 0.348 | 0.920 | 0.860 | 1.000 | 0.720 | 0.827 |
| 0.80 | 0.389 | 0.840 | 0.660 | 1.000 | 0.320 | 0.824 |
| **0.85 (runtime default)** | 0.390 | 0.640 | 0.500 | **1.000** | 0.000 | 0.831 |
| 0.90 | 0.273 | 0.360 | 0.480 | 0.960 | 0.000 | 0.804 |
| 0.95 | 0.000 | 0.000 | 0.260 | 0.520 | 0.000 | 1.000 |
| 0.99 | 0.000 | 0.000 | 0.060 | 0.120 | 0.000 | 1.000 |

(0.50–0.65 and 0.97 are in the raw data.) **No threshold reaches zero false
hits.**

## Findings

1. **The bundled embedder is unsafe for the semantic tier.** Near-misses score
   *higher* than true paraphrases (mean 0.954 vs 0.863), because a near-miss
   shares almost all of its characters with the original ("enable" vs
   "disable"), while a paraphrase rewords it. No threshold separates them: at
   the 0.85 default every near-miss is served the wrong answer, and about 83%
   of all answers the cache serves are wrong.
2. **Unrelated prompts are the easy case.** At >= 0.85 none of them hit.
   Most of the risk is near-misses, which a character-level embedder cannot
   tell apart by construction.
3. The earlier guidance in `docs/benchmarks.md`, that the n-gram provider is
   a placeholder and semantic numbers need a real model, is confirmed and
   now quantified.

## Recommendations

- Do not attach `SemanticCacheIndex` with `BruteForceEmbeddingProvider` in
  production. Use a semantic embedder via `continuum.embeddings` (a local
  sentence-transformer, or an OpenAI-compatible `/v1/embeddings` endpoint).
- Re-run this harness with the embedder you deploy, and pick the lowest
  threshold whose false-hit rate fits your budget
  (`safest_threshold(result, max_false_hit_rate)`):

  ```bash
  PYTHONPATH=python python benchmarks/scripts/e9_semantic_false_hits.py \
      --embed-base-url http://localhost:11434 --embed-model nomic-embed-text
  ```

- Near-miss pairs are the ones to extend with your own traffic: negations,
  swapped entities, different numbers.

Real-embedder numbers are not included here: the environment that produced
this report could not download model weights. The harness, dataset, and
report layout are ready for them.
