# E9 — Semantic Cache False Hits: Embedders and Hit Verification

**Date**: 2026-09-25
**Runner**: [`benchmarks/scripts/e9_semantic_false_hits.py`](../scripts/e9_semantic_false_hits.py),
method in `continuum.benchmarks.semantic_eval`
**Raw data**: [`benchmarks/data/e9_semantic_false_hits.json`](../data/e9_semantic_false_hits.json)

## Why

The semantic tier serves a cached answer when a new prompt is "similar
enough" to an old one. When that call is wrong, the user gets a confident
answer to a different question, which is worse than a cache miss. This report
measures the risk and the fix.

## Datasets

Hand-labeled prompt pairs `(a, b)`. The question for each pair: is it correct
to answer `b` with `a`'s cached answer?

| Label | Meaning | Example |
|---|---|---|
| paraphrase | same question, reworded (a hit is correct) | "how do I reset my password" / "I forgot my password, how can I reset it" |
| near_miss | minimal edit, different answer (a hit is a false hit) | "how do I **enable** two-factor authentication" / "how do I **disable** …" |
| unrelated | different topic (a hit is a false hit) | "where is my order" / "explain quantum entanglement simply" |

| Set | File | Pairs | Role |
|---|---|---|---|
| dev | `semantic_pairs.json` | 75 | used to design the verifier |
| validation | `semantic_pairs_validation.json` | 60 | written before tuning, then used once to diagnose it |
| **test** | `semantic_pairs_test.json` | 60 | written before the final rule was fixed; **never used for tuning**. Headline numbers. |

## Method

- **Pair level.** A pair hits when `cosine(embed(a), embed(b)) >= threshold`
  (and, with a verifier, the verifier accepts). Recall = paraphrase hits / 20;
  false-hit rate (FHR) = near-miss + unrelated hits / 40.
- **Cache level.** Every distinct `a` goes into one `SemanticCacheIndex`,
  every `b` is looked up, and any cached prompt may answer. A served answer
  counts as correct if it is the query itself or a labeled paraphrase, and as
  wrong if the pair is labeled near-miss or unrelated. Pairs with no label
  are reported as *unjudged* and listed in the raw data.

## What the research says

Semantic similarity is not "has the same answer". An off-the-shelf
embedder reached an AUC of 0.51, chance level, on a hard caching dataset
until it was fine-tuned for the task (Zhu et al., *Efficient Prompt Caching
via Embedding Similarity*, 2024). Minimal lexical edits reliably trigger
wrong hits in GPTCache-style caches (Afiffy et al., *SAFE-CACHE*, Sci. Rep.
2026). The recommended mitigations are a second, verification stage on
candidate hits, such as an LLM judge (Singh et al., *Krites*, EuroMLSys 2026)
or learned refinement, and better or ensembled embedders (Ghaffari et al.,
2025; Gill et al., *MeanCache*, IPDPS 2025).

Continuum now does both. It adds a real semantic embedder that runs locally,
and a **hit verifier** on the semantic tier, on by default:

- `LexicalNearMissVerifier` (default) rejects minimal edits: different
  numbers, a swapped content word with everything else unchanged, flipped
  polarity ("to"/"from", "on"/"off", "with"/"without"), negation, and a
  swapped direction ("miles to km" vs "km to miles"). Rewordings pass.
- `LLMJudgeVerifier` asks a chat model (e.g. Ollama) whether both prompts
  have the same answer. It runs only on candidates above the threshold, and
  verdicts are cached.

## Results (test set, never used for tuning)

| Embedder | Verifier | Threshold | Recall | False-hit rate | Near-miss FHR | Full-cache serves: wrong / unjudged |
|---|---|---|---|---|---|---|
| char-n-gram (built-in) | none | 0.85 | 0.70 | 0.500 | 1.00 | 19 / 24 of 54 |
| char-n-gram (built-in) | lexical | 0.85 | 0.50 | 0.000 | 0.00 | 0 / **37 of 44** (unrelated questions served) |
| WordLlama | none | 0.70 | 0.70 | 0.275 | 0.55 | 11 / 0 of 25 |
| **WordLlama** | **lexical** | **0.70** | **0.45** | **0.000** | **0.00** | **0 / 0 of 9** |
| WordLlama | lexical | 0.60 | 0.70 | 0.000 | 0.00 | 0 / 0 |

Zero-false-hit frontier (the lowest threshold with no pair-level false hits,
and the recall there):

| Embedder | Verifier | dev | validation | test |
|---|---|---|---|---|
| char-n-gram | none | never | 0.99 (recall 0.00) | never |
| char-n-gram | lexical | 0.85 (0.56) | 0.90 (0.35) | 0.85 (0.50) |
| WordLlama | none | never | never | never |
| WordLlama | lexical | 0.30 (0.84) | 0.30 (0.75) | 0.30 (0.75) |

## Findings

1. **No embedder alone is safe.** Without verification, neither embedder
   reaches zero false hits at any threshold on any set. Near-misses are the
   cause: with WordLlama, "enable"/"disable" two-factor still scores 0.87,
   because embedders encode topic, not the answer.
2. **The verifier removes the near-miss failure mode.** On the test set,
   near-miss false hits drop from 55% (WordLlama) or 100% (n-gram) to 0.
3. **The built-in n-gram embedder is not usable for semantic caching.** It
   rates unrelated English questions 0.62–0.85 similar, so with every prompt
   cached it serves unrelated answers (37 unjudged serves on the test set,
   all visibly wrong, e.g. "how fast does sound travel" served "what does a
   product manager do"). The verifier cannot fix that: those pairs are not
   minimal edits. Use a semantic embedder.
4. **WordLlama + lexical verifier is the recommended default.** It runs
   locally, has no network dependency, and embeds in ~1 ms. At **0.7** it
   served 9 test answers, none wrong. 0.7 was chosen on dev + validation: at
   lower thresholds, dev showed topically-related-but-different questions
   being served ("which ways can I pay" → "can I pay in installments").
5. **What the lexical verifier costs.** It cannot tell a synonym swap from an
   antonym swap. Word-level similarity doesn't separate them either:
   "cancel"/"stop" 0.35 vs "enable"/"disable" 0.34. So it also turns down
   paraphrases that differ by exactly one synonym ("cancel" vs "end my gym
   membership"). Those become cache misses, never wrong answers.
6. **Residual risk, and where the LLM judge fits.** Related-but-different
   questions that are not minimal edits (dev at 0.7: "sort a list
   descending" served by "sort a list"; "shipping cost" served by
   "shipping time") are beyond a lexical rule. The `LLMJudgeVerifier`
   targets exactly these and the synonym case. It was not measured here
   because this environment had no model server. Run the command below
   against Ollama to add it.

## Recommendations

```python
from continuum.embeddings import WordLlamaEmbeddingProvider   # pip install "continuum-ai[semantic]"
from continuum._native import SemanticCacheIndex

sc = SemanticCacheIndex(2048, 0.7)                  # LexicalNearMissVerifier is on by default
session.set_semantic_cache(sc)
session.set_embedding_provider(WordLlamaEmbeddingProvider())

# Higher stakes: judge candidates with a local model instead
from continuum.verifiers import LLMJudgeVerifier
sc.set_verifier(LLMJudgeVerifier("http://localhost:11434", "gemma4"))
```

Re-run on your traffic, and add an Ollama embedder and judge:

```bash
PYTHONPATH=python python benchmarks/scripts/e9_semantic_false_hits.py \
    --base-url http://localhost:11434 --embed-model nomic-embed-text --judge-model gemma4
```
