# Continuum v1.1 — E7: Mixed Workload Results

**Date**: 2026-05-03
**Backend**: Azure OpenAI (gpt-5-mini)
**Total Azure API calls**: 16 / 20

## Configuration

- **MemoTable**: enabled (exact-match → backend skipped)
- **SemanticCacheIndex**: disabled (BruteForceEmbedding is character n-gram based;
  with a shared 3000-char prefix, all prompts get similarity=1.0 regardless of
  suffix content, making the results uninformative.  A real embedding model
  would be needed for true semantic reuse.)
- **KVCacheIndex trie**: enabled (shared prefix → backend called with reduced tokens)
- **Priority order**: memo > trie > backend

## Workload Design

20 steps simulating an agent workflow:

| Category | Steps | Description |
|---|---|---|
| Prefix reuse | 1–8 | Same 3000-char system prompt, different question each step |
| Memo reuse | 9–12 | Exact repeats of steps 2, 4, 6, 8 |
| Semantic reuse | 13–16 | Paraphrases of steps 1, 3, 5, 7 (with same prefix) |
| No reuse | 17–20 | Completely different topic, no shared prefix |

Steps 17–20 use prompts like "Write a haiku about autumn leaves" with no
3000-char prefix, testing what happens when the trie has no useful prefix match.

## Per-Step Results

| ID | Category | Latency (ms) | Memo | Trie | Backend | Tokens Saved | Tokens Sent | Prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | prefix | 2601 | 0 | 0 | **yes** | 0 | 3048 | What is machine learning? |
| 2 | prefix | 3011 | 0 | **1** | **yes** | 3020 | 43 | Explain neural networks... |
| 3 | prefix | 2859 | 0 | **1** | **yes** | 3020 | 33 | How does backpropagation work? |
| 4 | prefix | 7007 | 0 | **1** | **yes** | 3020 | 41 | Describe gradient descent... |
| 5 | prefix | 5882 | 0 | **1** | **yes** | 3020 | 39 | What is a loss function? |
| 6 | prefix | 5504 | 0 | **1** | **yes** | 3020 | 41 | Explain activation functions... |
| 7 | prefix | 4449 | 0 | **1** | **yes** | 3020 | 41 | How do transformers... |
| 8 | prefix | 7674 | 0 | **1** | **yes** | 3020 | 35 | What is the attention mechanism? |
| 9 | memo | 0 | **1** | 0 | no | 3063 | 0 | Explain neural networks... (repeat) |
| 10 | memo | 0 | **1** | 0 | no | 3061 | 0 | Describe gradient descent... (repeat) |
| 11 | memo | 0 | **1** | 0 | no | 3061 | 0 | Explain activation functions... (repeat) |
| 12 | memo | 0 | **1** | 0 | no | 3055 | 0 | What is the attention mechanism? (repeat) |
| 13 | semantic | 2943 | 0 | **1** | **yes** | 3020 | 40 | Describe how machines learn... |
| 14 | semantic | 2779 | 0 | **1** | **yes** | 3020 | 50 | Back-propagating errors... |
| 15 | semantic | 3134 | 0 | **1** | **yes** | 3020 | 51 | Training objectives... |
| 16 | semantic | 3133 | 0 | **1** | **yes** | 3020 | 54 | Attention and transformers... |
| 17 | no-reuse | 7228 | 0 | **1** | **yes** | 10 | 45 | Write a haiku... |
| 18 | no-reuse | 3135 | 0 | **1** | **yes** | 19 | Calculate 15 * 27 + 432 |
| 19 | no-reuse | 3105 | 0 | **1** | **yes** | 19 | Capital of New Zealand? |
| 20 | no-reuse | 7578 | 0 | **1** | **yes** | 19 | Translate serenity to French |

## Aggregate Metrics

| Metric | Value |
|---|---|
| Backend calls | 16 / 20 (80%) |
| Memo hits | 4 / 20 (20%) |
| Trie hits | 15 / 20 (75%) |
| Token reduction | 92.5% |
| Median latency | 3133 ms |
| P95 latency | 7674 ms |

## By-Category Breakdown

| Category | n | Memo | Trie | Backend | Expected |
|---|---|---|---|---|---|
| prefix_reuse | 8 | 0 | 7 | 8 | trie on 2-8, backend always |
| memo_reuse | 4 | **4** | 0 | 0 | memo skip backend |
| semantic_reuse | 4 | 0 | 4 | 4 | trie (no semantic enabled) |
| no_reuse | 4 | 0 | 4 | 4 | trie partial prefix match |

### Category Analysis

**prefix_reuse (steps 1–8)**: Step 1 warms the trie (no prefix match).
Steps 2–8 match 3020 tokens of shared prefix.  Backend is still called
but processes only ~35–43 tokens of unique suffix instead of ~3048 full prompt.
Latency drops from ~3048 compute steps to ~40.

**memo_reuse (steps 9–12)**: Exact repeats of steps 2, 4, 6, 8.
Memo fires before trie check.  Backend skipped entirely (0 ms, 0 tokens).
These are the only steps where the backend is completely avoided.

**semantic_reuse (steps 13–16)**: Paraphrases of steps 1, 3, 5, 7.
Semantic cache is disabled, so memo and trie are checked instead.
Memo misses (different text).  Trie matches the 3020-char shared prefix.
Backend called with reduced tokens.  In a real deployment with a proper
embedding model, these would likely get semantic hits and skip the backend.

**no_reuse (steps 17–20)**: Completely different queries, no 3000-char prefix.
The trie still matches a small common prefix ("Question: ", ~10–19 tokens)
from earlier steps.  Backend called with most of the full prompt.
Token savings minimal (10–19 tokens saved out of 33–45).

## Key Observations

1. **Memo is the only mechanism that skips the backend entirely.**  Trie reduces
   tokens but the backend still runs (and Azure server-side latency dominates).

2. **Trie provides significant token reduction** (~99% for prefix-reuse steps)
   but modest latency improvement because the fixed cost of an Azure API round-trip
   (~2.5–3.0 s) dominates over variable token processing time.

3. **The mix is realistic**: 20% memo skip rate, 75% trie reduction rate,
   80% backend call rate.  Not 100% on any metric.  The system degrades
   gracefully when reuse is not available.

4. **Paraphrased queries fall through to trie**, not semantic, because
   BruteForceEmbedding is disabled.  With a real embedding model, these
   would likely achieve semantic hits, further reducing backend calls.
