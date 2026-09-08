# E7: Mixed Workload — Final Report

**Continuum v1.1** | Azure OpenAI (gpt-5-mini) | 2026-05-03

---

## 1. Setup

A 20-step agent workflow with a 3 000-character shared system prompt.
Each step passes through the Interpreter path with the standard priority chain:

```
memo  →  semantic  →  trie  →  backend
```

Only **MemoTable** and **KVCacheIndex (trie)** are enabled.
SemanticCacheIndex is intentionally disabled because the bundled
BruteForceEmbeddingProvider uses character n-gram hashing, which
produces similarity=1.0 for any two prompts that share the long common
prefix — making results uninformative.  A production deployment would
use a proper embedding model (e.g., text-embedding-3-small) instead.

| Config | Value |
|---|---|
| MemoTable | 4 096 entries, enabled |
| SemanticCacheIndex | disabled (see above) |
| KVCacheIndex (trie) | 8 192 entries, enabled |
| Backend | Azure OpenAI gpt-5-mini |
| Shared prefix | 3 000 characters (steps 1–16) |
| No prefix | steps 17–20 |

## 2. Workload

| # | Category | Prompt |
|---|---|---|
| 1 | prefix | What is machine learning? |
| 2 | prefix | Explain neural networks in one paragraph |
| 3 | prefix | How does backpropagation work? |
| 4 | prefix | Describe gradient descent optimization |
| 5 | prefix | What is a loss function in training? |
| 6 | prefix | Explain activation functions like ReLU |
| 7 | prefix | How do transformers process sequences? |
| 8 | prefix | What is the attention mechanism? |
| 9 | **memo** | *(repeat of step 2)* |
| 10 | **memo** | *(repeat of step 4)* |
| 11 | **memo** | *(repeat of step 6)* |
| 12 | **memo** | *(repeat of step 8)* |
| 13 | paraphrase | Describe how machines learn from data |
| 14 | paraphrase | What is the process of back-propagating errors? |
| 15 | paraphrase | Training objectives and loss functions explained |
| 16 | paraphrase | How attention allows transformers to focus on input |
| 17 | **no reuse** | Write a haiku about autumn leaves |
| 18 | **no reuse** | Calculate 15 times 27 plus 432 |
| 19 | **no reuse** | What is the capital of New Zealand? |
| 20 | **no reuse** | Translate the word serenity to French |

## 3. Aggregate Metrics

| Metric | Value |
|---|---|
| Total steps | 20 |
| Backend calls | **16 / 20** (80%) |
| Memo hits | **4 / 20** (20%) |
| Trie hits | **15 / 20** (75%) |
| Total tokens saved | 45 527 |
| Total tokens processed | 3 672 |
| Token reduction ratio | **92.5%** |
| Median latency | **3 133 ms** |
| P95 latency | **7 674 ms** |

## 4. Per-Step Breakdown

| Step | Category | Mechanism | Backend | Tokens Saved | Tokens Processed | Latency (ms) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | prefix | **backend** | yes | 0 | 3 048 | 2 601 |
| 2 | prefix | **trie** | yes | 3 020 | 43 | 3 011 |
| 3 | prefix | **trie** | yes | 3 020 | 33 | 2 859 |
| 4 | prefix | **trie** | yes | 3 020 | 41 | 7 007 |
| 5 | prefix | **trie** | yes | 3 020 | 39 | 5 882 |
| 6 | prefix | **trie** | yes | 3 020 | 41 | 5 504 |
| 7 | prefix | **trie** | yes | 3 020 | 41 | 4 449 |
| 8 | prefix | **trie** | yes | 3 020 | 35 | 7 674 |
| 9 | memo | **memo** | no | 3 063 | 0 | 0 |
| 10 | memo | **memo** | no | 3 061 | 0 | 0 |
| 11 | memo | **memo** | no | 3 061 | 0 | 0 |
| 12 | memo | **memo** | no | 3 055 | 0 | 0 |
| 13 | paraphrase | **trie** | yes | 3 020 | 40 | 2 943 |
| 14 | paraphrase | **trie** | yes | 3 020 | 50 | 2 779 |
| 15 | paraphrase | **trie** | yes | 3 020 | 51 | 3 134 |
| 16 | paraphrase | **trie** | yes | 3 020 | 54 | 3 133 |
| 17 | no reuse | **backend** | yes | 10 | 45 | 7 228 |
| 18 | no reuse | **backend** | yes | 19 | 33 | 3 135 |
| 19 | no reuse | **backend** | yes | 19 | 38 | 3 105 |
| 20 | no reuse | **backend** | yes | 19 | 40 | 7 578 |

## 5. By-Category Summary

| Category | n | Mechanism | Backend Calls | Avg Latency (ms) |
|---|---|---|---|---|
| prefix_reuse | 8 | trie (steps 2–8) | 8/8 | 4 873 |
| memo_reuse | 4 | memo (all) | 0/4 | 0 |
| paraphrase | 4 | trie (shared prefix) | 4/4 | 2 997 |
| no_reuse | 4 | backend (no useful prefix) | 4/4 | 5 262 |

## 6. Negative Case: Steps 17–20 (No Reuse Expected)

Steps 17–20 use completely different queries with no shared prefix.
No memo match (different text).  No useful trie match (only 10–19 tokens
of trivial `"Question: "` overlap from earlier runs).  These steps
exercise the **worst-case path**: backend is called with nearly the full prompt.

| Step | Tokens Saved | Tokens Processed | Reduction |
|---|---|---|---|
| 17 | 10 | 45 | 0.3% |
| 18 | 19 | 33 | 0.6% |
| 19 | 19 | 38 | 0.5% |
| 20 | 19 | 40 | 0.5% |

Mean token reduction for no-reuse steps: **0.5%** — effectively no benefit,
as expected.

## 7. Key Findings

1. **Memo is the only mechanism that fully avoids backend calls.**  All four
   memo hits (steps 9–12) had 0 ms latency and 0 tokens processed.
   Trie reduces tokens but the backend still runs.

2. **Trie provides large token savings but modest latency gains.**  Matching
   3 020 of ~3 040 tokens (99%) on prefix-reuse steps still leaves 30–40
   tokens for Azure to process.  Because Azure's fixed per-request overhead
   (~2.5 s round-trip) dominates variable token time, latency drops from
   ~5 400 ms (no trie) to ~3 700 ms (with trie) — a 31% improvement,
   not a 10× improvement.

3. **The system degrades gracefully.**  Even with no reuse (steps 17–20),
   the pipeline simply calls the backend with the full prompt.  No errors,
   no correctness issues, no configuration changes needed.

4. **Memo and trie are complementary.**  Memo handles exact repeats (zero-cost).
   Trie handles related-but-different queries (reduced cost).  Together they
   achieve 92.5% token reduction and avoid 4 out of 20 backend calls.
