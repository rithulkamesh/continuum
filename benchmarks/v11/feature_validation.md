# Continuum v1.1 Feature Validation Report

**Date:** 2026-05-02
**Method:** C++ validation function (`validate_v11_features`) exercising the full Interpreter TokenOp execution path with FakeLLM backend. Logs captured via spdlog.

---

## Test 1 — Exact Repeat (Memoization)

**Setup:** Same prompt executed 3 times in same Interpreter session.

| Run | Log Event | Result |
|-----|-----------|--------|
| 1 | `cache_miss → backend_run(tokens_sent=34, tokens_saved=0)` | MISS — seeds memo cache |
| 2 | `memo_hit backend=fake model=fake/m1 node=generate` | HIT — backend NOT called |
| 3 | `memo_hit backend=fake model=fake/m1 node=generate` | HIT — backend NOT called |

**Subsystem sizes:** memo=1, sc=1, trie=34
**Verdict: PASS** — Memoization fires on exact repeat. Backend skipped on hits.

---

## Test 2 — Paraphrase (Semantic Reuse)

**Setup:** Three different prompts about Continuum caching in same session.

| Run | Prompt | Log Event | Result |
|-----|--------|-----------|--------|
| 1 | "Summarize Continuum cache behavior" | `cache_miss → backend_run` | MISS — seeds semantic cache |
| 2 | "Explain how Continuum caching works" | `semantic_hit similarity=0.8573` | HIT — backend skipped |
| 3 | "Give an overview of Continuum's cache system" | `semantic_hit similarity=0.8668` | HIT — backend skipped |

**Threshold:** 0.80. All paraphrases exceeded threshold.
**Verdict: PASS** — Semantic reuse fires on paraphrased prompts. Backend skipped on hits.

**Caveat:** BruteForceEmbedding uses character n-gram hashing, not true semantic embeddings. Similarity reflects string overlap, not meaning. Works here because prompts share many words. Would NOT distinguish "What is caching?" from "What is baking?" with different vocabulary.

---

## Test 3 — Prefix Sharing (Trie)

**Setup:** Same 67-char prefix, 3 different suffixes. Semantic cache DISABLED to isolate trie.

| Run | Total Tokens | Log Event | Prefix Saved |
|-----|-------------|-----------|-------------|
| 1 | 91 | `cache_miss → backend_run` | 0 |
| 2 | 100 | `trie_hit prefix_len=69 remaining=31 tokens_saved=69` | 69 tokens (76%) |
| 3 | 94 | `trie_hit prefix_len=69 remaining=25 tokens_saved=69` | 69 tokens (73%) |

**Verdict: PASS** — Trie fires on shared prefix. 69/91 = 76% token reduction. Backend still called but with fewer tokens.

**Note:** Trie finds 69 chars of common prefix (the shared system message), even though Run 1 is 91 tokens and Run 2 is 100 tokens — the trie matches as far as the shorter common prefix.

---

## Test 4 — Priority Order

**Setup:** Same prompt executed twice (exact repeat). All three caches active.

| Run | Log Event | Path Taken |
|-----|-----------|------------|
| 1 | `cache_miss → backend_run` | Full miss — seeds all caches |
| 2 | `memo_hit` | MEMO fires first — semantic and trie NOT checked |

**Verdict: PASS** — Memo takes priority over semantic and trie. No semantic_hit or trie_hit logged for Run 2.

---

## Summary Table

| Test | Memo | Semantic | Trie | Backend Skipped | Verdict |
|------|------|----------|------|-----------------|---------|
| T1: Exact repeat | Run2=HIT, Run3=HIT | — | — | Yes (2 runs) | PASS |
| T2: Paraphrase | — | Run2=HIT(0.86), Run3=HIT(0.87) | — | Yes (2 runs) | PASS |
| T3: Prefix sharing | — | — | Run2=HIT(69tok), Run3=HIT(69tok) | No, but reduced | PASS |
| T4: Priority | Run2=HIT | NOT checked | NOT checked | Yes (1 run) | PASS |

---

## Verdict

| Question | Answer |
|----------|--------|
| Is memoization working? | **YES** — fires on exact repeat, backend skipped |
| Is semantic reuse working? | **YES** — fires on paraphrases above 0.80 threshold, backend skipped |
| Is trie reuse unaffected? | **YES** — fires on shared prefix, 69 tokens saved (76% reduction) |
| Is priority ordering correct? | **YES** — memo fires first; semantic and trie never checked when memo matches |

## Known Limitations

1. **BruteForceEmbedding is not truly semantic** — it uses character n-gram hashing. High cosine similarity reflects string overlap, not meaning. Will match unrelated prompts with shared vocabulary.
2. **Test 3 creates same graph** — different prompts are passed as input to the SAME graph node. MemoTable key includes inputs_hash, so different inputs = different keys = no memo cross-talk (correct).
3. **FakeLLMBackend** — deterministic, no real latency benefit from caching. Real benefit measured with Azure backend in E1 (12% speedup, 98.8% token reduction).
