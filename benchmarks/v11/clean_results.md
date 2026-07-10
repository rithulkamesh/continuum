# Continuum v1.1 — Clean Isolated Mechanism Results

**Date**: 2026-05-03
**Backend**: Azure OpenAI (gpt-5-mini)
**Total Azure API calls**: 16 (10 + 1 + 5)

## Methodology

Each experiment isolates exactly ONE v1.1 reuse mechanism by
disabling the others.  Every trial logs four flags:

| Flag | Meaning |
|---|---|
| `memo_hit` | Exact match in MemoTable → backend skipped |
| `semantic_hit` | Embedding similarity ≥ threshold in SemanticCacheIndex → backend skipped |
| `trie_hit` | Prefix match in KVCacheIndex → backend called with reduced tokens |
| `backend_called` | Backend was actually invoked |

Disable rules:
- **Disabled subsystem is not wired** into the Session.
- When trie is disabled, a fresh KVCacheIndex is created per trial.
- No exact prompt repetition unless testing memo (Rule 2).
- All prompts are unique within E1 and E2 (Rule 2).

---

## E1 — Trie / Prefix Reuse

**Config**: memo=off, semantic=off, trie=on. 10 unique suffixes, 3000-char shared prefix.

| Trial | Backend | Memo | Semantic | Trie | Tokens Saved | Tokens Sent | Latency |
|---|---|---|---|---|---|---|---|
| 1 | **yes** | 0 | 0 | 0 | 0 | 3037 | 3390 ms |
| 2 | **yes** | 0 | 0 | **1** | **3010** | 26 | 3287 ms |
| 3 | **yes** | 0 | 0 | **1** | **3010** | 32 | 5228 ms |
| 4 | **yes** | 0 | 0 | **1** | **3010** | 41 | 2969 ms |
| 5 | **yes** | 0 | 0 | **1** | **3010** | 38 | 3021 ms |
| 6 | **yes** | 0 | 0 | **1** | **3010** | 41 | 3005 ms |
| 7 | **yes** | 0 | 0 | **1** | **3010** | 40 | 5814 ms |
| 8 | **yes** | 0 | 0 | **1** | **3010** | 34 | 2839 ms |
| 9 | **yes** | 0 | 0 | **1** | **3010** | 31 | 2718 ms |
| 10 | **yes** | 0 | 0 | **1** | **3010** | 33 | 2704 ms |

**Summary**: 10/10 backend calls. 0 memo, 0 semantic, 9/9 trie hits (trial 1 is cache warm).
The trie matched 3010 tokens of shared prefix on every subsequent trial.
Backend still executed but processed only ~30 tokens of unique suffix.
Token reduction: **~99%** (3010 / ~3040 per trie-hit trial).

---

## E2 — Semantic Cache Reuse

**Config**: memo=off, semantic=on (threshold=0.80), trie=off. 10 unique paraphrases, no shared prefix.

| Trial | Backend | Memo | Semantic | Trie | Similarity | Latency |
|---|---|---|---|---|---|---|
| 1 | **yes** | 0 | 0 | 0 | — | 2731 ms |
| 2 | no | 0 | **1** | 0 | 0.9812 | 0 ms |
| 3 | no | 0 | **1** | 0 | 0.9815 | 0 ms |
| 4 | no | 0 | **1** | 0 | 0.9795 | 0 ms |
| 5 | no | 0 | **1** | 0 | 0.9808 | 0 ms |
| 6 | no | 0 | **1** | 0 | 0.9699 | 0 ms |
| 7 | no | 0 | **1** | 0 | 0.9848 | 0 ms |
| 8 | no | 0 | **1** | 0 | 0.9819 | 0 ms |
| 9 | no | 0 | **1** | 0 | 0.9803 | 0 ms |
| 10 | no | 0 | **1** | 0 | 0.9814 | 0 ms |

**Summary**: 1/10 backend calls. 0 memo, 9/9 semantic hits, 0 trie.
All paraphrases exceeded the 0.80 threshold (range 0.97–0.98).
Backend skipped entirely on semantic hits (0 ms latency, 0 tokens processed).

**Caveat**: BruteForceEmbedding uses character n-gram hashing, so similarity
reflects textual overlap, not semantic meaning. A real embedding model
(e.g., Ada, text-embedding-3-small) would be needed for true semantic reuse.

---

## E3 — Memo / Exact-Match Reuse

**Config**: memo=on, semantic=off, trie=off. 5 unique tool calls, each repeated once. No shared prefix.

| Trial | Phase | Backend | Memo | Semantic | Trie | Latency |
|---|---|---|---|---|---|---|
| 1 | FIRST | **yes** | 0 | 0 | 0 | 2540 ms |
| 2 | FIRST | **yes** | 0 | 0 | 0 | 2380 ms |
| 3 | FIRST | **yes** | 0 | 0 | 0 | 5509 ms |
| 4 | FIRST | **yes** | 0 | 0 | 0 | 3222 ms |
| 5 | FIRST | **yes** | 0 | 0 | 0 | 2548 ms |
| 6 | REPEAT | no | **1** | 0 | 0 | 0 ms |
| 7 | REPEAT | no | **1** | 0 | 0 | 0 ms |
| 8 | REPEAT | no | **1** | 0 | 0 | 0 ms |
| 9 | REPEAT | no | **1** | 0 | 0 | 0 ms |
| 10 | REPEAT | no | **1** | 0 | 0 | 0 ms |

**Summary**: 5/10 backend calls. 5/5 memo hits, 0 semantic, 0 trie.
Every repeat was an exact match in MemoTable.  Backend skipped entirely.
MemoTable contains 5 entries.

---

## Cross-Experiment Isolation Verification

| Experiment | Memo Hits | Semantic Hits | Trie Hits | Backend Calls | Mechanism Active? |
|---|---|---|---|---|---|
| E1 (trie) | 0 | 0 | 9 | 10 | **Yes** |
| E2 (semantic) | 0 | 9 | 0 | 1 | **Yes** |
| E3 (memo) | 5 | 0 | 0 | 5 | **Yes** |

No cross-contamination.  Each mechanism fires exclusively when enabled.
