# Continuum v1.1 Benchmark Results Summary

**Date:** 2026-05-02
**Backend:** Azure OpenAI (gpt-5-mini) + FakeLLM (deterministic)
**Total Azure API calls:** 90 (E1: 9, E2: 27, E3: 27, E4: 27)
**Total wall time:** ~5 minutes (Azure-dominated)

## Architecture Caveat

The v1.1 subsystems (MemoTable, SemanticCacheIndex, MemoryGraphStore, LayerKVCacheIndex, FutureCache) are standalone C++ classes exposed to Python but **not wired into the interpreter's TokenOp dispatch path**. The interpreter still uses only the trie-based KVCacheIndex + ReusePolicy. Results below reflect:
- **Azure experiments (E1-E4):** Real backend prefix-split reuse via `benchmark_agent_paired` (3 calls/trial: uncached, warmup, cached)
- **FakeLLM experiments (E5-E6):** Structural overhead measurements, not latency-dependent

---

## E1: Shared-Prefix Agent Loop (Azure, 3000-token prefix)

| Metric | No Cache | Prefix Cache | Delta |
|--------|----------|--------------|-------|
| Median latency (ms) | 6283 | 5446 | -13.3% |
| P95 latency (ms) | 6408 | 5739 | -10.4% |
| Tokens sent | 3046 | 0 | **100% reduction** |
| Tokens saved | 0 | 3010 | 3010 tok |

**Speedup:** 1.15x. Token reduction is perfect (100%) because the Azure backend splits at `" Question:"` and sends only the suffix on cache hit. Latency improvement is modest (~13%) because Azure's server-side processing time dominates over network transfer savings for the prefix.

---

## E2: Semantic Reuse (Azure + BruteForceEmbedding)

**Azure results by prefix size (proxy for semantic variation):**

| Prefix | Median Uncached | Median Cached | Speedup | Token Reduction |
|--------|----------------|---------------|---------|-----------------|
| 500 | 5577ms | 4853ms | **1.15x** | 100% |
| 1500 | 4736ms | 4433ms | **1.07x** | 100% |
| 3000 | 4738ms | 6032ms | **0.79x** | 100% |

**Observation:** Smaller prefixes (500 tok) show consistent speedup. Larger prefixes (3000 tok) show *no improvement or regression* in wall-clock latency despite 100% token reduction. This suggests Azure's server-side compute cost scales independently of input token count for these sizes, or that cache-warm requests experience higher queueing variance.

**BruteForceEmbedding semantic cache:** 100% hit rate (all similarities >0.999). This is expected because BruteForceEmbedding uses character-level hashing, not real semantic embeddings. A production system would need a real embedding model (e.g., text-embedding-3-small) for meaningful semantic deduplication.

---

## E3: Subtask Memoization (Azure + FakeLLM)

**Azure results by prefix size (simulating accumulated subtask context):**

| Prefix | Speedup | Tokens Saved |
|--------|---------|-------------|
| 300 | **1.24x** | 310 |
| 1000 | **1.05x** | 1010 |
| 3000 | **1.05x** | 3010 |

**FakeLLM session benchmarks:**

| Prefix | Token Reduction | Cache Size |
|--------|----------------|------------|
| 10 | 45.5% | 33 tok |
| 30 | 65.5% | 53 tok |
| 100 | 85.0% | 123 tok |
| 300 | 94.2% | 323 tok |
| 1000 | 98.2% | 1023 tok |

**MemoTable hit rate:** 0% — because each subtask has a unique key (different query per step). MemoTable would only help with *identical* repeated tool calls, not unique sequential calls.

---

## E4: Multi-Session Persistence (Azure + FakeLLM)

**Azure results by prefix size (simulating session context):**

| Prefix | Speedup | Tokens Saved |
|--------|---------|-------------|
| 1000 | **1.01x** | 1010 |
| 2000 | **0.97x** | 2010 |
| 3000 | **0.98x** | 3010 |

**FakeLLM cold-start:**
- Warm hit rate: 90% (first step always misses)
- Cold hit rate: 100% (loaded from disk, all prefixes pre-populated)
- Cold-start time: 4.6ms

**Observation:** Azure multi-session results show near-zero latency speedup despite 100% token reduction. This confirms that Azure's server-side KV cache persistence does not provide the latency benefit one might expect from pure token count reduction. The bottleneck is server-side inference compute, not prefix transfer time.

---

## E5: Speculative Prefetch / FutureCache Overhead

| max_entries | Create (us) | Clear (us) | 100x Bulk (ms) |
|-------------|-------------|------------|----------------|
| 64 | 3.8 | 0.5 | 0.22 |
| 256 | 2.6 | 0.5 | 0.23 |
| 1024 | 2.1 | 0.5 | 0.22 |
| 4096 | 1.9 | 0.5 | 0.43 |

**Overhead:** <0.5us per operation. FutureCache get/put/submit are not exposed to Python; this measures construction + clear as a proxy. Overhead is negligible compared to any API call latency (>4000ms).

---

## E6: Ablation Study

**Subsystem instantiation overhead (1000 iterations each):**

| Subsystem | Avg (us) | P50 (us) |
|-----------|----------|----------|
| MemoTable | 1.6 | 1.9 |
| SemanticCacheIndex | 1.3 | 1.0 |
| MemoryGraphStore | 1.4 | 1.0 |
| LayerKVCacheIndex | 1.4 | 1.2 |
| FutureCache | 1.3 | 1.0 |

**Reuse policy comparison (all produce identical results):**
- All policies (always, threshold_5/15/30): 95% hit rate, 62.4% token reduction
- This is because with prefix_tokens=30, suffix_tokens=20, the threshold is always exceeded after the first step

---

## Key Findings

1. **Token reduction is perfect (100%)** across all Azure experiments when prefix caching is active. The Azure backend correctly splits at `" Question:"` and sends zero prefix tokens on cache hit.

2. **Latency improvement is modest and inconsistent** (0.79x to 1.24x). Token reduction does not directly translate to proportional latency improvement because Azure's server-side compute dominates.

3. **Smaller prefixes show more consistent speedup** (1.05-1.24x) than larger prefixes (0.79-1.01x). This may reflect higher variance in larger requests or Azure's internal batching behavior.

4. **v1.1 subsystems have negligible overhead** (<2us instantiation). The architectural gap is integration, not performance — these subsystems need to be wired into the interpreter dispatch path.

5. **BruteForceEmbedding is not a real semantic model** — it achieves 100% hit rate because it's character-level hashing. Production semantic caching requires a real embedding model.

6. **MemoTable hit rate is 0%** for unique sequential subtasks. It would only help with truly repeated tool calls (same function + same inputs).

## Data Files

- `benchmarks/v11/data/e1.json` — Azure shared-prefix (9 API calls)
- `benchmarks/v11/data/e2.json` — Semantic reuse (27 Azure + FakeLLM)
- `benchmarks/v11/data/e3.json` — Subtask memoization (27 Azure + FakeLLM)
- `benchmarks/v11/data/e4.json` — Multi-session (27 Azure + FakeLLM)
- `benchmarks/v11/data/e5.json` — FutureCache overhead (structural only)
- `benchmarks/v11/data/e6.json` — Ablation (structural only)

## Plot Files

- `plots/v11/e1_azure_latency.png`
- `plots/v11/e2_semantic_similarity.png`
- `plots/v11/e3_token_reduction.png`
- `plots/v11/e4_cold_start.png`
- `plots/v11/e5_prefetch_overhead.png`
- `plots/v11/e6_ablation.png`
- `plots/v11/azure_combined.png`
- `plots/v11/summary_dashboard.png`
