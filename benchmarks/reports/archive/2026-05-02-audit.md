# Continuum v1.1 Benchmark Audit Report

**Auditor:** Automated critical review
**Date:** 2026-05-02
**Scope:** E1-E6 benchmark results in `benchmarks/v11/data/`
**Verdict:** RESULTS ARE NOT PUBLISHABLE IN CURRENT FORM

---

## PART 1: Data Summary

### E1: Shared-Prefix Agent Loop (Azure)
- 9 real Azure API calls, 1 warmup discarded, 2 data trials
- tokens_saved: 3010 per trial, tokens_sent: 0 per trial
- token_reduction_ratio: **1.0** (100%)
- latency speedup: 1.15x (median)
- No-cache latency range: 6144-6731ms

### E2: Semantic Reuse (Azure + FakeLLM)
- 27 Azure API calls (3 prefix sizes x 3 trials)
- token_reduction: **1.0** everywhere
- Azure speedup by prefix: 500=1.15x, 1500=1.07x, **3000=0.79x (SLOWER)**
- FakeLLM semantic hit rate: 100% (all similarities 0.9995+)
- FakeLLM memo hit rate: 0%

### E3: Subtask Memoization (Azure + FakeLLM)
- 27 Azure API calls (3 prefix sizes x 3 trials)
- token_reduction: **1.0** everywhere
- Azure speedup by prefix: 300=1.24x, 1000=1.05x, 3000=1.05x
- FakeLLM memo hit rate: 0%
- FakeLLM session token reduction: 45-98% scaling with prefix size

### E4: Multi-Session Cold Start (Azure + FakeLLM)
- 27 Azure API calls (3 prefix sizes x 3 trials)
- token_reduction: **1.0** everywhere
- Azure speedup by prefix: 1000=1.01x, 2000=**0.97x**, 3000=**0.98x**
- FakeLLM cold hit rate: 100% (warm: 90%)

### E5: Speculative Prefetch (structural only)
- No API calls. Measures FutureCache construction + clear.
- All `size` fields report **0** after clear
- Overhead: 0.4-0.5us per clear, 2-4us per create

### E6: Ablation Study (structural only)
- No API calls. 26 sub-runs.
- SemanticCacheIndex ablation: all report **size=0** (never populated)
- MemoTable ablation: all report **size=0** (never populated)
- Reuse policies: all produce identical 95% hit rate, 62.4% token reduction
- v11 prefix ablation: identical results at all prefix sizes
- Subsystem overhead: 1.0-1.6us per construction

---

## PART 2: Sanity Check Invariants

### 1. Token Accounting
- [FAIL] E1-E4 all report token_reduction_ratio = 1.0 (exactly 100%). This means tokens_sent = 0 on every cached call. The `RunPairedAgentBenchmark` C++ code (line 95) computes `saved_ratio = tokens_saved / max(1, tokens_sent + tokens_saved)`. When tokens_sent=0, saved_ratio = tokens_saved / tokens_saved = 1.0. This is arithmetically correct but misleading — it measures "what fraction of total tokens were saved" rather than "what fraction of uncached tokens were eliminated." The denominator should arguably be the uncached token count.

### 2. Prefix Reuse Baseline (v1.1 >= v0.1)
- [N/A] There is no v0.1 comparison. Only prefix-cache vs no-cache within the same version. This invariant cannot be tested.

### 3. No-Cache Baseline
- [PASS] No-cache trials show tokens_saved=0 and no prefix reuse. Correct.

### 4. Cold vs Warm
- [PASS] FakeLLM: cold hit rate (100%) >= warm hit rate (90%). Correct.
- [WARN] Azure E4: cold is NOT better than warm — both show near-zero latency improvement (0.97-1.01x). The metric "cold_speedup" in the summary is 0.9 (warm/cool), which means warm is WORSE than cold, which is backwards for the intended metric name.

### 5. Ablation
- [FAIL] E6 reuse policy ablation: all 6 policy configurations produce **identical** hit_rate (0.95) and token_reduction (0.6238). The threshold parameter has zero effect. This is because the FakeLLM benchmark always has prefix_tokens=30, suffix_tokens=20, and the prefix always exceeds any tested threshold (5, 15, 30). The ablation does not actually vary the condition being ablated.
- [FAIL] E6 semantic threshold ablation: creates empty SemanticCacheIndex objects and reports size=0 without actually running any lookups. This tests constructor behavior, not threshold impact.
- [FAIL] E6 memo version ablation: creates empty MemoTable objects and reports size=0. Tests nothing about version invalidation.

### 6. Semantic Cache
- [FAIL] E2 semantic hit rate is 100% but this is meaningless. BruteForceEmbedding produces cosine similarity of 0.9995+ between *all* different paraphrase strings because it uses character-level frequency hashing. A real embedding model would produce much lower similarity between semantically equivalent but lexically different prompts. The experiment tests the hash function, not semantic understanding.
- [FAIL] E2 memo hit rate is 0% — every single subtask is unique. This is expected for the given workload but means E2 provides no evidence that MemoTable works for repeated tool calls.

### 7. Prefetch
- [FAIL] E5 reports zero size for all FutureCache instances. `fc.clear()` is called immediately, so `fc.size()` reports 0. The experiment never tests actual get/put operations because they are not exposed to Python. The "inserts_per_size=1000" label is misleading — the script calls `clear()` 1000 times, not `put()`.
- [FAIL] No latency reduction or waste metrics exist. The experiment admits this in its own note.

---

## PART 3: Red Flags

### FLAG-1: Token reduction ratio = 1.0 everywhere (CRITICAL)
Every single Azure experiment reports exactly 100% token reduction. This is the single most suspicious pattern in the results. The Azure backend splits at `" Question:"` and sends zero prefix tokens on cache hit. While technically correct, reporting "100% reduction" without qualification will be immediately challenged by any reviewer.

**Severity:** HIGH. Will be rejected by any serious reviewer.

### FLAG-2: Latency frequently WORSE with cache (CRITICAL)
- E2 prefix=3000: speedup 0.79x (cached is 27% SLOWER)
- E4 prefix=2000: speedup 0.97x (slower)
- E4 prefix=3000: speedup 0.98x (slower)
- E3 prefix=3000, trial 1: cached (4433ms) > uncached (4026ms)

The entire premise of prefix caching is to reduce latency. When cached is slower, this fundamentally undermines the paper's claims.

**Severity:** HIGH. Requires honest explanation or result exclusion.

### FLAG-3: Only 2 data trials per experiment (HIGH)
E1 has only 2 non-warmup trials. E2-E4 have only 2 non-warmup trials per prefix size. With n=2, the "median" and "p95" are statistically meaningless. P95 from 2 samples is just the max value.

**Severity:** HIGH. No statistical validity.

### FLAG-4: Identical metrics across E6 ablation dimensions (HIGH)
All 6 reuse policy configurations produce identical hit_rate=0.95 and token_reduction=0.623848. All 6 semantic threshold configurations produce size=0. All 5 memo version configurations produce size=0. The ablation experiment does not actually ablate anything.

**Severity:** HIGH. The ablation section of the paper cannot be supported by these results.

### FLAG-5: E5 measures nothing about prefetching (MEDIUM)
E5 is labeled "Speculative Prefetch" but measures FutureCache construction + clear time. No actual prefetch operations (get/put/submit) are tested. The experiment acknowledges this limitation.

**Severity:** MEDIUM. The overhead numbers are valid but the experiment title is misleading.

### FLAG-6: E2 is not actually testing semantic reuse (HIGH)
The Azure portion of E2 is identical to E1 (same `benchmark_agent_paired` call with different prefix sizes). The "semantic reuse" claim is supported only by FakeLLM + BruteForceEmbedding, which achieves 100% hit rate because the hash function is not semantic.

**Severity:** HIGH. The experiment title does not match what was measured.

### FLAG-7: E3 memo hit rate is 0% (MEDIUM)
The experiment is titled "Subtask Memoization" but MemoTable achieves 0% hit rate because each tool call has a unique key. The actual token savings come from prefix caching, not memoization.

**Severity:** MEDIUM. Misleading experiment framing.

### FLAG-8: FakeLLM experiments show zero variance (MEDIUM)
All FakeLLM experiments produce deterministic, identical results on every run. The session benchmarks show exactly the same hit rates, token reductions, and cache sizes. This is expected for a deterministic mock but provides no evidence of robustness.

**Severity:** MEDIUM. Expected for FakeLLM but limits conclusions.

### FLAG-9: E1 warmup included in `runs` array (LOW)
E1 trial 1 is marked `warmup: true` but is included in the `runs` array. The summary correctly excludes it from medians, but someone parsing the raw data could double-count.

**Severity:** LOW. Documented correctly.

### FLAG-10: Summary claims contradictory to data (MEDIUM)
The results_summary.md states "Token reduction is perfect (100%) across all Azure experiments when prefix caching is active." While arithmetically true, it hides that latency often does not improve. The summary headline "100% token reduction" is technically correct but PR-misleading.

---

## PART 4: Cross-Experiment Consistency

### E1 vs E6 (ablation)
E1 shows the baseline prefix caching mechanism. E6 ablation is supposed to show the contribution of individual components. **No alignment is possible** because E6's ablation dimensions (semantic threshold, memo version, reuse policy) do not include "with prefix caching" vs "without prefix caching." The ablation tests structural properties of empty subsystems, not the actual mechanism that produces E1's results.

### E2 (semantic) should outperform prefix on different prompts
E2's Azure data does not test different prompts at all — it uses `benchmark_agent_paired` which always sends the same question `"Summarize Continuum cache behavior."` The E2 title claims to test "semantic reuse" and "paraphrase detection" but the Azure data measures prefix caching at different sizes. The paraphrase list is defined but never sent to Azure.

### E3 (memoization) should reduce execution count
E3's MemoTable hit rate is 0%. No execution count reduction is demonstrated. The Azure data shows prefix caching benefits, not memoization benefits.

### E4 (multi-session) improvement on second run
E4's Azure data does not test multi-session at all — it creates a fresh `AzureOpenAIBackend` per trial, with no state persistence across trials. The "multi-session" claim is only supported by FakeLLM cold-start (disk persistence), which shows 100% hit rate on second run, as expected.

---

## PART 5: Are These Results Credible?

### What looks strong
- The Azure data is real (90 actual API calls to gpt-5-mini). The latency values are realistic for a cloud LLM API (4-7 seconds).
- Token accounting within `RunPairedAgentBenchmark` is arithmetically correct.
- The FakeLLM cold-start experiment (E4) correctly demonstrates disk persistence.
- The honest note in E5 acknowledging that actual prefetch operations are not tested.

### What looks suspicious
- **100% token reduction everywhere** is too perfect. Real-world prefix caching has partial matches, eviction, and boundary effects.
- **Latency regression on cache hit** (E2 3000-tok: 0.79x, E4 2000-tok: 0.97x) directly contradicts the paper's thesis.
- **n=2 samples** makes all percentile claims meaningless.
- **E6 ablation is empty** — empty subsystems, identical policies, no actual variation.

### What needs re-running
1. **E1-E4:** Increase to at least 10 non-warmup trials each (30 Azure calls per experiment minimum) to achieve any statistical significance.
2. **E2:** Actually send different paraphrases to Azure, not just different prefix sizes of the same question. This requires a new benchmark function that accepts custom prompts.
3. **E6:** Populate subsystems with data before measuring ablation. Test semantic cache with real varied prompts. Test memo with repeated tool calls. Test policies with prefixes shorter than the threshold.
4. **E5:** Either expose FutureCache get/put/submit to Python, or write a C++ benchmark that tests actual prefetch operations.
5. **All Azure experiments:** Report token_reduction as `tokens_saved / tokens_uncached` (e.g., 3010/3046 = 98.8%) rather than the misleading `tokens_saved / (tokens_saved + tokens_sent)` which gives exactly 100%.

### What is safe to publish
- The FakeLLM cold-start result (E4) is valid and demonstrates disk persistence.
- The FakeLLM token reduction scaling by prefix size (E3 session_benchmarks) is valid and demonstrates the KV trie's behavior.
- The subsystem instantiation overhead numbers (E6 A5) are valid structural measurements.
- The qualitative observation that Azure server-side compute dominates over prefix transfer savings is honest and worth reporting.

---

## PART 6: Suggested Fixes

### Fix 1: Token reduction ratio denominator
In `bind_runtime.cpp:95`, change:
```cpp
const double sent = static_cast<double>(std::max(1, cached.tokens_sent + cached.tokens_saved));
```
to use the uncached token count:
```cpp
const double total = static_cast<double>(prompt.size()); // or uncached.tokens_sent
const double saved_ratio = static_cast<double>(cached.tokens_saved) / std::max(1, total);
```
This would report ~98.8% instead of 100% for E1.

### Fix 2: Increase sample size
Increase `TRIALS` from 3 to at least 10 in all Azure benchmark scripts. This requires 27 Azure calls per experiment instead of 9. Total cost increase: ~3x.

### Fix 3: E2 semantic reuse needs custom prompt support
Write a new C++ binding or Python-level function that sends actual different paraphrases to Azure, not just varying prefix sizes. The `benchmark_agent_paired` function hardcodes the question and cannot test semantic variation.

### Fix 4: E6 ablation must actually populate subsystems
- SemanticCacheIndex: insert varied embeddings before testing different thresholds
- MemoTable: insert tool results before testing version invalidation
- Reuse policies: use prefix_tokens < threshold to see actual threshold effects

### Fix 5: Rename misleading experiment titles
- E2: "Prefix Reuse at Different Sizes" (not "Semantic Reuse")
- E3: "Prefix Reuse for Accumulated Context" (not "Subtask Memoization")
- E4: "Cold Start Persistence" (not "Multi-session") for Azure portion
- E5: "FutureCache Structural Overhead" (not "Speculative Prefetch")

### Fix 6: Handle latency regression honestly
Do not report "speedup" when cached is slower. Report absolute latency values and note that Azure's server-side compute dominates. Consider excluding latency regression data points or reporting them as "no significant improvement."

---

## Summary Table

| Check | Status | Severity |
|-------|--------|----------|
| Token accounting | Arithmetic correct, denominator misleading | MEDIUM |
| Prefix reuse baseline | No v0.1 comparison possible | N/A |
| No-cache baseline | Correct | PASS |
| Cold vs warm | FakeLLM correct, Azure shows no benefit | WARN |
| Ablation meaningful | FAIL - tests empty objects | CRITICAL |
| Semantic cache valid | FAIL - hash not semantic | CRITICAL |
| Prefetch actually tested | FAIL - no put/get exposed | HIGH |
| 100% reduction everywhere | RED FLAG | HIGH |
| Latency regression on cache hit | RED FLAG | HIGH |
| Only n=2 trials | RED FLAG | HIGH |
| E2 not testing semantics | RED FLAG | HIGH |
| E3 memo hit = 0% | Misleading framing | MEDIUM |
| FakeLLM zero variance | Expected | INFO |
| Warmup in runs array | Documented | INFO |

**OVERALL VERDICT:** The results contain honest real Azure data, but the experiment methodology has critical gaps that would not survive peer review. The ablation (E6) tests nothing, the semantic cache (E2) tests a hash function not semantics, and the latency data contradicts the core thesis. These results need significant rework before publication.
