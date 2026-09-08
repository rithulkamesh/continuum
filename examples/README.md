# Examples

Runnable scenarios for the Continuum runtime. Every script here uses the
deterministic `FakeLLM` backend (or pure Python), so the output is stable and
CI-checkable -- no API keys, no network.

```bash
# from the repo root, with the native extension built (scripts/build.sh)
PYTHONPATH=python python examples/01_reuse_stack.py
```

`10_prompt_optimization.py` is pure Python and needs no build.

| # | Script | Scenario | Continuum surface |
|---|--------|----------|-------------------|
| 01 | `01_reuse_stack.py` | One workload, repeated. Each reuse tier catches a different kind of redundancy. | trie prefix KV, memo, semantic, layer KV, memory graph |
| 02 | `02_durable_agent.py` | Run to step 2, serialize to bytes, "crash", finish in a brand-new runtime. | `DurableAgent.run_until_step` / `resume_from` |
| 03 | `03_time_travel_fork.py` | Rewind a finished run, edit one step, replay the alternate timeline. | `DurableAgent.fork` |
| 04 | `04_semantic_support_cache.py` | Support queue where the same question arrives reworded five ways; the semantic tier deflects the paraphrases. | `SemanticCacheIndex`, `BruteForceEmbeddingProvider` |
| 05 | `05_ci_eval_replay.py` | Nightly eval suite: cases share a big grader preamble, and re-runs ride the cache instead of the API budget. | `benchmark_deterministic_m1`, `run_session_benchmark` |
| 06 | `06_agent_fleet_prefix.py` | 64 chat sessions sharing one 3k-char system prompt send it once; projects the input-token bill. | `run_session_benchmark` (trie prefix KV) |
| 07 | `07_spot_eviction_resume.py` | Long pipeline on a preemptible node; a fresh pod resumes from the retained checkpoint, KV cache still warm. | `DurableAgent` + checkpoint retention |
| 08 | `08_prompt_ab_replay.py` | Offline prompt A/B: fork one checkpoint per variant so steps 1-3 replay and only the edited step recomputes. | `DurableAgent.fork` |
| 09 | `09_hybrid_tensor_token.py` | Local tensor ops next to token gen: libtorch/MLX op parity, then a tensor-op reranker that picks the doc to hand the LLM. | `run_tensor_op` (libtorch + MLX) |
| 10 | `10_prompt_optimization.py` | Tune a prompt program's wording and decision threshold against a metric on a train split (DSPy / TextGrad style). | `ct.Param`, `ct.nn.Module`, `ct.Optimizer` |

## `milestones/`

Parity and training demos wired into the reproducibility check
(`scripts/bench.sh` -> `benchmarks/scripts/run_examples.py`): deterministic
reuse, a tiny transformer, a HotpotQA-style optimizer sweep over five seeds,
paired backend benchmarks, and a libtorch training run.
