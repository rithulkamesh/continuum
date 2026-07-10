import os, sys, json, time

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_isolated

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PARAPHRASES = [
    "Continuum optimizes KV cache for efficient LLM computation reuse",
    "Efficient computation reuse in LLM through Continuum's KV cache optimization",
    "KV cache optimization enables efficient LLM computation reuse via Continuum",
    "Continuum enables efficient computation reuse through KV cache for LLM inference",
    "Efficient LLM computation reuse is achieved via Continuum KV cache optimization",
    "Continuum's KV cache design provides efficient computation reuse for LLM workloads",
    "Through KV cache optimization, Continuum achieves efficient LLM computation reuse",
    "LLM computation reuse is enabled by Continuum's efficient KV cache optimization",
    "Continuum optimizes KV cache to enable efficient computation reuse in LLM systems",
    "Efficient computation reuse in LLM is achieved through Continuum's KV cache optimization",
]

print("=" * 60)
print("E2: Semantic cache reuse (isolated)")
print("=" * 60)
print(f"Trials: {len(PARAPHRASES)} (all unique, NO exact repeats)")
print(f"Shared prefix: 0 (no shared prefix to prevent trie)")
print(f"Enabled: semantic ONLY (memo=off, trie=off)")
print(f"Semantic threshold: 0.80")
print(f"Expected: trial 1 = backend call, trials 2+ = semantic hit (if sim >= threshold)")
print(f"Caveat: BruteForceEmbedding is character n-gram based, not truly semantic")
print()

results = {
    "experiment": "E2: Semantic cache reuse (isolated)",
    "mechanism": "semantic",
    "shared_prefix_tokens": 0,
    "n_trials": len(PARAPHRASES),
    "enable_memo": False,
    "enable_semantic": True,
    "enable_trie": False,
    "semantic_threshold": 0.80,
    "note": "No shared prefix. Memo is disabled. Trie is disabled (fresh KVCacheIndex per trial). "
    "Only SemanticCacheIndex is active. BruteForceEmbedding uses character n-gram hashing, "
    "so similarity depends on textual overlap, not meaning. Paraphrases share domain terms.",
}

t_start = time.time()

print("-" * 60)
print("Running isolated semantic benchmark with Azure...")
try:
    raw = benchmark_azure_isolated(
        prompts=PARAPHRASES,
        shared_prompt_tokens=0,
        enable_memo=False,
        enable_semantic=True,
        enable_trie=False,
        semantic_threshold=0.80,
    )

    results["raw"] = {
        "total_backend_calls": raw["total_backend_calls"],
        "total_memo_hits": raw["total_memo_hits"],
        "total_semantic_hits": raw["total_semantic_hits"],
        "total_trie_hits": raw["total_trie_hits"],
        "memo_table_size": raw["memo_table_size"],
        "semantic_cache_size": raw["semantic_cache_size"],
        "trie_cache_size": raw["trie_cache_size"],
        "per_trial": list(raw["per_trial"]),
    }

    for row in raw["per_trial"]:
        t = row["trial"]
        print(
            f"  Trial {t:2d}: lat={row['latency_ms']:6.0f}ms "
            f"memo={int(row['memo_hit'])} sem={int(row['semantic_hit'])} "
            f"trie={int(row['trie_hit'])} backend={int(row['backend_called'])} "
            f"tok_saved={row['tokens_saved']:5d} tok_proc={row['tokens_processed']:5d} "
            f"| {row['prompt'][:50]}"
        )

except Exception as exc:
    results["error"] = str(exc)
    import traceback

    traceback.print_exc()

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e2_isolated.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("-" * 60)
r = results.get("raw", {})
print(f"  Backend calls:       {r.get('total_backend_calls', '?')}")
print(f"  Memo hits:           {r.get('total_memo_hits', '?')}")
print(f"  Semantic hits:       {r.get('total_semantic_hits', '?')}")
print(f"  Trie hits:           {r.get('total_trie_hits', '?')}")
print(f"  Semantic cache size: {r.get('semantic_cache_size', '?')}")
print(f"  Wall time:           {results['wall_time_s']:.1f}s")
print(f"  Results saved to:    {out_path}")
