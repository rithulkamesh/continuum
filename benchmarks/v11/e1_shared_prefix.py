import os, sys, json, time

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_isolated

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PREFIX_TOKENS = 3000

SUFFIXES = [
    "What is machine learning?",
    "Explain neural networks.",
    "How does backpropagation work?",
    "Describe gradient descent optimization.",
    "What is a loss function in training?",
    "Explain activation functions like ReLU.",
    "How do transformers process sequences?",
    "What is the attention mechanism?",
    "Describe batch normalization.",
    "What is dropout regularization?",
]

print("=" * 60)
print("E1: Trie / prefix reuse (isolated)")
print("=" * 60)
print(f"Trials: {len(SUFFIXES)} (all unique suffixes, NO repeats)")
print(f"Shared prefix: {PREFIX_TOKENS} chars")
print(f"Enabled: trie ONLY (memo=off, semantic=off)")
print(f"Expected: trial 1 = backend call, trials 2+ = trie hit + backend")
print()

results = {
    "experiment": "E1: Trie / prefix reuse (isolated)",
    "mechanism": "trie",
    "shared_prefix_tokens": PREFIX_TOKENS,
    "n_trials": len(SUFFIXES),
    "enable_memo": False,
    "enable_semantic": False,
    "enable_trie": True,
    "note": "Each trial uses a DIFFERENT suffix. Memo is disabled so no exact-match skip. "
    "Semantic is disabled. The KVCacheIndex trie matches the shared prefix on trials 2+, "
    "reducing tokens sent to the backend. Backend IS still called (with fewer tokens).",
}

t_start = time.time()

print("-" * 60)
print("Running isolated trie benchmark with Azure...")
try:
    raw = benchmark_azure_isolated(
        prompts=SUFFIXES,
        shared_prompt_tokens=PREFIX_TOKENS,
        enable_memo=False,
        enable_semantic=False,
        enable_trie=True,
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
            f"| {row['prompt'][:40]}"
        )

except Exception as exc:
    results["error"] = str(exc)
    import traceback

    traceback.print_exc()

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e1_isolated.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("-" * 60)
r = results.get("raw", {})
print(f"  Backend calls:    {r.get('total_backend_calls', '?')}")
print(f"  Memo hits:        {r.get('total_memo_hits', '?')}")
print(f"  Semantic hits:    {r.get('total_semantic_hits', '?')}")
print(f"  Trie hits:        {r.get('total_trie_hits', '?')}")
print(f"  Trie cache size:  {r.get('trie_cache_size', '?')}")
print(f"  Wall time:        {results['wall_time_s']:.1f}s")
print(f"  Results saved to: {out_path}")
