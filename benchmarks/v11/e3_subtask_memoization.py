import os, sys, json, time

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_isolated

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

TOOL_CALLS = [
    "search_database(query='SELECT * FROM users WHERE active=true')",
    "lookup_cache(key='session_token_abc123')",
    "get_weather(city='San Francisco', units='celsius')",
    "calculate_hash(input='continuum_v1.1_release', algorithm='sha256')",
    "validate_token(token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9')",
]

PROMPTS = TOOL_CALLS + TOOL_CALLS

print("=" * 60)
print("E3: Memo / exact-match reuse (isolated)")
print("=" * 60)
print(f"Unique inputs: {len(TOOL_CALLS)}")
print(f"Trials: {len(PROMPTS)} ({TOOL_CALLS[0][:30]}... x2)")
print(f"Shared prefix: 0 (no shared prefix to prevent trie)")
print(f"Enabled: memo ONLY (semantic=off, trie=off)")
print(f"Expected: first occurrence = backend call, repeat = memo hit (backend skipped)")
print()

results = {
    "experiment": "E3: Memo / exact-match reuse (isolated)",
    "mechanism": "memo",
    "shared_prefix_tokens": 0,
    "n_trials": len(PROMPTS),
    "unique_inputs": len(TOOL_CALLS),
    "enable_memo": True,
    "enable_semantic": False,
    "enable_trie": False,
    "note": "No shared prefix. Semantic is disabled. Trie is disabled (fresh KVCacheIndex per trial). "
    "Only MemoTable is active. First occurrence of each input hits the backend. "
    "Second occurrence is an exact match in MemoTable -> backend skipped.",
}

t_start = time.time()

print("-" * 60)
print("Running isolated memo benchmark with Azure...")
try:
    raw = benchmark_azure_isolated(
        prompts=PROMPTS,
        shared_prompt_tokens=0,
        enable_memo=True,
        enable_semantic=False,
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
        tag = "FIRST" if t <= len(TOOL_CALLS) else "REPEAT"
        print(
            f"  Trial {t:2d} [{tag:6s}]: lat={row['latency_ms']:6.0f}ms "
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

out_path = os.path.join(DATA_DIR, "e3_isolated.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("-" * 60)
r = results.get("raw", {})
print(f"  Backend calls:    {r.get('total_backend_calls', '?')} (should be {len(TOOL_CALLS)})")
print(f"  Memo hits:        {r.get('total_memo_hits', '?')} (should be {len(TOOL_CALLS)})")
print(f"  Semantic hits:    {r.get('total_semantic_hits', '?')} (should be 0)")
print(f"  Trie hits:        {r.get('total_trie_hits', '?')} (should be 0)")
print(f"  Memo table size:  {r.get('memo_table_size', '?')}")
print(f"  Wall time:        {results['wall_time_s']:.1f}s")
print(f"  Results saved to: {out_path}")
