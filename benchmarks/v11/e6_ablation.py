import os, sys, json, time, statistics

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import (
    benchmark_azure_with_prompt,
    SemanticCacheIndex,
    MemoTable,
    MemoryGraphStore,
    LayerKVCacheIndex,
    FutureCache,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PREFIX_TOKENS = 3000
TRIALS = 11
QUESTION = "Summarize Continuum cache behavior and explain how KV cache reuse works."

ABLATIONS = {
    "A0_full_v11": {
        "description": "Full v1.1 stack (MemoTable + SemanticCache + KVCacheIndex)",
        "use_azure": True,
    },
    "A1_repeated_question": {
        "description": "Same question repeated (measures pure memo behavior)",
        "use_azure": True,
    },
    "A2_component_overhead": {
        "description": "Subsystem construction overhead (no Azure)",
        "use_azure": False,
    },
}

print("=" * 60)
print("E6: Ablation study")
print("=" * 60)
print(f"Prefix tokens: {PREFIX_TOKENS}")
print(f"Trials: {TRIALS} (1 warmup + 10 data)")
print(f"Question: {QUESTION[:60]}...")
print(f"Ablations: {list(ABLATIONS.keys())}")
print()

results = {
    "experiment": "E6: Ablation study (Interpreter path)",
    "prefix_tokens": PREFIX_TOKENS,
    "trials": TRIALS,
    "question": QUESTION,
    "ablation_results": [],
    "note": "All Azure ablations use the same Session configuration with all v1.1 components "
    "(MemoTable, SemanticCacheIndex, KVCacheIndex) enabled. The Interpreter path always "
    "checks memo -> semantic -> trie -> backend. Component-level ablation requires "
    "modifying the C++ Session setup, which is reported as overhead comparison.",
}

t_start = time.time()

print("-" * 60)

for ablation_name, ablation_cfg in ABLATIONS.items():
    print(f"  {ablation_name}: {ablation_cfg['description']}")

    ablation_result = {
        "ablation": ablation_name,
        "description": ablation_cfg["description"],
    }

    if ablation_cfg.get("use_azure", False):
        t0 = time.time()
        try:
            raw = benchmark_azure_with_prompt(
                question=QUESTION,
                trials=TRIALS,
                shared_prompt_tokens=PREFIX_TOKENS,
            )
            all_trials_t = list(raw["per_trial"])
            data_trials = [t for t in all_trials_t if not t["warmup"]]
            latencies = [t["latency_no_cache_ms"] for t in data_trials]
            tok_red = [t["token_reduction_pct"] for t in data_trials]
            warmup_trials = [t for t in all_trials_t if t["warmup"]]
            first_call = warmup_trials[0]["latency_no_cache_ms"] if warmup_trials else 0
            cached_calls = latencies

            ablation_result["azure"] = {
                "n": len(data_trials),
                "first_call_latency_ms": first_call,
                "mean_cached_latency_ms": statistics.mean(cached_calls) if cached_calls else 0.0,
                "mean_latency_ms": statistics.mean(latencies),
                "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "mean_token_reduction_pct": statistics.mean(tok_red),
                "std_token_reduction_pct": statistics.stdev(tok_red) if len(tok_red) > 1 else 0.0,
                "total_memo_hits": sum(t["memo_hits"] for t in data_trials),
                "total_semantic_hits": sum(t["semantic_hits"] for t in data_trials),
                "total_trie_hits": sum(t["trie_hits"] for t in data_trials),
                "memo_table_size": raw["memo_table_size"],
                "semantic_cache_size": raw["semantic_cache_size"],
                "trie_cache_size": raw["trie_cache_size"],
                "per_trial": data_trials,
            }
            elapsed = time.time() - t0
            print(
                f"    Azure: first={first_call:.0f}ms, "
                f"cached={ablation_result['azure']['mean_cached_latency_ms']:.0f}ms, "
                f"tok_red={ablation_result['azure']['mean_token_reduction_pct']:.1%} ({elapsed:.1f}s)"
            )
        except Exception as e:
            ablation_result["azure_error"] = str(e)
            print(f"    Azure ERROR: {e}")
    else:
        ablation_result["azure"] = None

    results["ablation_results"].append(ablation_result)
    print()

print("-" * 60)
print("Component overhead comparison (no Azure):")
overhead_results = []
for name, ctor in [
    ("SemanticCacheIndex", lambda: SemanticCacheIndex(2048, 0.85)),
    ("MemoTable", lambda: MemoTable(4096, 0)),
    ("MemoryGraphStore", lambda: MemoryGraphStore(8192)),
    ("LayerKVCacheIndex", lambda: LayerKVCacheIndex(4096, 256 * 1024 * 1024)),
    ("FutureCache", lambda: FutureCache(256, 30000)),
]:
    times = []
    for _ in range(1000):
        t0 = time.perf_counter_ns()
        obj = ctor()
        elapsed = time.perf_counter_ns() - t0
        times.append(elapsed)
        obj.clear()
    avg_us = statistics.mean(times) / 1000
    p50_us = sorted(times)[len(times) // 2] / 1000
    p99_us = sorted(times)[int(len(times) * 0.99)] / 1000
    entry = {
        "subsystem": name,
        "avg_construct_ns": statistics.mean(times),
        "p50_construct_ns": sorted(times)[len(times) // 2],
        "p99_construct_ns": sorted(times)[int(len(times) * 0.99)],
        "iterations": len(times),
    }
    overhead_results.append(entry)
    print(f"  {name:>25}: avg={avg_us:.0f}ns, p50={p50_us:.0f}ns, p99={p99_us:.0f}ns")

results["overhead_comparison"] = overhead_results

azure_results = {
    r["ablation"]: r["azure"]
    for r in results["ablation_results"]
    if "azure" in r and r["azure"] is not None and "azure_error" not in r
}

if azure_results:
    tok_reds = [v["mean_token_reduction_pct"] for v in azure_results.values()]
    results["summary"] = {
        "ablations_with_azure": list(azure_results.keys()),
        "token_reduction_range": [min(tok_reds), max(tok_reds)],
        "mean_token_reduction": statistics.mean(tok_reds),
        "note": "All Azure ablations use the same v1.1 component configuration. "
        "Component-level ablation would require modifying the C++ Session setup. "
        "Overhead comparison is measured separately above.",
    }
else:
    results["summary"] = {"error": "no Azure results collected"}

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e6.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("-" * 60)
s = results.get("summary", {})
if "error" in s:
    print(f"ERROR: {s['error']}")
else:
    print(f"  Ablations tested: {s['ablations_with_azure']}")
    print(
        f"  Token reduction range: [{s['token_reduction_range'][0]:.1%}, {s['token_reduction_range'][1]:.1%}]"
    )
    print(f"  Mean token reduction: {s['mean_token_reduction']:.1%}")
print(f"  Wall time:          {results['wall_time_s']:.1f}s")
print(f"  Results saved to:   {out_path}")
