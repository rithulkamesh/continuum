import os, sys, json, time, statistics

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import FutureCache

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

WORKFLOW_STEPS = [
    "user_query",
    "search_results",
    "rag_context",
    "tool_call_1",
    "tool_result_1",
    "tool_call_2",
    "tool_result_2",
    "synthesis",
    "final_answer",
]

HIT_RATE_PREDICTIONS = {
    "user_query": 0.0,
    "search_results": 0.1,
    "rag_context": 0.2,
    "tool_call_1": 0.5,
    "tool_result_1": 0.6,
    "tool_call_2": 0.7,
    "tool_result_2": 0.8,
    "synthesis": 0.9,
    "final_answer": 0.95,
}

NUM_WORKFLOWS = 100
CACHE_CONFIGS = [
    {"max_entries": 64, "ttl_ms": 30000},
    {"max_entries": 256, "ttl_ms": 30000},
    {"max_entries": 1024, "ttl_ms": 30000},
    {"max_entries": 256, "ttl_ms": 5000},
    {"max_entries": 256, "ttl_ms": 60000},
]

print("=" * 60)
print("E5: Speculative prefetch (FutureCache get/put)")
print("=" * 60)
print(f"Workflow steps: {len(WORKFLOW_STEPS)}")
print(f"Simulated workflows: {NUM_WORKFLOWS}")
print(f"Cache configs: {len(CACHE_CONFIGS)}")
print()

results = {
    "experiment": "E5: Speculative prefetch (FutureCache get/put)",
    "workflow_steps": WORKFLOW_STEPS,
    "prediction_rates": HIT_RATE_PREDICTIONS,
    "num_workflows": NUM_WORKFLOWS,
    "cache_configs": CACHE_CONFIGS,
}

t_start = time.time()


def simulate_prefetch(config):
    fc = FutureCache(max_entries=config["max_entries"], ttl_ms=config["ttl_ms"])
    hits = 0
    misses = 0
    wasted_puts = 0
    correct_puts = 0
    put_times = []
    get_times = []

    for wf_idx in range(NUM_WORKFLOWS):
        for step_idx, step_name in enumerate(WORKFLOW_STEPS):
            pred_rate = HIT_RATE_PREDICTIONS[step_name]
            step_key = f"{step_name}_{wf_idx}"

            t0 = time.perf_counter_ns()
            cached = fc.has(step_key)
            t1 = time.perf_counter_ns()
            get_times.append(t1 - t0)

            if cached:
                t0 = time.perf_counter_ns()
                val = fc.get(step_key)
                t1 = time.perf_counter_ns()
                get_times.append(t1 - t0)
                if val is not None:
                    hits += 1
                else:
                    misses += 1
            else:
                misses += 1

            if step_idx < len(WORKFLOW_STEPS) - 1:
                next_step = WORKFLOW_STEPS[step_idx + 1]
                next_key = f"{next_step}_{wf_idx}"

                import random

                should_prefetch = random.random() < pred_rate

                if should_prefetch:
                    predicted_value = [wf_idx, step_idx + 1]
                    t0 = time.perf_counter_ns()
                    existing = fc.get(next_key)
                    if existing is None:
                        fc.put(next_key, predicted_value)
                        wasted_puts += 0
                        correct_puts += 1
                    else:
                        wasted_puts += 0
                    t1 = time.perf_counter_ns()
                    put_times.append(t1 - t0)

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "total_lookups": total,
        "hit_rate": hits / total if total > 0 else 0.0,
        "correct_prefetch_puts": correct_puts,
        "avg_put_ns": statistics.mean(put_times) if put_times else 0,
        "avg_get_ns": statistics.mean(get_times) if get_times else 0,
        "p50_get_ns": sorted(get_times)[len(get_times) // 2] if get_times else 0,
        "p99_get_ns": sorted(get_times)[int(len(get_times) * 0.99)] if get_times else 0,
        "cache_size": fc.size(),
    }


print("-" * 60)
config_results = []
for config in CACHE_CONFIGS:
    label = f"max={config['max_entries']},ttl={config['ttl_ms']}ms"
    print(f"  Testing {label}...")
    t0 = time.time()
    r = simulate_prefetch(config)
    elapsed = time.time() - t0
    r["config"] = config
    r["wall_time_ms"] = elapsed * 1000
    config_results.append(r)
    print(
        f"    hit_rate={r['hit_rate']:.1%}, avg_get={r['avg_get_ns']:.0f}ns, avg_put={r['avg_put_ns']:.0f}ns, size={r['cache_size']} ({elapsed:.2f}s)"
    )

results["config_results"] = config_results

no_prefetch_hits = 0
no_prefetch_total = 0
fc_no_prefetch = FutureCache(max_entries=256, ttl_ms=30000)
for wf_idx in range(NUM_WORKFLOWS):
    for step_idx, step_name in enumerate(WORKFLOW_STEPS):
        key = f"{step_name}_{wf_idx}"
        if fc_no_prefetch.has(key):
            no_prefetch_hits += 1
        no_prefetch_total += 1
        if step_idx < len(WORKFLOW_STEPS) - 1:
            next_step = WORKFLOW_STEPS[step_idx + 1]
            next_key = f"{next_step}_{wf_idx}"
            actual_value = [wf_idx, step_idx + 1]
            fc_no_prefetch.put(next_key, actual_value)

results["no_prefetch_baseline"] = {
    "hits": no_prefetch_hits,
    "total": no_prefetch_total,
    "hit_rate": no_prefetch_hits / no_prefetch_total if no_prefetch_total > 0 else 0.0,
    "note": "Baseline: put actual results only after computing them. Next workflow sees them.",
}

results["summary"] = {
    "best_hit_rate": max(r["hit_rate"] for r in config_results),
    "best_config": max(config_results, key=lambda r: r["hit_rate"])["config"],
    "avg_get_latency_ns": statistics.mean([r["avg_get_ns"] for r in config_results]),
    "avg_put_latency_ns": statistics.mean([r["avg_put_ns"] for r in config_results]),
    "prefetch_benefit_vs_no_prefetch": (
        max(r["hit_rate"] for r in config_results) - results["no_prefetch_baseline"]["hit_rate"]
    ),
    "note": "Hit rate depends on prediction accuracy. Higher prediction rates for later steps "
    "reflect that workflow patterns become more predictable. TTL and cache size affect "
    "hit rate for multi-workload scenarios.",
}

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e5.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

s = results["summary"]
print("-" * 60)
print(
    f"  Best hit rate:       {s['best_hit_rate']:.1%} (config: max={s['best_config']['max_entries']}, ttl={s['best_config']['ttl_ms']}ms)"
)
print(f"  Avg get latency:     {s['avg_get_latency_ns']:.0f} ns")
print(f"  Avg put latency:     {s['avg_put_latency_ns']:.0f} ns")
print(f"  No-prefetch hit rate: {results['no_prefetch_baseline']['hit_rate']:.1%}")
print(f"  Prefetch benefit:    +{s['prefetch_benefit_vs_no_prefetch']:.1%}")
print(f"  Wall time:           {results['wall_time_s']:.1f}s")
print(f"  Results saved to:    {out_path}")
