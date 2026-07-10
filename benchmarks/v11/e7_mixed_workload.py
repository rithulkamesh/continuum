import os, sys, json, time

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_isolated

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PREFIX = "S" * 3000

STEPS = [
    {"id": 1, "category": "prefix_reuse", "prompt": "What is machine learning?"},
    {"id": 2, "category": "prefix_reuse", "prompt": "Explain neural networks in one paragraph"},
    {"id": 3, "category": "prefix_reuse", "prompt": "How does backpropagation work?"},
    {"id": 4, "category": "prefix_reuse", "prompt": "Describe gradient descent optimization"},
    {"id": 5, "category": "prefix_reuse", "prompt": "What is a loss function in training?"},
    {"id": 6, "category": "prefix_reuse", "prompt": "Explain activation functions like ReLU"},
    {"id": 7, "category": "prefix_reuse", "prompt": "How do transformers process sequences?"},
    {"id": 8, "category": "prefix_reuse", "prompt": "What is the attention mechanism?"},
    {"id": 9, "category": "memo_reuse", "prompt": "Explain neural networks in one paragraph"},
    {"id": 10, "category": "memo_reuse", "prompt": "Describe gradient descent optimization"},
    {"id": 11, "category": "memo_reuse", "prompt": "Explain activation functions like ReLU"},
    {"id": 12, "category": "memo_reuse", "prompt": "What is the attention mechanism?"},
    {"id": 13, "category": "semantic_reuse", "prompt": "Describe how machines learn from data"},
    {
        "id": 14,
        "category": "semantic_reuse",
        "prompt": "What is the process of back-propagating errors?",
    },
    {
        "id": 15,
        "category": "semantic_reuse",
        "prompt": "Training objectives and loss functions explained",
    },
    {
        "id": 16,
        "category": "semantic_reuse",
        "prompt": "How attention allows transformers to focus on input",
    },
    {"id": 17, "category": "no_reuse", "prompt": "Write a haiku about autumn leaves"},
    {"id": 18, "category": "no_reuse", "prompt": "Calculate 15 times 27 plus 432"},
    {"id": 19, "category": "no_reuse", "prompt": "What is the capital of New Zealand?"},
    {"id": 20, "category": "no_reuse", "prompt": "Translate the word serenity to French"},
]

PROMPTS = []
for s in STEPS:
    if s["category"] in ("prefix_reuse", "memo_reuse", "semantic_reuse"):
        PROMPTS.append(PREFIX + "\nQuestion: " + s["prompt"] + ".")
    else:
        PROMPTS.append("Question: " + s["prompt"] + ".")

EXPECTED = {
    "prefix_reuse": {"memo": False, "semantic": False, "trie": True, "backend": True},
    "memo_reuse": {"memo": True, "semantic": False, "trie": False, "backend": False},
    "semantic_reuse": {"memo": False, "semantic": False, "trie": True, "backend": True},
    "no_reuse": {"memo": False, "semantic": False, "trie": False, "backend": True},
}

cats = {}
for s in STEPS:
    cats.setdefault(s["category"], []).append(s["id"])

print("=" * 60)
print("E7: Mixed workload (all mechanisms enabled)")
print("=" * 60)
print(f"Steps: {len(STEPS)}")
for cat, ids in cats.items():
    print(f"  {cat:16s}: {len(ids)} steps ({ids})")
print(f"Shared prefix: 3000 chars (steps 1-16 only)")
print(f"Steps 17-20: no prefix (completely different domain)")
print(
    f"Mechanisms: memo=ON, semantic=OFF (BruteForceEmbedding is n-gram based, "
    "not meaningfully semantic), trie=ON"
)
print(f"Priority order: memo > trie > backend")
print()

results = {
    "experiment": "E7: Mixed workload (memo + trie)",
    "n_steps": len(STEPS),
    "workload": STEPS,
    "enable_memo": True,
    "enable_semantic": False,
    "enable_trie": True,
    "shared_prefix_tokens": 3000,
    "semantic_disabled_reason": "BruteForceEmbedding uses character n-gram hashing. "
    "With a shared 3000-char prefix, all prompts get similarity=1.0 regardless of "
    "suffix content. A real embedding model (e.g. text-embedding-3-small) would be "
    "needed for true semantic reuse. Semantic is disabled here to isolate memo and trie.",
    "note": "Simulates a realistic agent workflow: shared system prompt with varied suffixes, "
    "occasional repeated subtasks, occasional paraphrased queries, and queries outside "
    "the main topic domain. Memo and trie are enabled with priority order: "
    "memo > trie > backend.",
}

t_start = time.time()

print("-" * 60)
print("Running mixed workload with Azure...")
try:
    raw = benchmark_azure_isolated(
        prompts=PROMPTS,
        shared_prompt_tokens=0,
        enable_memo=True,
        enable_semantic=False,
        enable_trie=True,
        semantic_threshold=0.80,
    )

    trial_data = list(raw["per_trial"])
    latencies = [r["latency_ms"] for r in trial_data]

    results["raw"] = {
        "total_backend_calls": raw["total_backend_calls"],
        "total_memo_hits": raw["total_memo_hits"],
        "total_semantic_hits": raw["total_semantic_hits"],
        "total_trie_hits": raw["total_trie_hits"],
        "memo_table_size": raw["memo_table_size"],
        "semantic_cache_size": raw["semantic_cache_size"],
        "trie_cache_size": raw["trie_cache_size"],
        "per_trial": trial_data,
    }

    sorted_lat = sorted(latencies)
    results["metrics"] = {
        "total_backend_calls": raw["total_backend_calls"],
        "total_steps": len(STEPS),
        "backend_call_rate": raw["total_backend_calls"] / len(STEPS),
        "memo_hit_count": raw["total_memo_hits"],
        "semantic_hit_count": raw["total_semantic_hits"],
        "trie_hit_count": raw["total_trie_hits"],
        "memo_hit_rate": raw["total_memo_hits"] / len(STEPS),
        "semantic_hit_rate": raw["total_semantic_hits"] / len(STEPS),
        "trie_hit_rate": raw["total_trie_hits"] / len(STEPS),
        "median_latency_ms": sorted_lat[len(sorted_lat) // 2],
        "p95_latency_ms": sorted_lat[int(len(sorted_lat) * 0.95)]
        if len(sorted_lat) > 1
        else sorted_lat[0],
        "total_tokens_saved": sum(r["tokens_saved"] for r in trial_data),
        "total_tokens_processed": sum(r["tokens_processed"] for r in trial_data),
    }
    total_tokens = (
        results["metrics"]["total_tokens_saved"] + results["metrics"]["total_tokens_processed"]
    )
    results["metrics"]["token_reduction_ratio"] = (
        results["metrics"]["total_tokens_saved"] / total_tokens if total_tokens > 0 else 0.0
    )

    cat_stats = {}
    for i, row in enumerate(trial_data):
        cat = STEPS[i]["category"]
        cs = cat_stats.setdefault(
            cat, {"memo": 0, "semantic": 0, "trie": 0, "backend": 0, "latencies": []}
        )
        if row["memo_hit"]:
            cs["memo"] += 1
        if row["semantic_hit"]:
            cs["semantic"] += 1
        if row["trie_hit"]:
            cs["trie"] += 1
        if row["backend_called"]:
            cs["backend"] += 1
        cs["latencies"].append(row["latency_ms"])
    results["by_category"] = {}
    for cat, cs in cat_stats.items():
        results["by_category"][cat] = {
            "n": len(cs["latencies"]),
            "memo_hits": cs["memo"],
            "semantic_hits": cs["semantic"],
            "trie_hits": cs["trie"],
            "backend_calls": cs["backend"],
            "mean_latency_ms": sum(cs["latencies"]) / len(cs["latencies"]),
        }

    print(
        f"  {'ID':>3s}  {'Category':16s}  {'Lat':>7s}  {'Memo':>4s}  {'Sem':>3s}  {'Trie':>4s}  {'BE':>2s}  {'TokSav':>6s}  {'TokProc':>7s}  Prompt"
    )
    print(
        f"  {'---':>3s}  {'--------':16s}  {'---':>7s}  {'----':>4s}  {'---':>3s}  {'----':>4s}  {'--':>2s}  {'------':>6s}  {'-------':>7s}  ------"
    )
    for i, row in enumerate(trial_data):
        s = STEPS[i]
        print(
            f"  {s['id']:3d}  {s['category']:16s}  {row['latency_ms']:7.0f}  "
            f"{int(row['memo_hit']):4d}  {int(row['semantic_hit']):3d}  "
            f"{int(row['trie_hit']):4d}  {int(row['backend_called']):2d}  "
            f"{row['tokens_saved']:6d}  {row['tokens_processed']:7d}  "
            f"{s['prompt'][:45]}"
        )

except Exception as exc:
    results["error"] = str(exc)
    import traceback

    traceback.print_exc()

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e7_mixed.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print()
print("-" * 60)
m = results.get("metrics", {})
bc = results.get("by_category", {})
print(
    f"  Backend calls:       {m.get('total_backend_calls', '?')} / {m.get('total_steps', '?')} ({m.get('backend_call_rate', 0):.0%})"
)
print(f"  Memo hits:           {m.get('memo_hit_count', '?')} ({m.get('memo_hit_rate', 0):.0%})")
print(
    f"  Semantic hits:       {m.get('semantic_hit_count', '?')} ({m.get('semantic_hit_rate', 0):.0%})"
)
print(f"  Trie hits:           {m.get('trie_hit_count', '?')} ({m.get('trie_hit_rate', 0):.0%})")
print(f"  Token reduction:     {m.get('token_reduction_ratio', 0):.1%}")
print(f"  Median latency:      {m.get('median_latency_ms', 0):.0f} ms")
print(f"  P95 latency:         {m.get('p95_latency_ms', 0):.0f} ms")
print()
print("  By category:")
for cat in ["prefix_reuse", "memo_reuse", "semantic_reuse", "no_reuse"]:
    c = bc.get(cat, {})
    exp = EXPECTED.get(cat, {})
    n = c.get("n", 0)
    print(
        f"    {cat:16s} n={n}: memo={c.get('memo_hits', 0)} sem={c.get('semantic_hits', 0)} "
        f"trie={c.get('trie_hits', 0)} be={c.get('backend_calls', 0)} "
        f"(expected: memo={int(exp['memo'])} sem={int(exp['semantic'])} "
        f"trie={int(exp['trie'])} be={int(exp['backend'])})"
    )
print(f"  Wall time:           {results['wall_time_s']:.1f}s")
print(f"  Results saved to:    {out_path}")
