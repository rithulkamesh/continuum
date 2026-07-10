import os, sys, json, time, statistics

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_with_prompt, SemanticCacheIndex

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PARAPHRASES = [
    "What is the Continuum framework and its main purpose?",
    "Can you describe the Continuum system to me?",
    "Tell me about the Continuum project.",
    "How would you define what Continuum is?",
    "What does Continuum do exactly?",
    "Explain Continuum in your own words.",
    "Could you elaborate on what Continuum is?",
    "What is Continuum all about?",
    "Give me a summary of Continuum.",
    "How does Continuum work at a high level?",
    "What are the main features of Continuum?",
    "Describe the Continuum architecture briefly.",
    "What problem does Continuum solve?",
    "Outline Continuum's key components.",
    "What is the purpose of Continuum?",
    "Break down what Continuum is.",
    "Provide an overview of Continuum.",
    "What can you tell me about Continuum?",
    "Clarify what Continuum entails.",
    "Sum up Continuum briefly.",
]

DIFFERENT_QUESTIONS = [
    "How do neural networks learn from training data?",
    "What is the difference between TCP and UDP?",
    "Explain the CAP theorem in distributed systems.",
    "What is a hash table and how does it work?",
    "Describe the concept of garbage collection.",
    "How does a B-tree differ from a binary search tree?",
    "What is the difference between a process and a thread?",
    "Explain the concept of virtual memory.",
    "What is a deadlock and how can it be prevented?",
    "Describe the MapReduce programming model.",
    "How does DNS resolution work step by step?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain the concept of eventual consistency.",
    "What is a content delivery network and why is it used?",
    "How does public key cryptography work?",
    "What is the difference between a stack and a queue?",
    "Describe the observer design pattern.",
    "What is a bloom filter and when would you use one?",
    "How does the TLS handshake protocol work?",
    "What is the difference between compilation and interpretation?",
]

UNRELATED_QUESTIONS = [
    "What is the best recipe for chocolate chip cookies?",
    "How do you train a puppy to sit?",
    "What is the capital of Australia?",
    "Describe the lifecycle of a butterfly.",
    "How do plants perform photosynthesis?",
    "What are the rules of chess?",
    "How do you change a flat tire on a car?",
    "What is the history of the Olympic Games?",
    "How do you play the guitar?",
    "What causes the northern lights?",
    "Describe the water cycle in simple terms.",
    "What is the difference between a crocodile and an alligator?",
    "How do bees make honey?",
    "What are the main types of clouds?",
    "How do you grow tomatoes in a home garden?",
    "What is the tallest mountain in the world?",
    "How does the human respiratory system work?",
    "What are the primary colors and why?",
    "How do submarines work underwater?",
    "What is the speed of light in a vacuum?",
]

PREFIX_TOKENS = 3000
TRIALS_PER_QUESTION = 3

print("=" * 60)
print("E2: Semantic reuse with real Azure prompts")
print("=" * 60)
print(f"Prefix tokens: {PREFIX_TOKENS}")
print(f"Paraphrases: {len(PARAPHRASES)}")
print(f"Different questions: {len(DIFFERENT_QUESTIONS)}")
print(f"Unrelated questions: {len(UNRELATED_QUESTIONS)}")
print(f"Trials per question: {TRIALS_PER_QUESTION} (1 warmup + 2 data)")
print(
    f"Total Azure calls: {(len(PARAPHRASES) + len(DIFFERENT_QUESTIONS) + len(UNRELATED_QUESTIONS)) * TRIALS_PER_QUESTION * 3}"
)
print()

results = {
    "experiment": "E2: Semantic reuse (paraphrase detection with Azure)",
    "prefix_tokens": PREFIX_TOKENS,
    "trials_per_question": TRIALS_PER_QUESTION,
    "data_trials_per_question": TRIALS_PER_QUESTION - 1,
}

t_start = time.time()


def run_group(name, questions):
    print(f"  Running {name} ({len(questions)} questions)...")
    group_results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            raw = benchmark_azure_with_prompt(
                question=q,
                trials=TRIALS_PER_QUESTION,
                shared_prompt_tokens=PREFIX_TOKENS,
            )
            data_trials = [t for t in list(raw["per_trial"]) if not t["warmup"]]
            cached_lat = [t["latency_with_cache_ms"] for t in data_trials]
            uncached_lat = [t["latency_no_cache_ms"] for t in data_trials]
            tok_red = [t["token_reduction_pct"] for t in data_trials]
            speed = [t["latency_speedup"] for t in data_trials]

            entry = {
                "question": q,
                "n": len(data_trials),
                "mean_cached_ms": statistics.mean(cached_lat),
                "std_cached_ms": statistics.stdev(cached_lat) if len(cached_lat) > 1 else 0.0,
                "mean_uncached_ms": statistics.mean(uncached_lat),
                "std_uncached_ms": statistics.stdev(uncached_lat) if len(uncached_lat) > 1 else 0.0,
                "mean_token_reduction_pct": statistics.mean(tok_red),
                "mean_speedup": statistics.mean(speed),
                "per_trial": data_trials,
            }
            group_results.append(entry)
            elapsed = time.time() - t0
            if (i + 1) % 5 == 0:
                print(
                    f"    [{i + 1}/{len(questions)}] avg_tok_red={entry['mean_token_reduction_pct']:.1%}, avg_speedup={entry['mean_speedup']:.2f}x ({elapsed:.1f}s)"
                )
        except Exception as e:
            group_results.append({"question": q, "error": str(e)})
            print(f"    [{i + 1}/{len(questions)}] ERROR: {e}")
    print(f"  {name} done ({time.time() - t0:.1f}s)")
    return group_results


print("-" * 60)
paraphrase_results = run_group("Paraphrases", PARAPHRASES)
print()

different_results = run_group("Different questions", DIFFERENT_QUESTIONS)
print()

unrelated_results = run_group("Unrelated questions", UNRELATED_QUESTIONS)

results["paraphrases"] = paraphrase_results
results["different_questions"] = different_results
results["unrelated_questions"] = unrelated_results


def summarize_group(group):
    valid = [g for g in group if "error" not in g]
    if not valid:
        return {"n": 0, "error": "all failed"}
    tok_reds = [g["mean_token_reduction_pct"] for g in valid]
    speeds = [g["mean_speedup"] for g in valid]
    cached = [g["mean_cached_ms"] for g in valid]
    uncached = [g["mean_uncached_ms"] for g in valid]
    return {
        "n": len(valid),
        "mean_token_reduction_pct": statistics.mean(tok_reds),
        "std_token_reduction_pct": statistics.stdev(tok_reds) if len(tok_reds) > 1 else 0.0,
        "median_token_reduction_pct": statistics.median(tok_reds),
        "mean_speedup": statistics.mean(speeds),
        "median_speedup": statistics.median(speeds),
        "mean_cached_ms": statistics.mean(cached),
        "mean_uncached_ms": statistics.mean(uncached),
    }


results["summary"] = {
    "paraphrases": summarize_group(paraphrase_results),
    "different_questions": summarize_group(different_results),
    "unrelated_questions": summarize_group(unrelated_results),
    "note": "All groups use same prefix. Token reduction is based on prefix cache reuse, "
    "not semantic similarity. SemanticCacheIndex with BruteForceEmbedding is not "
    "truly semantic and cannot distinguish paraphrases from unrelated questions.",
}

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e2.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

s = results["summary"]
print("-" * 60)
for name in ["paraphrases", "different_questions", "unrelated_questions"]:
    g = s[name]
    if "error" in g:
        print(f"  {name}: ERROR")
    else:
        print(
            f"  {name}: tok_reduction={g['mean_token_reduction_pct']:.1%} (+/-{g['std_token_reduction_pct']:.1%}), speedup={g['mean_speedup']:.2f}x"
        )
print(f"  Wall time:       {results['wall_time_s']:.1f}s")
print(f"  Results saved to: {out_path}")
