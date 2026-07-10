import os, sys, json, time, statistics

os.chdir("/Users/rkamesh/dev/continuum")
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")
from continuum._native import benchmark_azure_with_prompt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PREFIX_TOKENS = 3000
TRIALS = 11

SESSION_QUESTIONS = [
    "Summarize the Continuum cache system.",
    "What is the MemoTable used for in Continuum?",
    "How does speculative prefetching work in Continuum?",
]

NUM_SESSIONS = 3

print("=" * 60)
print("E4: Multi-session persistence")
print("=" * 60)
print(f"Sessions: {NUM_SESSIONS}")
print(f"Questions per session: {len(SESSION_QUESTIONS)}")
print(f"Trials per question: {TRIALS} (1 warmup + 10 data)")
print(f"Prefix tokens: {PREFIX_TOKENS}")
print(
    f"Azure API calls: {NUM_SESSIONS * len(SESSION_QUESTIONS)} "
    f"(1 per question per session, memo handles trials 2+)"
)
print()

results = {
    "experiment": "E4: Multi-session persistence (Interpreter path)",
    "prefix_tokens": PREFIX_TOKENS,
    "trials_per_question": TRIALS,
    "num_sessions": NUM_SESSIONS,
    "session_questions": SESSION_QUESTIONS,
    "azure_api_calls": NUM_SESSIONS * len(SESSION_QUESTIONS),
    "note": "Each benchmark_azure_with_prompt call creates an independent Session. "
    "There is NO cross-session cache sharing. Each session independently: "
    "trial 1 = Azure call, trials 2+ = memo hit.",
}

t_start = time.time()

print("-" * 60)
all_sessions = []
for session_idx in range(NUM_SESSIONS):
    print(f"  Session {session_idx + 1}/{NUM_SESSIONS}...")
    session_data = {"session": session_idx + 1, "questions": []}
    for q_idx, question in enumerate(SESSION_QUESTIONS):
        t0 = time.time()
        try:
            raw = benchmark_azure_with_prompt(
                question=question,
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

            q_entry = {
                "question": question,
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
            session_data["questions"].append(q_entry)
            elapsed = time.time() - t0
            print(
                f"    Q{q_idx + 1}: first={first_call:.0f}ms, "
                f"cached={q_entry['mean_cached_latency_ms']:.0f}ms, "
                f"tok_red={q_entry['mean_token_reduction_pct']:.1%} ({elapsed:.1f}s)"
            )
        except Exception as e:
            session_data["questions"].append({"question": question, "error": str(e)})
            print(f"    Q{q_idx + 1}: ERROR: {e}")
    all_sessions.append(session_data)

results["sessions"] = all_sessions

all_tok_reds_by_session = []
all_memo_by_session = []
for s in all_sessions:
    valid = [q for q in s["questions"] if "error" not in q]
    if valid:
        session_tok_reds = [q["mean_token_reduction_pct"] for q in valid]
        session_memos = [q["total_memo_hits"] for q in valid]
        all_tok_reds_by_session.append(statistics.mean(session_tok_reds))
        all_memo_by_session.append(sum(session_memos))

if all_tok_reds_by_session:
    results["summary"] = {
        "sessions_tested": len(all_tok_reds_by_session),
        "per_session_mean_token_reduction": all_tok_reds_by_session,
        "per_session_total_memo_hits": all_memo_by_session,
        "overall_mean_token_reduction_pct": statistics.mean(all_tok_reds_by_session),
        "std_token_reduction_pct": statistics.stdev(all_tok_reds_by_session)
        if len(all_tok_reds_by_session) > 1
        else 0.0,
        "cross_session_consistency": (
            statistics.stdev(all_tok_reds_by_session) / statistics.mean(all_tok_reds_by_session)
            if statistics.mean(all_tok_reds_by_session) > 0
            else float("inf")
        ),
        "note": "Each session creates independent MemoTable/KVCacheIndex. No cross-session "
        "persistence. Consistency measures whether memo behavior is reproducible across sessions.",
    }
else:
    results["summary"] = {"error": "all sessions failed"}

results["wall_time_s"] = time.time() - t_start

out_path = os.path.join(DATA_DIR, "e4.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

s = results.get("summary", {})
print("-" * 60)
if "error" in s:
    print(f"ERROR: {s['error']}")
else:
    print(f"  Sessions tested: {s['sessions_tested']}")
    print(
        f"  Per-session token reduction: {[f'{v:.1%}' for v in s['per_session_mean_token_reduction']]}"
    )
    print(
        f"  Overall mean token reduction: {s['overall_mean_token_reduction_pct']:.1%} (+/- {s['std_token_reduction_pct']:.1%})"
    )
    print(f"  Cross-session CV: {s['cross_session_consistency']:.2%}")
print(f"  Wall time:          {results['wall_time_s']:.1f}s")
print(f"  Results saved to:   {out_path}")
