import json, os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "v11")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load(name):
    with open(os.path.join(DATA_DIR, f"{name}.json")) as f:
        return json.load(f)


def plot_e1():
    d = load("e1")
    s = d["summary"]
    trials = d.get("per_trial", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if trials:
        x = np.arange(len(trials))
        lats = [t["latency_no_cache_ms"] for t in trials]
        colors = ["#e74c3c" if t["memo_hits"] == 0 else "#2ecc71" for t in trials]
        ax1.bar(x, lats, color=colors, alpha=0.8)
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title(f"E1: Session Latency\nRed = Azure, Green = Memo Hit")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"T{t['trial']}" for t in trials])

    hits = ["memo", "semantic", "trie"]
    counts = [
        s.get("total_memo_hits", 0),
        s.get("total_semantic_hits", 0),
        s.get("total_trie_hits", 0),
    ]
    ax2.bar(hits, counts, color=["#2ecc71", "#3498db", "#f39c12"], alpha=0.8)
    ax2.set_ylabel("Hits")
    ax2.set_title(f"v1.1 Cache Hits\nMemo={counts[0]}, Sem={counts[1]}, Trie={counts[2]}")
    for i, v in enumerate(counts):
        if v > 0:
            ax2.text(i, v + 0.1, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e1_session_latency.png"))
    plt.close(fig)


def plot_e2():
    d = load("e2")
    per_q = d.get("per_question", [])

    if not per_q:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = [f"Q{q['question_idx']}" for q in per_q]
    memo_h = [q["total_memo_hits"] for q in per_q]
    sem_h = [q["total_semantic_hits"] for q in per_q]
    trie_h = [q["total_trie_hits"] for q in per_q]

    x = np.arange(len(labels))
    w = 0.25
    ax1.bar(x - w, memo_h, w, label="Memo", color="#2ecc71", alpha=0.8)
    ax1.bar(x, sem_h, w, label="Semantic", color="#3498db", alpha=0.8)
    ax1.bar(x + w, trie_h, w, label="Trie", color="#f39c12", alpha=0.8)
    ax1.set_xlabel("Question")
    ax1.set_ylabel("Hits")
    ax1.set_title("E2: Cache Hits per Question")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)

    lats = [q["mean_latency_ms"] for q in per_q]
    ax2.bar(labels, lats, color="#9b59b6", alpha=0.8)
    ax2.set_ylabel("Mean Latency (ms)")
    ax2.set_title("E2: Mean Latency per Question")
    for i, v in enumerate(lats):
        ax2.text(i, v + 10, f"{v:.0f}ms", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e2_semantic_hits.png"))
    plt.close(fig)


def plot_e3():
    d = load("e3")
    per_call = d.get("per_call", [])

    if not per_call:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    valid = [c for c in per_call if "error" not in c]
    names = [c["tool_call"][:30] for c in valid]
    first = [c["first_call_latency_ms"] for c in valid]
    cached = [c["mean_cached_latency_ms"] for c in valid]
    tok_red = [c["mean_token_reduction_pct"] for c in valid]

    x = np.arange(len(names))
    ax1.bar(x - 0.15, first, 0.3, label="First (Azure)", color="#e74c3c", alpha=0.8)
    ax1.bar(x + 0.15, cached, 0.3, label="Cached (Memo)", color="#2ecc71", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("E3: Tool Call Latency")
    ax1.legend(fontsize=8)

    ax2.bar(names, [v * 100 for v in tok_red], color="#3498db", alpha=0.8)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Token Reduction (%)")
    ax2.set_title("E3: Token Reduction")
    ax2.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e3_memoization.png"))
    plt.close(fig)


def plot_e4():
    d = load("e4")
    sessions = d.get("sessions", [])
    if not sessions:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    session_means = []
    session_labels = []
    for s in sessions:
        valid = [q for q in s["questions"] if "error" not in q]
        if valid:
            session_means.append(statistics.mean([q["mean_token_reduction_pct"] for q in valid]))
            session_labels.append(f"S{s['session']}")

    if session_means:
        ax1.bar(session_labels, [v * 100 for v in session_means], color="#2ecc71", alpha=0.8)
        ax1.set_ylabel("Mean Token Reduction (%)")
        ax1.set_title("E4: Token Reduction by Session")
        ax1.set_ylim(0, 110)
        for i, v in enumerate(session_means):
            ax1.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=8)

    all_first = []
    all_cached = []
    all_labels = []
    for s in sessions:
        for q in s["questions"]:
            if "error" not in q:
                all_first.append(q["first_call_latency_ms"])
                all_cached.append(q["mean_cached_latency_ms"])
                all_labels.append(f"S{s['session']}-Q{q['question'][:20]}")

    if all_labels:
        x = np.arange(len(all_labels))
        ax2.bar(x - 0.15, all_first, 0.3, label="First", color="#e74c3c", alpha=0.8)
        ax2.bar(x + 0.15, all_cached, 0.3, label="Cached", color="#2ecc71", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_labels, rotation=90, fontsize=5)
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("E4: First vs Cached Latency")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e4_multi_session.png"))
    plt.close(fig)


def plot_e5():
    d = load("e5")
    config_results = d.get("config_results", [])
    if not config_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = [
        f"max={c['config']['max_entries']}\nttl={c['config']['ttl_ms']}ms" for c in config_results
    ]
    hit_rates = [c["hit_rate"] * 100 for c in config_results]
    avg_get = [c["avg_get_ns"] for c in config_results]

    ax1.bar(labels, hit_rates, color="#3498db", alpha=0.8)
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_title("E5: FutureCache Hit Rate\nby Configuration")
    for i, v in enumerate(hit_rates):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax1.tick_params(axis="x", labelsize=7)

    ax2.bar(labels, avg_get, color="#9b59b6", alpha=0.8)
    ax2.set_ylabel("Avg Get Latency (ns)")
    ax2.set_title("E5: FutureCache Get Latency")
    ax2.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e5_prefetch.png"))
    plt.close(fig)


def plot_e6():
    d = load("e6")
    overhead = d.get("overhead_comparison", [])
    ablation_results = d.get("ablation_results", [])

    if not overhead and not ablation_results:
        return

    n_plots = 0
    if overhead:
        n_plots += 1
    if ablation_results:
        n_plots += 1

    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(10, 4))

    ax_idx = 0
    if overhead:
        ax = axes[ax_idx] if n_plots > 1 else axes
        names = [o["subsystem"] for o in overhead]
        avg_ns = [o["avg_construct_ns"] for o in overhead]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        ax.barh(names, avg_ns, color=colors[: len(names)], alpha=0.8)
        ax.set_xlabel("Avg Construction Time (ns)")
        ax.set_title("E6: v1.1 Subsystem Overhead")
        for i, v in enumerate(avg_ns):
            ax.text(v + 0.5, i, f"{v:.0f}ns", va="center", fontsize=8)
        ax_idx += 1

    if ablation_results:
        azure_results = [r for r in ablation_results if r.get("azure") is not None]
        if azure_results:
            ax = axes[ax_idx] if n_plots > 1 else axes
            names = [r["ablation"] for r in azure_results]
            tok_reds = [r["azure"]["mean_token_reduction_pct"] * 100 for r in azure_results]
            ax.bar(names, tok_reds, color="#1abc9c", alpha=0.8)
            ax.set_ylabel("Token Reduction (%)")
            ax.set_title("E6: Ablation Results")
            for i, v in enumerate(tok_reds):
                ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "e6_ablation.png"))
    plt.close(fig)


def plot_summary():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Continuum v1.1 Benchmark Summary (E1-E6)", fontsize=14, fontweight="bold")

    try:
        d1 = load("e1")
        s = d1.get("summary", {})
        ax = axes[0, 0]
        if s:
            categories = ["First\n(Azure)", "Cached\n(Memo)"]
            values = [s.get("first_call_latency_ms", 0), s.get("mean_cached_latency_ms", 0)]
            ax.bar(categories, values, color=["#e74c3c", "#2ecc71"], alpha=0.8)
            ax.set_ylabel("ms")
            ax.set_title(f"E1: Shared Prefix\nMemo: {s.get('total_memo_hits', 0)} hits")
    except Exception:
        axes[0, 0].text(0.5, 0.5, "E1: No data", ha="center", va="center")
        axes[0, 0].set_title("E1: Shared Prefix")

    try:
        d2 = load("e2")
        s = d2.get("summary", {})
        ax = axes[0, 1]
        if s:
            categories = ["Memo", "Semantic", "Trie"]
            values = [
                s.get("total_memo_hits", 0),
                s.get("total_semantic_hits", 0),
                s.get("total_trie_hits", 0),
            ]
            ax.bar(categories, values, color=["#2ecc71", "#3498db", "#f39c12"], alpha=0.8)
            ax.set_title(f"E2: Semantic Cache\nSem hit: {s.get('semantic_hit_rate', 0):.0%}")
    except Exception:
        axes[0, 1].text(0.5, 0.5, "E2: No data", ha="center", va="center")
        axes[0, 1].set_title("E2: Semantic Cache")

    try:
        d3 = load("e3")
        s = d3.get("summary", {})
        ax = axes[0, 2]
        if s:
            categories = ["First\n(Azure)", "Cached\n(Memo)"]
            values = [s.get("mean_first_call_ms", 0), s.get("mean_cached_latency_ms", 0)]
            ax.bar(categories, values, color=["#e74c3c", "#2ecc71"], alpha=0.8)
            ax.set_ylabel("ms")
            ax.set_title(
                f"E3: Memoization\n{s.get('mean_token_reduction_pct', 0):.0%} tok reduction"
            )
    except Exception:
        axes[0, 2].text(0.5, 0.5, "E3: No data", ha="center", va="center")
        axes[0, 2].set_title("E3: Memoization")

    try:
        d4 = load("e4")
        s = d4.get("summary", {})
        ax = axes[1, 0]
        if s and "per_session_mean_token_reduction" in s:
            labels = [f"S{i + 1}" for i in range(len(s["per_session_mean_token_reduction"]))]
            values = [v * 100 for v in s["per_session_mean_token_reduction"]]
            ax.bar(labels, values, color="#9b59b6", alpha=0.8)
            ax.set_ylabel("Token Reduction (%)")
            ax.set_title(f"E4: Multi-Session\nCV: {s.get('cross_session_consistency', 0):.2%}")
    except Exception:
        axes[1, 0].text(0.5, 0.5, "E4: No data", ha="center", va="center")
        axes[1, 0].set_title("E4: Multi-Session")

    try:
        d5 = load("e5")
        s = d5.get("summary", {})
        ax = axes[1, 1]
        overhead = d5.get("overhead_comparison", [])
        if overhead:
            names = [o["subsystem"] for o in overhead]
            times = [o["avg_construct_ns"] for o in overhead]
            ax.barh(names, times, color="#1abc9c", alpha=0.8)
            ax.set_xlabel("ns")
            ax.set_title("E5: FutureCache\nOverhead")
    except Exception:
        axes[1, 1].text(0.5, 0.5, "E5: No data", ha="center", va="center")
        axes[1, 1].set_title("E5: FutureCache")

    try:
        d6 = load("e6")
        overhead = d6.get("overhead_comparison", [])
        ax = axes[1, 2]
        if overhead:
            names = [o["subsystem"] for o in overhead]
            times = [o["avg_construct_ns"] for o in overhead]
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
            ax.barh(names, times, color=colors[: len(names)], alpha=0.8)
            ax.set_xlabel("ns")
            ax.set_title("E6: Subsystem\nOverhead")
    except Exception:
        axes[1, 2].text(0.5, 0.5, "E6: No data", ha="center", va="center")
        axes[1, 2].set_title("E6: Ablation")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOT_DIR, "summary_dashboard.png"))
    plt.close(fig)


if __name__ == "__main__":
    import statistics

    plot_e1()
    plot_e2()
    plot_e3()
    plot_e4()
    plot_e5()
    plot_e6()
    plot_summary()
    print(f"Plots saved to {PLOT_DIR}/")
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")
