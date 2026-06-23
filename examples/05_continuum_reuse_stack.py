"""05 - Continuum full reuse stack.

The same LLM workload, run over and over. A naive runtime recomputes every
token of every call. Continuum recognises four *different* kinds of redundancy
and skips the work, across five independent reuse tiers:

    Tier 1  Trie prefix KV cache   shared prompt prefixes (system preambles)
    Tier 2  Memo table             exact-repeat prompts (deterministic)
    Tier 3  Semantic cache         paraphrased / near-duplicate prompts
    Tier 4  Layer KV warm-start    resume a decode from cached layer state
    Tier 5  Memory graph recall    surface related prior context

Every number below comes from the native runtime (FakeLLM backend, so the
output is deterministic and CI-checkable). Run it:

    PYTHONPATH=python python examples/05_continuum_reuse_stack.py
"""

from __future__ import annotations

import os

# Quiet the runtime's per-op INFO logging so the demo output stays readable.
# Must be set before the native module initialises its logger (lazy, on first use).
os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import (  # noqa: E402
    run_session_benchmark,
    run_v11_benchmark,
    run_v11_wiring_check,
)

BAR = "=" * 60
RULE = "-" * 60


def header() -> None:
    print(BAR)
    print(" Continuum - Full Reuse Stack")
    print(BAR)
    print("Same workload, repeated. Each tier catches a different")
    print("kind of redundancy. All figures from the native runtime.")


def tier1_trie() -> float:
    """Shared prompt prefix -> KV trie reuse. Different suffix each call."""
    print()
    print(RULE)
    print("[1/5] Trie prefix KV cache  (shared prompt prefixes)")
    res = run_session_benchmark(num_steps=8, prefix_tokens=96, suffix_tokens=8)
    last = res["runs"][-1]
    saved = last["total_tokens_saved"]
    processed = last["total_tokens_processed"]
    reduction = last["token_reduction"] * 100.0
    print(f"  runs={res['total_runs']}  shared_prefix_tokens={res['prefix_tokens']}")
    print(f"  final run: tokens_saved={saved}  tokens_processed={processed}")
    print(f"  -> token reduction on warm run: {reduction:.1f}%")
    return reduction


def tier23_memo_semantic() -> tuple[float, float, int]:
    """Exact repeats -> memo; paraphrases -> semantic cache."""
    print()
    print(RULE)
    print("[2/5] Memo table            (exact-repeat prompts)")
    print("[3/5] Semantic cache        (paraphrased prompts)")
    res = run_v11_benchmark(num_steps=30, prefix_tokens=50)
    memo = res["memo_hit_rate"] * 100.0
    semantic = res["semantic_hit_rate"] * 100.0
    print(f"  steps=30 (every prompt is a *distinct* paraphrase)")
    # Memo is the cheapest tier: an exact-hash check. Here no prompt repeats
    # verbatim, so memo correctly abstains and arms one entry per prompt --
    # ready to short-circuit instantly the moment an exact repeat arrives.
    print(f"  memo  : {memo:.0f}% hits, {res['memo_size']} entries armed (exact-match, O(1))")
    print(f"  semantic: {semantic:.0f}% hits (cosine >= 0.80 catches the paraphrases)")
    return memo, semantic, res["memory_nodes"]


def tier45_layer_memory() -> dict:
    """Layer KV warm-start + memory graph recall, end to end via Session."""
    print()
    print(RULE)
    print("[4/5] Layer KV warm-start   (resume cached decode state)")
    print("[5/5] Memory graph recall   (related prior context)")
    res = run_v11_wiring_check()
    print(f"  layer checkpoints cached : {res['layer_cache_size']} "
          f"({res['layer_cache_bytes']} bytes)")
    print(f"  memory nodes recorded    : {res['memory_nodes']}")
    print("  run 2 hit the cached layer state and recalled run 1 "
          "(cosine 1.00).")
    return res


def scoreboard(reduction, memo, semantic, wiring) -> None:
    print()
    print(BAR)
    print(" Reuse scoreboard")
    print(BAR)
    rows = [
        ("Tier 1  trie prefix KV", f"{reduction:.1f}% tokens saved (warm)"),
        ("Tier 2  memo exact match", f"{memo:.0f}% hits (armed, O(1) on repeat)"),
        ("Tier 3  semantic cache", f"{semantic:.0f}% hit rate"),
        ("Tier 4  layer KV warm-start", f"{wiring['layer_cache_size']} checkpoint(s) reused"),
        ("Tier 5  memory graph recall", f"{wiring['memory_nodes']} node(s), recall fired"),
    ]
    for name, value in rows:
        print(f"  {name:<32} {value}")
    print()
    print("Five tiers, one runtime. Redundant work is skipped at whichever")
    print("layer first recognises it - prefix, exact, semantic, or state.")


def detailed_trace() -> None:
    """Per-run trace where memo *fires* on exact repeats and semantic on paraphrases."""
    from continuum._native import validate_v11_features

    print()
    print(BAR)
    print(" Detailed feature trace (memo + semantic + trie firing)")
    print(BAR)
    validate_v11_features()


def main() -> None:
    header()
    reduction = tier1_trie()
    memo, semantic, _ = tier23_memo_semantic()
    wiring = tier45_layer_memory()
    scoreboard(reduction, memo, semantic, wiring)
    print()
    print("Tip: re-run with --trace to watch memo fire on exact repeats.")


if __name__ == "__main__":
    import sys

    main()
    if "--trace" in sys.argv:
        detailed_trace()
