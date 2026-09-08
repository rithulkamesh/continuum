"""05 - Eval / CI loop that stops paying to re-run the same suite.

Every eval case shares a big grader preamble (rubric, format rules, few-shot
examples). A naive harness re-tokenizes and re-sends that preamble for every
case, every run. Continuum's prefix KV cache sends it once; case 2..N only pay
for their own suffix. Re-running the suite hour after hour in CI then costs a
fraction of the first pass.

Two acts, both on the deterministic FakeLLM backend:

    Act 1  benchmark_deterministic_m1 - 5 eval cases, one 3,000-char rubric
           preamble. Reports the runtime's own cache-hit and latency numbers.
    Act 2  run_session_benchmark      - the same suite re-run 8 times. Token
           reduction climbs from ~0 on the cold run to its warm ceiling.

    PYTHONPATH=python python examples/05_ci_eval_replay.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import (  # noqa: E402
    benchmark_deterministic_m1,
    run_session_benchmark,
)

BAR = "=" * 64


def act1_shared_preamble() -> None:
    print("-" * 64)
    print("[Act 1] 5 eval cases sharing one 3,000-char rubric preamble")
    res = benchmark_deterministic_m1(cost_per_token_ms=2.0)
    for row in res["steps"]:
        tag = "cache hit " if row["cache_hit"] else "cold      "
        print(
            f"  case {row['step']}: {tag} "
            f"compute {row['compute_steps_no_cache']:>5} -> {row['compute_steps_with_cache']:>5} steps  "
            f"tokens_saved={row['tokens_saved']}"
        )
    print(
        f"  cache_hit_rate={res['cache_hit_rate'] * 100:.0f}%  "
        f"latency_reduction={res['latency_reduction_ratio'] * 100:.0f}%"
    )
    assert res["meets_cache_hit_target"] and res["meets_latency_target"], "eval-cache targets regressed"


def act2_repeat_the_suite() -> None:
    print("-" * 64)
    print("[Act 2] same suite re-run 8x against a persistent session cache")
    res = run_session_benchmark(num_steps=8, prefix_tokens=3000, suffix_tokens=16)
    cold = res["runs"][0]
    warm = res["runs"][-1]
    print(f"  cold run 1 : token_reduction={cold['token_reduction'] * 100:5.1f}%  "
          f"tokens_processed={cold['total_tokens_processed']}")
    print(f"  warm run {res['total_runs']} : token_reduction={warm['token_reduction'] * 100:5.1f}%  "
          f"tokens_processed={warm['total_tokens_processed']}")
    print(f"  session cache entries: {res['final_cache_size']}")
    assert warm["token_reduction"] > cold["token_reduction"], "warm run should reuse the shared prefix"


def main() -> None:
    print(BAR)
    print(" Continuum - Eval / CI Replay (shared preamble, repeated runs)")
    print(BAR)
    act1_shared_preamble()
    act2_repeat_the_suite()
    print(BAR)
    print(" ci eval replay: OK")
    print(BAR)


if __name__ == "__main__":
    main()
