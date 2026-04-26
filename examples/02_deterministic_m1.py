from continuum._native import benchmark_deterministic_m1


if __name__ == "__main__":
    result = benchmark_deterministic_m1(cost_per_token_ms=2.0)
    print("Deterministic M1 validation (fake backend)")
    print(f"cost_per_token_ms={result['cost_per_token_ms']:.2f}")
    for row in result["steps"]:
        print(
            "step={step} no_cache_ms={n:.2f} with_cache_ms={w:.2f} "
            "compute_no_cache={cn} compute_with_cache={cw} cache_hit={hit}".format(
                step=row["step"],
                n=row["latency_no_cache_ms"],
                w=row["latency_with_cache_ms"],
                cn=row["compute_steps_no_cache"],
                cw=row["compute_steps_with_cache"],
                hit=bool(row["cache_hit"]),
            )
        )
    print(f"cache_hit_rate={result['cache_hit_rate'] * 100:.2f}%")
    print(f"latency_no_cache_ms={result['latency_no_cache_ms']:.2f}")
    print(f"latency_with_cache_ms={result['latency_with_cache_ms']:.2f}")
    print(f"latency_reduction={result['latency_reduction_ratio'] * 100:.2f}%")
    print(
        "acceptance cache_hit>=80%={} latency_reduction>=20%={}".format(
            bool(result["meets_cache_hit_target"]),
            bool(result["meets_latency_target"]),
        )
    )
