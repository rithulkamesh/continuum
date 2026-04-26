from continuum._native import benchmark_azure_agent, benchmark_vllm_agent


def _print(name: str, result: dict) -> None:
    print(f"{name} robust paired benchmark")
    print(f"trials={result['trials']} discarded_warmup={result['discarded_warmup_runs']}")
    print(f"median_latency_no_cache={result['median_latency_no_cache']:.2f} ms")
    print(f"median_latency_with_cache={result['median_latency_with_cache']:.2f} ms")
    print(f"p50_latency_no_cache={result['p50_latency_no_cache']:.2f} ms")
    print(f"p50_latency_with_cache={result['p50_latency_with_cache']:.2f} ms")
    print(f"p95_latency_no_cache={result['p95_latency_no_cache']:.2f} ms")
    print(f"p95_latency_with_cache={result['p95_latency_with_cache']:.2f} ms")
    print(f"latency_ratio={result['latency_ratio']:.3f}")
    print(f"token_reduction_ratio={result['token_reduction_ratio']:.3f}")
    print(f"avg_tokens_saved_ratio={result['avg_tokens_saved_ratio']:.3f}")
    print(
        f"acceptance_primary_pass={bool(result['acceptance_primary_pass'])} "
        f"acceptance_secondary_pass={bool(result['acceptance_secondary_pass'])}"
    )
    print()


if __name__ == "__main__":
    _print("Azure", benchmark_azure_agent())
    _print("vLLM", benchmark_vllm_agent())
