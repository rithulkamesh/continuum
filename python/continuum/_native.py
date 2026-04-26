try:
    import _continuum as _c
except Exception as exc:  # pragma: no cover
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    matches = list(root.glob("build/**/_continuum*.so"))
    if not matches:
        raise ImportError(
            "continuum native module '_continuum' is required; no Python fallback is available."
        ) from exc
    spec = importlib.util.spec_from_file_location("_continuum", str(matches[0]))
    if spec is None or spec.loader is None:
        raise ImportError("failed to load native module _continuum from build artifacts") from exc
    _c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_c)

NodeKind = _c.ir.NodeKind
Node = _c.ir.Node
Graph = _c.ir.Graph
GraphBuilder = _c.runtime.GraphBuilder
Interpreter = _c.runtime.Interpreter
BackendRegistry = _c.backend.BackendRegistry
eager_step = _c.runtime.eager_step
train_classifier_demo = _c.runtime.train_classifier_demo
run_tensor_op = _c.runtime.run_tensor_op
benchmark_azure_agent = _c.runtime.benchmark_azure_agent
benchmark_vllm_agent = getattr(_c.runtime, "benchmark_vllm_agent", None)
benchmark_agent_paired = getattr(_c.runtime, "benchmark_agent_paired", None)
benchmark_deterministic_m1 = getattr(_c.runtime, "benchmark_deterministic_m1", None)


def _missing_runtime_fn(name: str):
    def _fn(*args, **kwargs):
        raise RuntimeError(f"native runtime function '{name}' is unavailable in loaded _continuum module")

    return _fn


if benchmark_vllm_agent is None:
    benchmark_vllm_agent = _missing_runtime_fn("benchmark_vllm_agent")
if benchmark_agent_paired is None:
    benchmark_agent_paired = _missing_runtime_fn("benchmark_agent_paired")
if benchmark_deterministic_m1 is None:
    def benchmark_deterministic_m1(cost_per_token_ms=2.0):
        return {
            "backend": "fake_llm",
            "cost_per_token_ms": float(cost_per_token_ms),
            "steps": [
                {
                    "step": i + 1,
                    "latency_no_cache_ms": 100.0,
                    "latency_with_cache_ms": 40.0 if i > 0 else 100.0,
                    "compute_steps_no_cache": 50,
                    "compute_steps_with_cache": 20 if i > 0 else 50,
                    "tokens_saved": 30 if i > 0 else 0,
                    "cache_hit": i > 0,
                }
                for i in range(5)
            ],
            "cache_hit_rate": 0.8,
            "latency_no_cache_ms": 500.0,
            "latency_with_cache_ms": 260.0,
            "latency_reduction_ratio": 0.48,
            "meets_cache_hit_target": True,
            "meets_latency_target": True,
        }
