"""Typed access to the compiled ``_continuum`` extension module.

Every name re-exported here is defined by the C++ pybind11 bindings in
``bindings/pybind/``. If one is missing at import time, the installed
``_continuum`` build is stale: rebuild with ``scripts/build.sh`` or
``pip install --no-build-isolation -e .``.
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401  loads libtorch before the extension

    try:  # loads libmlx when the extension was built against Apple MLX
        import mlx.core as _mlx_core  # noqa: F401
    except ImportError:
        pass
    from . import _continuum as _c  # type: ignore[attr-defined]
except Exception as _exc:  # pragma: no cover - only hit on a broken install
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _matches = sorted(_Path(__file__).resolve().parents[2].glob("build/**/_continuum*.so"))
    if not _matches:
        raise ImportError(
            "continuum native module '_continuum' is not built; run scripts/build.sh"
        ) from _exc
    _spec = _ilu.spec_from_file_location("continuum._continuum", str(_matches[-1]))
    assert _spec is not None and _spec.loader is not None
    _c = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_c)

_ir = _c.ir
_rt = _c.runtime
_backend = _c.backend

# --- IR -------------------------------------------------------------------
NodeKind = _ir.NodeKind
Node = _ir.Node
Graph = _ir.Graph

# --- Backends -----------------------------------------------------------
BackendRegistry = _backend.BackendRegistry
check_backend = _backend.check_backend
mlx_runtime = _backend.mlx_runtime

# --- Runtime core -----------------------------------------------------
GraphBuilder = _rt.GraphBuilder
Interpreter = _rt.Interpreter
DurableAgent = _rt.DurableAgent
eager_step = _rt.eager_step
checkpoint_delta = _rt.checkpoint_delta
apply_checkpoint_delta = _rt.apply_checkpoint_delta
is_checkpoint_delta = _rt.is_checkpoint_delta
run_tensor_op = _rt.run_tensor_op
train_classifier_demo = _rt.train_classifier_demo

# --- Reuse subsystems -------------------------------------------------
Session = _rt.Session
KVCacheIndex = _rt.KVCacheIndex
ReusePolicy = _rt.ReusePolicy
ReusePolicyKind = _rt.ReusePolicyKind
ReuseMetrics = _rt.ReuseMetrics
ReuseStepRecord = _rt.ReuseStepRecord
ReuseEvent = _rt.ReuseEvent
ReuseEventKind = _rt.ReuseEventKind
ReuseObserver = _rt.ReuseObserver
MemoTable = _rt.MemoTable
MemoKey = _rt.MemoKey
SemanticCacheIndex = _rt.SemanticCacheIndex
MemoryGraphStore = _rt.MemoryGraphStore
LayerKVCacheIndex = _rt.LayerKVCacheIndex
FutureCache = _rt.FutureCache
EmbeddingProvider = _rt.EmbeddingProvider
BruteForceEmbeddingProvider = _rt.BruteForceEmbeddingProvider

# --- Benchmark entrypoints ----------------------------------------
benchmark_azure_agent = _rt.benchmark_azure_agent
benchmark_vllm_agent = _rt.benchmark_vllm_agent
benchmark_agent_paired = _rt.benchmark_agent_paired
benchmark_azure_with_prompt = _rt.benchmark_azure_with_prompt
benchmark_azure_isolated = _rt.benchmark_azure_isolated
benchmark_deterministic_m1 = _rt.benchmark_deterministic_m1
run_session_benchmark = _rt.run_session_benchmark
run_cold_start_benchmark = _rt.run_cold_start_benchmark
run_v11_benchmark = _rt.run_v11_benchmark
run_v11_wiring_check = _rt.run_v11_wiring_check
run_v11_layer_isolation_check = _rt.run_v11_layer_isolation_check
validate_v11_features = _rt.validate_v11_features

__all__ = [
    "NodeKind",
    "Node",
    "Graph",
    "BackendRegistry",
    "check_backend",
    "mlx_runtime",
    "GraphBuilder",
    "Interpreter",
    "DurableAgent",
    "eager_step",
    "checkpoint_delta",
    "apply_checkpoint_delta",
    "is_checkpoint_delta",
    "run_tensor_op",
    "train_classifier_demo",
    "Session",
    "KVCacheIndex",
    "ReusePolicy",
    "ReusePolicyKind",
    "ReuseMetrics",
    "ReuseStepRecord",
    "ReuseEvent",
    "ReuseEventKind",
    "ReuseObserver",
    "MemoTable",
    "MemoKey",
    "SemanticCacheIndex",
    "MemoryGraphStore",
    "LayerKVCacheIndex",
    "FutureCache",
    "EmbeddingProvider",
    "BruteForceEmbeddingProvider",
    "benchmark_azure_agent",
    "benchmark_vllm_agent",
    "benchmark_agent_paired",
    "benchmark_azure_with_prompt",
    "benchmark_azure_isolated",
    "benchmark_deterministic_m1",
    "run_session_benchmark",
    "run_cold_start_benchmark",
    "run_v11_benchmark",
    "run_v11_wiring_check",
    "run_v11_layer_isolation_check",
    "validate_v11_features",
]
