"""09 - One runtime for local tensors and hosted tokens.

Continuum treats tensor ops and LLM calls as operators in the same graph, so a
pipeline can run a local scorer / reranker (libtorch or MLX) right next to a
token-generation step. This example stays on the tensor side:

    1. Cross-backend op parity - the same add / relu / softmax on libtorch and
       (if built) MLX, with the max elementwise difference.
    2. A local reranker         - fuse lexical and embedding scores with tensor
       ops, softmax to a distribution, pick the candidate to hand to the LLM.
       libtorch and MLX must agree on the winner.

    PYTHONPATH=python python examples/09_hybrid_tensor_token.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import run_tensor_op  # noqa: E402

BAR = "=" * 64

A = [1.0, 2.0, -3.0, 0.5]
B = [0.25, -1.0, 4.0, 2.0]

CANDIDATES = ["pricing.md", "faq.md", "changelog.md", "eula.md", "quickstart.md"]
LEXICAL = [0.20, 0.90, 0.40, 0.10, 0.65]
EMBEDDING = [0.55, 0.30, 0.20, 0.05, 0.60]


def _fmt(vec) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in vec) + "]"


def _mlx(op, a, b=None, dim=-1):
    """run_tensor_op on MLX, or None if the backend is not in this build."""
    try:
        return run_tensor_op(op, a, b, dim, "mlx")
    except Exception:
        return None


def parity() -> None:
    print("-" * 64)
    print("[1] tensor-op parity: libtorch vs MLX")
    for op in ("add", "relu", "softmax"):
        rhs = B if op == "add" else None
        lt = run_tensor_op(op, A, rhs, -1, "libtorch")
        mx = _mlx(op, A, rhs)
        if mx is None:
            print(f"  {op:8} libtorch={_fmt(lt)}   (mlx backend not built)")
            continue
        max_diff = max((abs(float(x) - float(y)) for x, y in zip(lt, mx)), default=0.0)
        print(f"  {op:8} libtorch={_fmt(lt)}   max_abs_diff={max_diff:.2e}")


def reranker() -> None:
    print("-" * 64)
    print("[2] local reranker (tensor ops) -> candidate for the LLM step")
    fused = run_tensor_op("add", LEXICAL, EMBEDDING, -1, "libtorch")
    gated = run_tensor_op("relu", fused, None, -1, "libtorch")
    dist = run_tensor_op("softmax", gated, None, -1, "libtorch")
    for name, p in zip(CANDIDATES, dist):
        print(f"  {name:14} p={float(p):.3f}")
    pick = max(range(len(dist)), key=lambda i: float(dist[i]))
    print(f"  -> selected: {CANDIDATES[pick]}")

    mx = _mlx("softmax", run_tensor_op("relu", fused, None, -1, "libtorch"))
    if mx is not None:
        mlx_pick = max(range(len(mx)), key=lambda i: float(mx[i]))
        assert mlx_pick == pick, "libtorch and MLX disagree on the reranked winner"
        print("  MLX agrees on the winner")


def main() -> None:
    print(BAR)
    print(" Continuum - Hybrid Tensor + Token Runtime")
    print(BAR)
    parity()
    reranker()
    print(BAR)
    print(" hybrid tensor/token: OK")
    print(BAR)


if __name__ == "__main__":
    main()
