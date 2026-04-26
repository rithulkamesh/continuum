from continuum._native import run_tensor_op, train_classifier_demo


def _op_parity_check():
    a = [1.0, 2.0, -3.0, 0.5]
    b = [0.25, -1.0, 4.0, 2.0]
    ops = [
        ("add", {"a": a, "b": b, "dim": -1}),
        ("relu", {"a": a, "b": None, "dim": -1}),
        ("softmax", {"a": a, "b": None, "dim": -1}),
    ]
    print("tensor-op parity (libtorch vs mlx)")
    for op, kwargs in ops:
        out_torch = run_tensor_op(op, kwargs["a"], kwargs["b"], kwargs["dim"], "libtorch")
        out_mlx = run_tensor_op(op, kwargs["a"], kwargs["b"], kwargs["dim"], "mlx")
        diffs = [abs(float(x) - float(y)) for x, y in zip(out_torch, out_mlx)]
        max_diff = max(diffs) if diffs else 0.0
        print(f"op={op} max_abs_diff={max_diff:.6f}")


if __name__ == "__main__":
    _op_parity_check()
    print("training convergence (libtorch baseline)")
    logs = train_classifier_demo(10, 0.1)
    for row in logs:
        print(f"epoch={row['epoch']} loss={row['loss']:.4f} accuracy={row['accuracy']:.4f}")
