from functools import wraps

from continuum.frontend.trace import trace


def program(fn):
    """Mark a function as a Continuum program; first call traces it to CIR."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "_cir"):
            with trace() as builder:
                fn(*args, **kwargs)
            wrapper._cir = builder.finalize()
        return wrapper._cir.run(args, kwargs)

    def compile_target(target: str):
        return {"target": target, "graph": getattr(wrapper, "_cir", None)}

    def cir():
        return getattr(wrapper, "_cir", None)

    wrapper.compile = compile_target
    wrapper.cir = cir
    return wrapper
