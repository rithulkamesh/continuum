The execution model
===================

Continuum runs a program as a dataflow graph. Understanding four pieces (the
IR, the interpreter, backends, and the reuse stack) is enough to reason about
what it will and will not recompute.

The graph
---------

A program is a :class:`~continuum._native.Graph` of nodes. Each node is one
step with typed inputs and one output:

===============  ==========================================================
Node kind        What it is
===============  ==========================================================
``TensorOp``     A tensor computation, run on a tensor backend such as libtorch.
``TokenOp``      An LLM generation or scoring call.
``PromptOp``     Prompt assembly. Pass-through, with an empty-string fallback.
``ToolOp``       A side-effecting call. **Never served from cache.**
``ControlOp``    Branching and control flow.
===============  ==========================================================

The graph plus its literal inputs is the whole program. It serializes to a
compact binary envelope (the "CIR"), which is what checkpoints carry.

The interpreter
---------------

:class:`~continuum._native.Interpreter` executes nodes in scheduler order,
which defaults to topological order. Input values are injected first, then each
remaining node is computed once its dependencies are materialized.

For a ``TokenOp`` the interpreter does more than dispatch: it canonicalizes the
input tokens, asks the reuse stack for the longest reusable prefix, and calls
the backend with the reusable state attached. The backend returns the output,
an updated state handle for future reuse, and metrics (``tokens_sent``,
``tokens_saved``, ``reused_prefix_len``, ``compute_steps``).

Backends
--------

A backend runs one node kind against one provider. The interpreter owns
scheduling and the cache index; the backend owns model execution and how its
state is encoded. Backends declare which capabilities they have (tensor, token,
cache), and the scheduler routes each node accordingly, converting tensors
between backends explicitly when a graph spans more than one.

See :doc:`backends` for the list.

The reuse stack
---------------

Before any ``TokenOp`` reaches a backend it passes through five reuse tiers, in
cost order. The first tier that can answer does, and the backend is not called.
See :doc:`reuse` for each tier and the rules that keep reuse correct.

Checkpoints
-----------

Serializing the graph together with every computed value and the portable KV
state produces a checkpoint. A fresh process deserializes it and resumes from
the next unexecuted node, or forks from a past node with one value replaced.
See :doc:`durable`.

The ownership boundary
----------------------

The cache is runtime-owned, but a reuse decision depends on backend state
handles. That coupling is deliberate: the runtime decides *when* a prefix is
reusable, the backend decides *how* that state is represented. The Python layer
is a thin wrapper over this C++ path. It does not reimplement scheduling or
cache logic; it calls the same code the test suite exercises.
