API reference
=============

Two surfaces: the pure-Python **frontend** (``continuum``) for writing
programs, and the **runtime** classes re-exported from ``continuum._native``.

Frontend
--------

.. automodule:: continuum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: continuum.frontend.param
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: continuum.frontend.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: continuum.nn.module
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: continuum.programs.program
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: continuum.checkpoints
   :members:
   :show-inheritance:

.. automodule:: continuum.embeddings
   :members: CallableEmbeddingProvider, OpenAICompatibleEmbeddingProvider, PrecomputedEmbeddingProvider
   :show-inheritance:

Runtime
-------

These classes are defined by the C++ pybind bindings and re-exported from
``continuum._native``. Import them from there:

.. code-block:: python

   from continuum import DurableAgent
   from continuum._native import Session, ReusePolicy

DurableAgent
~~~~~~~~~~~~

.. py:class:: continuum._native.DurableAgent

   A step-sequenced agent run that can be checkpointed, resumed, and forked.
   See :doc:`durable`.

   .. py:method:: begin(prompts, model_id=None, max_tokens=32)

      Build the graph for a list of step prompts. Returns the step count. Each
      prompt becomes a ``PromptOp`` feeding a ``TokenOp``; every step after the
      first also receives the previous step's output. ``model_id`` defaults to
      ``"vllm/gemma4"`` when ``VLLM_BASE_URL`` is set, else ``"fake/model"``.

   .. py:method:: run_until_step(step_index)

      Execute through ``step_index`` (0-based) and return the checkpoint as
      ``bytes``: the graph, every computed value, and the portable KV state.

   .. py:method:: resume_from(checkpoint)

      Deserialize ``checkpoint`` bytes into a fresh runtime and run to
      completion. Returns the list of node outputs. Deterministic: two resumes
      of one checkpoint return identical output.

   .. py:method:: cache_size()

      Number of warm KV entries currently held.

   .. py:method:: step_outputs()

      Each step's generated output in step order (``None`` if not yet run):
      text on a live server, token ids on the fake backend.

   .. py:attribute:: backend

      ``"vllm"`` when ``VLLM_BASE_URL`` is set, else ``"fake"``.

   .. py:staticmethod:: inspect(checkpoint)

      Read a checkpoint without a runtime. Returns a dict with
      ``executed_nodes`` and ``checkpoint_bytes``.

   .. py:staticmethod:: fork(checkpoint, node_id, new_value)

      Return a new checkpoint with the value at ``node_id`` replaced. Resuming
      it replays completed steps unchanged and diverges only at the edited node
      and downstream.

   .. py:attribute:: prompt_node_ids

      List of ``PromptOp`` node ids, one per step, in order.

   .. py:attribute:: step_node_ids

      List of ``TokenOp`` (generation) node ids, one per step, in order.

Session
~~~~~~~

.. py:class:: continuum._native.Session(id, backend_registry, max_cache=...)

   A reuse-aware execution context. Holds the cache index and the reuse policy
   across many ``run`` calls.

   .. py:method:: run(graph, inputs)

      Execute ``graph`` with a mapping of input node id to value, applying the
      reuse stack to every ``TokenOp``.

   .. py:attribute:: policy

      The :py:class:`~continuum._native.ReusePolicy` for this session.
      Assignable.

   .. py:method:: metrics()

      A :py:class:`~continuum._native.ReuseMetrics` snapshot: tokens sent,
      tokens saved, per-tier hit counts.

   .. py:method:: reset_metrics()

   .. py:method:: save_cache_metadata(path)
   .. py:method:: load_cache_metadata(path)

      Persist and reload the cache index across processes. See
      :doc:`reuse` on cross-session persistence.

   .. py:method:: generate(prompt_parts, model_id, max_tokens=128, temperature=0.0, op_name="generate")

      Run one generation through the reuse stack: each string in
      ``prompt_parts`` becomes a ``PromptOp`` feeding a single ``TokenOp``.
      Returns the output value and appends a step to ``metrics()``.

   .. py:method:: cache_size()

   .. py:method:: cache_stats()

      Per-tier occupancy: ``{tier: {"entries", "capacity", "bytes"}}`` for
      every attached tier. See "Eviction and memory bounds" in
      ``docs/design/cache.md``.

   .. py:method:: set_memo_table(table)
   .. py:method:: set_semantic_cache(index)
   .. py:method:: set_embedding_provider(provider)
   .. py:method:: set_layer_cache(index)
   .. py:method:: set_memory_graph(store)

      Attach the backing store for each reuse tier. A tier with no store
      attached is skipped.

ReusePolicy
~~~~~~~~~~~

.. py:class:: continuum._native.ReusePolicy

   .. py:staticmethod:: always()

      Reuse whenever any tier matches.

   .. py:staticmethod:: never()

      Bypass every tier. Every call reaches the backend.

   .. py:staticmethod:: threshold(min_len)

      Reuse a prefix only when it is at least ``min_len`` tokens.

   .. py:attribute:: kind

      A :py:class:`~continuum._native.ReusePolicyKind`.

   .. py:attribute:: min_prefix_len

      The threshold, when ``kind`` is ``ThresholdPrefixLen``.

.. py:class:: continuum._native.ReusePolicyKind

   Enum: ``Always``, ``Never``, ``ThresholdPrefixLen``.

Reuse-tier stores
~~~~~~~~~~~~~~~~~

Backing stores for the tiers, attached to a ``Session`` with the ``set_*``
methods above:

- ``MemoTable`` / ``MemoKey`` : exact-repeat memoization.
- ``SemanticCacheIndex`` : embedding-matched near-duplicates. Needs an
  ``EmbeddingProvider``: ``BruteForceEmbeddingProvider`` is bundled, and
  :mod:`continuum.embeddings` adapts local models, hosted endpoints, and
  precomputed vectors. Subclass ``EmbeddingProvider`` for anything else.
- ``LayerKVCacheIndex`` : warm attention-layer state.
- ``MemoryGraphStore`` : prior-run context recall.
- ``FutureCache`` : in-flight de-duplication of concurrent identical calls.

IR
~~

- ``Graph`` , ``Node`` , ``NodeKind`` : the dataflow IR. See :doc:`overview`.
- ``GraphBuilder`` : incremental graph construction.
- ``Interpreter`` : the executor. ``Session`` and ``DurableAgent`` wrap it.
- ``BackendRegistry`` : registers backends and their priorities.

Benchmark entrypoints
~~~~~~~~~~~~~~~~~~~~~~

``run_session_benchmark`` , ``run_cold_start_benchmark`` , ``run_v11_benchmark`` ,
``benchmark_azure_agent`` , ``benchmark_vllm_agent`` , and related functions
drive the numbers in :doc:`reuse` and ``docs/benchmarks.md``. They return
plain dicts of metrics. ``examples/01_reuse_stack.py`` calls them directly.
