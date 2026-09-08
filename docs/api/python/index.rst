Continuum
=========

**The AI runtime that never computes the same thing twice, and never loses its
place.**

Continuum is a C++ execution engine that runs LLM calls and tensor ops as
operators in one dataflow graph. Work it has already done is reused at the
runtime level, and a running job can be checkpointed to bytes, resumed in
another process, or forked from any past step.

.. code-block:: bash

   pip install continuum-ai

The engine is C++. This site documents the Python surface: the frontend for
writing programs, and the runtime classes exposed through
``continuum._native``.

What it does
------------

**Reuse.** Every call checks four caches in order (exact memo, semantic,
shared-prefix KV, warm layer KV) and stops at the first hit. Only a cold call
reaches the backend. On a mixed 20-step agent run against a live Azure OpenAI
backend this cut tokens sent by 92.5 percent. See :doc:`reuse`.

**Durable execution.** A checkpoint holds the graph, every computed value, and
the KV cache. Write it to a file and a different process reads it and carries
the run forward, warm, with deterministic replay. See :doc:`durable`.

**One graph for tokens and tensors.** LLM calls and tensor ops are scheduled by
the same cache-aware interpreter. Swap providers without touching the graph.
See :doc:`backends`.

A first look
------------

.. code-block:: python

   from continuum._native import DurableAgent

   agent = DurableAgent()
   agent.begin(["pull the ticket", "reproduce the bug", "draft a fix", "open the PR"])

   ckpt = agent.run_until_step(1)              # steps 1 and 2 run now
   DurableAgent.inspect(ckpt)                  # {'executed_nodes': 2, 'checkpoint_bytes': ...}

   outputs = DurableAgent().resume_from(ckpt)  # a fresh runtime finishes steps 3 and 4

.. toctree::
   :maxdepth: 1
   :caption: Guide

   quickstart
   overview
   reuse
   durable
   backends

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference

Links
-----

- Source, benchmarks, and design docs: https://github.com/rithulkamesh/continuum
- C++ API: https://ct.rithul.dev/cpp/
- Benchmarks with raw data: https://github.com/rithulkamesh/continuum/blob/master/docs/benchmarks.md
