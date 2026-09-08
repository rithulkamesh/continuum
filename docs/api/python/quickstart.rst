Quickstart
==========

Install
-------

.. code-block:: bash

   pip install continuum-ai

The wheel bundles the compiled engine. The import path is ``continuum``; the
runtime classes live in ``continuum._native``.

From a checkout, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/rithulkamesh/continuum
   cd continuum
   CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --all-extras

The parallelism cap matters: the extension links libtorch with LTO, and an
unbounded build can exhaust memory on a 16 GB machine.

Run the examples
----------------

Three scripts under ``examples/`` each demonstrate one capability. They use the
``FakeLLM`` backend, so output is deterministic and safe to run in CI.

.. code-block:: bash

   PYTHONPATH=python python examples/01_reuse_stack.py       # every reuse tier, one run
   PYTHONPATH=python python examples/02_durable_agent.py     # checkpoint, crash, resume
   PYTHONPATH=python python examples/03_time_travel_fork.py  # rewind, edit, replay

Add ``--trace`` to ``01_reuse_stack.py`` to see which tier answered each call.

A minimal program
-----------------

The frontend traces a decorated Python function to the canonical IR on its
first call, then runs it through the engine.

.. code-block:: python

   from continuum import program

   @program
   def pipeline(question: str):
       # token and tensor steps recorded here become IR nodes
       ...

   pipeline("what changed in the last release?")

Tunable values are declared with :class:`~continuum.frontend.param.Param` and
searched by :class:`~continuum.frontend.optimizer.Optimizer`; parameter
discovery uses the :class:`~continuum.nn.module.Module` base class.

Next
----

- :doc:`overview` for the execution model.
- :doc:`reuse` for how reuse decisions are made and kept correct.
- :doc:`durable` for checkpoint, resume, and fork.
- :doc:`reference` for the full API.
