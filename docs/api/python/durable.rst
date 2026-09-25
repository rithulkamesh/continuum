Durable execution
=================

A long agent run should survive a deploy, a crash, or a spot-instance
eviction, and it should be possible to rewind it and try a different step. Both
come from the same primitive: a checkpoint.

What a checkpoint is
--------------------

:meth:`DurableAgent.run_until_step <continuum._native.DurableAgent.run_until_step>`
returns a ``bytes`` object that contains:

- the graph,
- every value computed so far,
- the portable KV cache state.

That is the entire run state. There is no external database and no in-memory
handle to keep alive. Move the bytes and the run moves with them.

:meth:`DurableAgent.inspect <continuum._native.DurableAgent.inspect>` reads a
checkpoint without a runtime and reports ``executed_nodes`` and
``checkpoint_bytes``.

Checkpoint and resume
---------------------

.. code-block:: python

   from continuum import DurableAgent

   agent = DurableAgent()
   agent.begin(["pull the ticket", "reproduce the bug", "draft a fix", "open the PR"])

   ckpt = agent.run_until_step(1)              # steps 1 and 2 run now
   del agent                                   # the process can die here

   revived = DurableAgent()                    # a fresh runtime
   outputs = revived.resume_from(ckpt)         # finishes steps 3 and 4, nothing re-run
   revived.cache_size()                        # the KV cache came back with the checkpoint

``resume_from`` runs to completion and returns the list of node outputs. Two
resumes of the same checkpoint produce byte-identical output.

Across processes
----------------

The checkpoint is bytes, so the boundary between stages can be a file, an
object store, or a queue. Nothing else has to cross it.

.. code-block:: python

   # plan.py : a small box runs the cheap steps, then parks the job
   from pathlib import Path
   from continuum import DurableAgent

   agent = DurableAgent()
   agent.begin(["pull the ticket", "reproduce the bug", "draft a fix", "open the PR"])
   Path("job.ckpt").write_bytes(agent.run_until_step(1))

.. code-block:: python

   # work.py : a different process, a GPU box, an hour later
   from pathlib import Path
   from continuum import DurableAgent

   outputs = DurableAgent().resume_from(Path("job.ckpt").read_bytes())
   Path("job.result").write_text(repr(outputs))

.. code-block:: python

   # audit.py : a third process, anywhere, replays the same bytes
   from pathlib import Path
   from continuum import DurableAgent

   blob = Path("job.ckpt").read_bytes()
   assert DurableAgent().resume_from(blob) == DurableAgent().resume_from(blob)

Because the checkpoint carries the KV cache, ``work.py`` resumes warm rather
than re-tokenizing the prefix from cold.

Checkpoint stores
-----------------

Instead of moving bytes yourself, hand ``run_until_step`` / ``resume_from`` a
:class:`~continuum.checkpoints.CheckpointStore` and a key:

.. code-block:: python

   from continuum import DurableAgent
   from continuum.checkpoints import LocalDirectoryStore, S3Store, GCSStore

   store = LocalDirectoryStore("/mnt/shared/ckpt")
   # store = S3Store("my-bucket", prefix="continuum")     # pip install boto3
   # store = GCSStore("my-bucket", prefix="continuum")    # pip install google-cloud-storage

   agent = DurableAgent()
   agent.begin(["pull the ticket", "reproduce the bug", "draft a fix", "open the PR"])
   agent.run_until_step(1, store=store, key="ticket-42/step-2")

   # any machine that can reach the store:
   DurableAgent().resume_from(store=store, key="ticket-42/step-2")

Every store implements ``put``, ``get``, ``exists``, ``list``, ``delete``, and
an atomic ``put_if_absent``; subclass ``CheckpointStore`` for anything else.
Local writes are atomic renames, so a reader never sees a partial checkpoint.

Incremental checkpoints, forks, and many workers
------------------------------------------------

:class:`~continuum.checkpoints.CheckpointLog` keeps a whole run's checkpoint
stream in one store:

.. code-block:: python

   from continuum.checkpoints import CheckpointLog

   log = CheckpointLog(store, run_id="ticket-42")
   prev = None
   for step in range(4):
       prev = log.commit(agent.run_until_step(step), parent=prev, step=step)

   alt = log.fork(prev, agent.prompt_node_ids[3], "open a draft PR instead")
   DurableAgent().resume_from(log.load(alt))

- **Deltas.** A checkpoint committed with a ``parent`` is stored as a delta:
  only the values and KV entries that changed since the parent
  (:func:`continuum._native.checkpoint_delta`). A full checkpoint is written
  every ``max_chain`` links (default 16) to bound reconstruction.
- **Manifest.** Each checkpoint has a JSON record (``log.manifest()``,
  ``log.record(id)``) naming its parent, its delta base, and its depth, so any
  step rebuilds with ``log.load(id)``. ``log.lineage(id)`` walks back to the
  root and ``log.heads()`` lists branch tips.
- **Fork lineage.** ``log.fork(id, node_id, value)`` records the source
  checkpoint as ``parent`` and the edited node in ``record.fork``.
- **Integrity.** Ids are the SHA-256 of the full checkpoint, and ``load``
  verifies every rebuilt checkpoint against its id
  (``CheckpointCorruptionError`` otherwise).
- **Many workers.** Objects and records are write-once
  (``put_if_absent``) and never modified. Any number of workers can load from
  and commit to one log with no locks: committing a checkpoint that already
  exists is a no-op, and different checkpoints never share a key.

Fork a timeline
---------------

:meth:`DurableAgent.fork <continuum._native.DurableAgent.fork>` takes a
checkpoint, the node id of a value to replace, and the replacement, and returns
a new checkpoint. Completed steps replay from the checkpoint and are never
recomputed. Each step's generation sees its own prompt plus the previous step's
output, so an edit changes that step and every step after it. This is ``rr``
for agents.

.. code-block:: python

   from continuum import DurableAgent

   rec = DurableAgent()
   rec.begin(["summarize the bug", "find the module", "draft a fix", "write the changelog"])
   ckpt = rec.run_until_step(1)               # steps 1 and 2 executed
   step_3 = rec.prompt_node_ids[2]            # the step-3 prompt, still pending

   real, what_if = DurableAgent(), DurableAgent()
   real.resume_from(ckpt)
   what_if.resume_from(DurableAgent.fork(ckpt, step_3, "write a haiku instead"))
   real.step_outputs()      # one generated output per step, in step order
   what_if.step_outputs()   # steps 1-2 identical; steps 3 and 4 differ

``prompt_node_ids`` and ``step_node_ids`` expose the node ids for each step, so
you can target the prompt or the generation of any step by index.
``step_outputs()`` returns each step's generated output (``None`` for a step
that has not run).

Running on a real model (vLLM, Ollama)
--------------------------------------

By default ``DurableAgent`` runs on the deterministic fake backend and step
outputs are token ids. Set ``VLLM_BASE_URL`` to any server that speaks the
OpenAI ``/v1/completions`` format and the agent sends every step there instead,
and ``step_outputs()`` returns the generated text:

.. code-block:: bash

   ollama pull gemma4                    # or: vllm serve <model> --served-model-name gemma4
   export VLLM_BASE_URL=http://localhost:11434
   PYTHONPATH=python python examples/03_time_travel_fork.py

The model id defaults to ``vllm/gemma4``; the ``vllm/`` prefix is stripped, so
the server is asked for ``gemma4``. Pass ``begin(prompts, model_id="vllm/<name>")``
to pick another model. ``agent.backend`` reports ``"vllm"`` or ``"fake"``.
Each request carries the full prompt (the step's prompt, then the previous
step's output); prefix reuse happens server side and ``cached_tokens`` from the
response is reported as ``tokens_saved``.
