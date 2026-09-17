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

Fork a timeline
---------------

:meth:`DurableAgent.fork <continuum._native.DurableAgent.fork>` takes a
checkpoint, the node id of a value to replace, and the replacement, and returns
a new checkpoint. Completed steps replay from the checkpoint and are never
recomputed; only the edited node and its downstream generation diverge. This is
``rr`` for agents.

.. code-block:: python

   from continuum import DurableAgent

   rec = DurableAgent()
   rec.begin(["summarize the bug", "find the module", "draft a fix", "write the changelog"])
   ckpt = rec.run_until_step(1)               # steps 1 and 2 executed
   step_4 = rec.prompt_node_ids[3]            # the step-4 prompt, still pending

   real    = DurableAgent().resume_from(ckpt)
   what_if = DurableAgent().resume_from(
       DurableAgent.fork(ckpt, step_4, "write a haiku instead"),
   )
   # steps 1 to 3 replay bit for bit; only step 4 and its output differ

``prompt_node_ids`` and ``step_node_ids`` expose the node ids for each step, so
you can target the prompt or the generation of any step by index.
