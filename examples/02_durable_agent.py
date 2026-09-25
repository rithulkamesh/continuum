"""02 - Durable agent: checkpoint mid-run, crash, resume in a fresh runtime.

A 5-step agent workflow runs to step 2, writes its full execution state
(graph + every computed value + portable KV cache) to a checkpoint store, and
"crashes". A brand-new runtime instance -- as if the process restarted, or on
another machine sharing the store -- reads the checkpoint back and finishes
the remaining steps without redoing the completed ones.

The store here is a local directory; swap in ``S3Store`` or ``GCSStore`` from
``continuum.checkpoints`` to resume across machines.

FakeLLM backend, so everything is deterministic and CI-checkable. Run it:

    PYTHONPATH=python python examples/02_durable_agent.py
"""

from __future__ import annotations

import os
import shutil
import tempfile

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum import DurableAgent  # noqa: E402
from continuum.checkpoints import LocalDirectoryStore  # noqa: E402

BAR = "=" * 60

PROMPTS = [
    "research the destination",
    "plan the route",
    "book the flights",
    "book the hotel",
    "write the itinerary",
]


def main() -> None:
    print(BAR)
    print(" Continuum - Durable Agent (checkpoint / crash / resume)")
    print(BAR)

    # --- Phase 1: run to step 2, checkpoint, "crash" -------------------
    root = tempfile.mkdtemp(prefix="continuum-ckpt-")
    store = LocalDirectoryStore(root)
    key = "trip-planner/step-2"

    agent = DurableAgent()
    total_steps = agent.begin(PROMPTS)
    checkpoint = agent.run_until_step(1, store=store, key=key)  # steps 1-2 done

    info = DurableAgent.inspect(checkpoint)
    print(f"ran {total_steps}-step workflow up to step 2")
    print(f"checkpoint: {info['executed_nodes']} nodes executed, "
          f"{info['checkpoint_bytes']} bytes -> store key {key!r}")
    del agent  # simulate the process dying
    print("agent crashed (runtime discarded)")

    # --- Phase 2: fresh runtime resumes from the store ------------------
    revived = DurableAgent()
    outputs = revived.resume_from(store=LocalDirectoryStore(root), key=key)
    print(f"fresh runtime resumed: workflow completed, {len(outputs)} node outputs")
    print(f"KV cache restored across the process boundary: "
          f"{revived.cache_size()} warm entries")

    # --- Self-check: resume is deterministic and complete --------------
    again = DurableAgent()
    outputs2 = again.resume_from(store=store, key=key)
    assert len(outputs) == 2 * total_steps  # prompt + generation per step
    assert outputs == outputs2, "replay must be deterministic"
    assert revived.cache_size() > 0, "checkpoint must carry KV state"
    shutil.rmtree(root)
    print("determinism check: two resumes produced identical outputs")
    print(BAR)
    print(" durable agent: OK")
    print(BAR)


if __name__ == "__main__":
    main()
