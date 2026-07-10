"""06 - Durable agent: checkpoint mid-run, crash, resume in a fresh runtime.

A 5-step agent workflow runs to step 2, serializes its full execution state
(graph + every computed value) to bytes, and "crashes". A brand-new runtime
instance -- as if the process restarted -- deserializes the checkpoint and
finishes the remaining steps without redoing the completed ones.

FakeLLM backend, so everything is deterministic and CI-checkable. Run it:

    PYTHONPATH=python python examples/06_durable_agent.py
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import DurableAgent  # noqa: E402

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
    agent = DurableAgent()
    total_steps = agent.begin(PROMPTS)
    checkpoint = agent.run_until_step(1)  # steps 1-2 done, 3-5 pending

    info = DurableAgent.inspect(checkpoint)
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        f.write(checkpoint)
        path = f.name
    print(f"ran {total_steps}-step workflow up to step 2")
    print(f"checkpoint: {info['executed_nodes']} nodes executed, "
          f"{info['checkpoint_bytes']} bytes -> {path}")
    del agent  # simulate the process dying
    print("agent crashed (runtime discarded)")

    # --- Phase 2: fresh runtime resumes from the file ------------------
    revived = DurableAgent()
    outputs = revived.resume_from(open(path, "rb").read())
    print(f"fresh runtime resumed: workflow completed, {len(outputs)} node outputs")
    print(f"KV cache restored across the process boundary: "
          f"{revived.cache_size()} warm entries")

    # --- Self-check: resume is deterministic and complete --------------
    again = DurableAgent()
    outputs2 = again.resume_from(open(path, "rb").read())
    assert len(outputs) == 2 * total_steps  # prompt + generation per step
    assert outputs == outputs2, "replay must be deterministic"
    assert revived.cache_size() > 0, "checkpoint must carry KV state"
    os.unlink(path)
    print("determinism check: two resumes produced identical outputs")
    print(BAR)
    print(" durable agent: OK")
    print(BAR)


if __name__ == "__main__":
    main()
