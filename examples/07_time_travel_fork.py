"""07 - Time-travel debugging: fork an agent run from a past checkpoint.

Run a 4-step workflow to step 2 and checkpoint. Then resume it twice:

    branch A: unmodified          -> the timeline that "really happened"
    branch B: step-4 prompt edited -> a counterfactual timeline

Completed steps are replayed from the checkpoint (never recomputed); only
the edit and its downstream generation diverge. This is `rr` for agents:
rewind, patch one value, watch the alternate outcome.

FakeLLM backend, deterministic, CI-checkable. Run it:

    PYTHONPATH=python python examples/07_time_travel_fork.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import DurableAgent  # noqa: E402

BAR = "=" * 60

PROMPTS = [
    "summarize the bug report",
    "locate the faulty module",
    "draft a fix",
    "write the changelog entry",
]


def main() -> None:
    print(BAR)
    print(" Continuum - Time-Travel Fork (rewind, edit, replay)")
    print(BAR)

    recorder = DurableAgent()
    recorder.begin(PROMPTS)
    checkpoint = recorder.run_until_step(1)  # steps 1-2 executed
    edit_node = recorder.prompt_node_ids[3]  # step-4 prompt, not yet executed
    print("recorded 4-step workflow, checkpointed after step 2")

    # Branch A: what really happened.
    branch_a = DurableAgent().resume_from(checkpoint)

    # Branch B: rewind to the checkpoint, edit the step-4 prompt, replay.
    forked = DurableAgent.fork(checkpoint, edit_node, "write a haiku instead")
    branch_b = DurableAgent().resume_from(forked)
    print("branch A: resumed unmodified")
    print('branch B: step-4 prompt edited to "write a haiku instead"')

    diff = [i for i, (a, b) in enumerate(zip(branch_a, branch_b)) if a != b]
    print(f"divergence: {len(diff)}/{len(branch_a)} node outputs differ")

    # Self-check: exactly the edited prompt and its generation diverge;
    # every already-executed step is replayed bit-identical.
    assert len(branch_a) == len(branch_b) == 2 * len(PROMPTS)
    assert len(diff) == 2, "only the edited step may diverge"
    print("replay check: completed steps identical, only the edit diverged")
    print(BAR)
    print(" time-travel fork: OK")
    print(BAR)


if __name__ == "__main__":
    main()
