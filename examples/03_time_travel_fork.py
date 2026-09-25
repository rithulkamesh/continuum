"""03 - Time-travel debugging: fork an agent run from a past checkpoint.

Run a 4-step workflow to step 2 and checkpoint. Then resume it twice:

    branch A: unmodified          -> the timeline that "really happened"
    branch B: step-3 prompt edited -> a counterfactual timeline

Each step sees the previous step's output, so the edit cascades: step 3 and
everything after it diverge, while completed steps are replayed from the
checkpoint (never recomputed). This is `rr` for agents: rewind, patch one
value, watch the alternate outcome.

FakeLLM backend by default, deterministic and CI-checkable. Run it:

    PYTHONPATH=python python examples/03_time_travel_fork.py

Against a live OpenAI-compatible server (vLLM, or Ollama with a model named
`gemma4`), the same script prints the generated text of every step:

    VLLM_BASE_URL=http://localhost:11434 PYTHONPATH=python \
        python examples/03_time_travel_fork.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum import DurableAgent  # noqa: E402

BAR = "=" * 60
PREVIEW_CHARS = 160

PROMPTS = [
    "summarize the bug report",
    "locate the faulty module",
    "draft a fix",
    "write the changelog entry",
]


def render(value: object) -> str:
    """Generated text for a step output; token ids only when there is no text."""
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 3] + "..."
    if isinstance(value, list):
        return f"<{len(value)} token ids from the offline fake backend>"
    return repr(value)


def show(label: str, agent: DurableAgent) -> None:
    print(f"--- {label} ---")
    for step, output in enumerate(agent.step_outputs(), start=1):
        print(f"step {step}: {render(output)}")


def main() -> None:
    print(BAR)
    print(" Continuum - Time-Travel Fork (rewind, edit, replay)")
    print(BAR)

    recorder = DurableAgent()
    recorder.begin(PROMPTS)
    checkpoint = recorder.run_until_step(1)  # steps 1-2 executed
    edit_node = recorder.prompt_node_ids[2]  # step-3 prompt, not yet executed
    print(f"recorded 4-step workflow on the {recorder.backend} backend, checkpointed after step 2")

    # Branch A: what really happened.
    agent_a = DurableAgent()
    branch_a = agent_a.resume_from(checkpoint)

    # Branch B: rewind to the checkpoint, edit the step-4 prompt, replay.
    forked = DurableAgent.fork(checkpoint, edit_node, "write a haiku instead")
    agent_b = DurableAgent()
    branch_b = agent_b.resume_from(forked)
    show("branch A: resumed unmodified", agent_a)
    show('branch B: step-3 prompt edited to "write a haiku instead"', agent_b)

    diff = [i for i, (a, b) in enumerate(zip(branch_a, branch_b)) if a != b]
    print(f"divergence: {len(diff)}/{len(branch_a)} node outputs differ")

    # Self-check: completed steps are replayed bit-identical; the edited
    # prompt, its generation, and every later generation diverge.
    assert len(branch_a) == len(branch_b) == 2 * len(PROMPTS)
    steps_a, steps_b = agent_a.step_outputs(), agent_b.step_outputs()
    assert steps_a[:2] == steps_b[:2], "completed steps must replay identically"
    assert steps_a[2] != steps_b[2] and steps_a[3] != steps_b[3], "the edit must cascade"
    if recorder.backend == "fake":
        assert len(diff) == 3, "the edit and every later step must diverge"
    print("replay check: completed steps identical, the edit cascaded downstream")
    print(BAR)
    print(" time-travel fork: OK")
    print(BAR)


if __name__ == "__main__":
    main()
