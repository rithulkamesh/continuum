"""08 - Offline prompt A/B: fork one checkpoint N ways.

An incident-response agent runs three steps, then has to write a post-mortem
summary. You want to compare three phrasings of that final instruction -- but
re-running the first three steps for each variant is wasteful and adds noise.

Instead: check the run once, fork the checkpoint per variant with the final
prompt edited, and resume each fork. Steps 1-3 replay bit-identically from the
checkpoint; only the edited step and its generation differ. This is a clean A/B
harness for prompt engineering and prompt-regression tests.

FakeLLM backend, deterministic, CI-checkable.

    PYTHONPATH=python python examples/08_prompt_ab_replay.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum import DurableAgent  # noqa: E402

BAR = "=" * 64

STEPS = [
    "pull the incident timeline",
    "identify the root cause",
    "gather the customer impact numbers",
    "write the post-mortem summary",
]

VARIANTS = {
    "bullets": "write the post-mortem summary as three bullet points",
    "one-liner": "write the post-mortem summary in a single sentence",
    "exec": "write the post-mortem summary for an executive audience",
}

# resume_from returns [prompt_0..prompt_{N-1}, generation_0..generation_{N-1}].
N = len(STEPS)
EDITED = N - 1  # the summary step


def main() -> None:
    print(BAR)
    print(" Continuum - Prompt A/B Replay (fork a checkpoint)")
    print(BAR)

    recorder = DurableAgent()
    recorder.begin(STEPS)
    checkpoint = recorder.run_until_step(1)  # steps 1-2 executed
    edit_node = recorder.prompt_node_ids[EDITED]
    print("ran 2/4 steps, checkpointed before the summary step")
    print("-" * 64)

    baseline = DurableAgent().resume_from(checkpoint)
    unchanged = [i for i in range(2 * N) if i not in (EDITED, N + EDITED)]

    for name, text in VARIANTS.items():
        forked = DurableAgent.fork(checkpoint, edit_node, text)
        branch = DurableAgent().resume_from(forked)
        diff = [i for i, (a, b) in enumerate(zip(baseline, branch)) if a != b]
        n_tokens = len(branch[N + EDITED])
        print(f"  [{name:9}] prompt -> {branch[EDITED]!r}")
        print(f"  {'':11} regenerated {n_tokens} tokens, diverged at {diff}")
        assert [branch[i] for i in unchanged] == [baseline[i] for i in unchanged], \
            "steps 1-3 must replay identically for every variant"
        assert diff == [EDITED, N + EDITED], "only the edited prompt and its generation may differ"

    print("-" * 64)
    print("every variant shares steps 1-3; only the summary step was recomputed")
    print(BAR)
    print(" prompt a/b replay: OK")
    print(BAR)


if __name__ == "__main__":
    main()
