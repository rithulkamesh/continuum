"""07 - Spot-instance eviction: resume a long job from a retained checkpoint.

A 6-step data pipeline runs on a preemptible node. The orchestrator keeps a
rolling checkpoint in durable storage. The node is reclaimed mid-run; a fresh
pod picks up the newest checkpoint and finishes, without redoing completed
steps and with the KV cache still warm across the process boundary.

Also shows checkpoint *retention*: an older checkpoint is still a valid restore
point -- it just replays a little more.

FakeLLM backend, deterministic, CI-checkable.

    PYTHONPATH=python python examples/07_spot_eviction_resume.py
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import DurableAgent  # noqa: E402

BAR = "=" * 64

PIPELINE = [
    "ingest the raw export",
    "validate the schema",
    "deduplicate records",
    "enrich with metadata",
    "run quality checks",
    "publish the dataset",
]


def checkpoint_after(step_index: int) -> bytes:
    """A fresh runtime runs the pipeline up to and including `step_index`."""
    agent = DurableAgent()
    agent.begin(PIPELINE)
    return agent.run_until_step(step_index)


def main() -> None:
    print(BAR)
    print(" Continuum - Spot Eviction Resume (retained checkpoints)")
    print(BAR)

    total = len(PIPELINE)
    early = checkpoint_after(1)  # after step 2
    late = checkpoint_after(3)   # after step 4  <- newest retained

    for name, blob in (("early (step 2)", early), ("late  (step 4)", late)):
        info = DurableAgent.inspect(blob)
        print(f"retained {name}: {info['executed_nodes']} nodes done, {info['checkpoint_bytes']} bytes")

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        f.write(late)
        path = f.name
    print("node reclaimed by the cloud provider; newest checkpoint is in object storage")

    # Fresh pod, brand-new runtime.
    revived = DurableAgent()
    outputs = revived.resume_from(open(path, "rb").read())
    print(f"fresh pod resumed from {os.path.basename(path)}: "
          f"{len(outputs)} node outputs, {revived.cache_size()} warm KV entries")

    # Retention check: the older checkpoint still resumes to the same result.
    from_early = DurableAgent().resume_from(early)
    again = DurableAgent().resume_from(late)
    os.unlink(path)

    assert len(outputs) == len(from_early) == 2 * total
    assert outputs == again, "resume must be deterministic"
    assert outputs == from_early, "an older restore point must reach the same final state"
    assert revived.cache_size() > 0, "checkpoint must carry KV state across the process boundary"
    print("checks: warm resume, deterministic replay, older checkpoint equivalent")
    print(BAR)
    print(" spot eviction resume: OK")
    print(BAR)


if __name__ == "__main__":
    main()
