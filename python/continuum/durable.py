"""Public durable-agent API.

:class:`DurableAgent` checkpoints mid-run (value map + portable KV state),
resumes on a fresh process, and forks alternate timelines by editing a
computed node. The engine lives in the compiled extension; this subclass adds
checkpointing straight to a :class:`~continuum.checkpoints.CheckpointStore`
(local directory, S3, GCS) so a run can resume on another machine.
"""

from __future__ import annotations

from typing import Any

from continuum._native import DurableAgent as _NativeDurableAgent
from continuum.checkpoints import CheckpointStore
from continuum.telemetry import auto_instrument

__all__ = ["DurableAgent"]


class DurableAgent(_NativeDurableAgent):
    """Step-sequenced agent run that can be checkpointed, resumed, and forked.

    With ``CONTINUUM_OTEL=1`` in the environment, each agent exports its
    reuse events through OpenTelemetry (see :mod:`continuum.telemetry`).
    """

    def __init__(self) -> None:
        super().__init__()
        auto_instrument(self)

    def run_until_step(
        self, step_index: int, store: CheckpointStore | None = None, key: str | None = None
    ) -> bytes:
        """Run through ``step_index`` and return the checkpoint bytes.

        With ``store`` and ``key``, the checkpoint is also written there.
        """
        if (store is None) != (key is None):
            raise ValueError("pass both store and key, or neither")
        blob = super().run_until_step(step_index)
        if store is not None and key is not None:
            store.put(key, blob)
        return blob

    def resume_from(
        self,
        checkpoint: bytes | None = None,
        *,
        store: CheckpointStore | None = None,
        key: str | None = None,
    ) -> list[Any]:
        """Resume from checkpoint bytes, or read them from ``store[key]``."""
        if checkpoint is None:
            if store is None or key is None:
                raise ValueError("pass checkpoint bytes, or both store and key")
            checkpoint = store.get(key)
        elif store is not None or key is not None:
            raise ValueError("pass checkpoint bytes or store/key, not both")
        return super().resume_from(checkpoint)
