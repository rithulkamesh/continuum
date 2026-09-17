"""Public durable-agent API.

:class:`DurableAgent` checkpoints mid-run (value map + portable KV state),
resumes on a fresh process, and forks alternate timelines by editing a
computed node. The implementation lives in the compiled extension; this
module re-exports it with a stable import path.
"""

from __future__ import annotations

from continuum._native import DurableAgent as DurableAgent

__all__ = ["DurableAgent"]
