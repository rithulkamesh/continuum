"""OpenTelemetry export for per-tier reuse events.

The interpreter can report every reuse-tier lookup (memo, semantic, prefix KV,
layer KV, memory graph) and every node it executes as a
:class:`~continuum._native.ReuseEvent`. Nothing is emitted unless an observer
is attached, so telemetry costs nothing when off.

:class:`OpenTelemetryObserver` turns those events into:

- **spans**: one per executed node (``continuum.node``), with one child span
  per tier lookup (``continuum.reuse.<tier>``), timed with the interpreter's
  own clock readings;
- **metrics**: ``continuum.reuse.lookups`` / ``continuum.reuse.hits`` /
  ``continuum.reuse.tokens_saved`` counters and a
  ``continuum.reuse.lookup.duration`` histogram (ms), all by tier, plus a
  ``continuum.node.executions`` counter by node kind and ``served_by``.

Turn it on per session, or for every :class:`~continuum.DurableAgent` with an
environment variable::

    from continuum.telemetry import instrument
    instrument(session)                  # uses the global OTel providers

    CONTINUUM_OTEL=1 python my_agent.py  # DurableAgent instances self-instrument

Requires ``opentelemetry-api`` (and an SDK / exporter to send data anywhere):
``pip install "continuum-ai[otel]"``.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, Protocol

from continuum._native import ReuseEvent, ReuseEventKind, ReuseObserver

__all__ = [
    "ENV_VAR",
    "CallbackObserver",
    "OpenTelemetryObserver",
    "auto_instrument",
    "enabled_from_env",
    "event_to_dict",
    "instrument",
]

ENV_VAR = "CONTINUUM_OTEL"
_log = logging.getLogger(__name__)

_FIELDS = (
    "tier",
    "node_name",
    "node_kind",
    "backend",
    "model_id",
    "cache_namespace",
    "hit",
    "served_by",
    "similarity",
    "match_len",
    "total_tokens",
    "tokens_saved",
    "tokens_sent",
    "reused_prefix_len",
    "compute_steps",
    "used_cached_state",
    "start_unix_ns",
    "end_unix_ns",
)


class _Observable(Protocol):
    def set_observer(self, observer: ReuseObserver | None) -> None: ...


def event_to_dict(event: ReuseEvent) -> dict[str, Any]:
    """Copy an event into a plain dict (events are only valid during the callback)."""
    out: dict[str, Any] = {name: getattr(event, name) for name in _FIELDS}
    out["kind"] = "tier_lookup" if event.kind == ReuseEventKind.TierLookup else "node_execution"
    return out


class CallbackObserver(ReuseObserver):
    """Call ``fn(event_dict)`` for every event; exceptions are logged, not raised."""

    def __init__(self, fn: Callable[[dict[str, Any]], None]) -> None:
        super().__init__()
        self._fn = fn

    def on_event(self, event: ReuseEvent) -> None:
        try:
            self._fn(event_to_dict(event))
        except Exception:  # never propagate into the engine
            _log.exception("continuum reuse observer callback failed")


class OpenTelemetryObserver(ReuseObserver):
    """Export reuse events as OpenTelemetry spans and metrics.

    Args:
        tracer_provider: Defaults to the global ``opentelemetry.trace`` provider.
        meter_provider: Defaults to the global ``opentelemetry.metrics`` provider.
    """

    def __init__(self, tracer_provider: Any = None, meter_provider: Any = None) -> None:
        super().__init__()
        from opentelemetry import metrics, trace

        from continuum import __version__

        self._trace = trace
        tp = tracer_provider or trace.get_tracer_provider()
        mp = meter_provider or metrics.get_meter_provider()
        self._tracer = tp.get_tracer("continuum", __version__)
        meter = mp.get_meter("continuum", __version__)
        self._lookups = meter.create_counter(
            "continuum.reuse.lookups", unit="{lookup}", description="Reuse-tier lookups"
        )
        self._hits = meter.create_counter(
            "continuum.reuse.hits", unit="{hit}", description="Reuse-tier lookups that matched"
        )
        self._saved = meter.create_counter(
            "continuum.reuse.tokens_saved", unit="{token}", description="Prompt tokens a tier saved"
        )
        self._latency = meter.create_histogram(
            "continuum.reuse.lookup.duration", unit="ms", description="Reuse-tier lookup latency"
        )
        self._nodes = meter.create_counter(
            "continuum.node.executions", unit="{node}", description="Executed IR nodes"
        )
        self._pending: list[dict[str, Any]] = []

    def on_event(self, event: ReuseEvent) -> None:
        try:
            ev = event_to_dict(event)
            if ev["kind"] == "tier_lookup":
                self._record_tier(ev)
            else:
                self._record_node(ev)
        except Exception:  # never propagate into the engine
            _log.exception("continuum OpenTelemetry export failed")

    def _record_tier(self, ev: dict[str, Any]) -> None:
        attrs = {"continuum.reuse.tier": ev["tier"], "continuum.model_id": ev["model_id"]}
        self._lookups.add(1, attrs)
        if ev["hit"]:
            self._hits.add(1, attrs)
        if ev["tokens_saved"]:
            self._saved.add(ev["tokens_saved"], attrs)
        self._latency.record((ev["end_unix_ns"] - ev["start_unix_ns"]) / 1e6, attrs)
        self._pending.append(ev)

    def _record_node(self, ev: dict[str, Any]) -> None:
        self._nodes.add(
            1, {"continuum.node.kind": ev["node_kind"], "continuum.served_by": ev["served_by"]}
        )
        attrs: dict[str, Any] = {
            "continuum.node.name": ev["node_name"],
            "continuum.node.kind": ev["node_kind"],
            "continuum.served_by": ev["served_by"],
            "continuum.cache_namespace": ev["cache_namespace"],
        }
        if ev["backend"]:
            attrs["continuum.backend"] = ev["backend"]
        if ev["model_id"]:
            attrs["continuum.model_id"] = ev["model_id"]
            attrs["gen_ai.request.model"] = ev["model_id"]
        if ev["node_kind"] == "TokenOp" or ev["served_by"] == "backend":
            attrs.update(
                {
                    "continuum.tokens.total": ev["total_tokens"],
                    "continuum.tokens.saved": ev["tokens_saved"],
                    "continuum.tokens.sent": ev["tokens_sent"],
                    "continuum.reused_prefix_len": ev["reused_prefix_len"],
                    "continuum.compute_steps": ev["compute_steps"],
                    "continuum.used_cached_state": ev["used_cached_state"],
                }
            )
        span = self._tracer.start_span(
            "continuum.node", start_time=ev["start_unix_ns"], attributes=attrs
        )
        ctx = self._trace.set_span_in_context(span)
        for tier in self._pending:
            child = self._tracer.start_span(
                f"continuum.reuse.{tier['tier']}",
                context=ctx,
                start_time=tier["start_unix_ns"],
                attributes={
                    "continuum.reuse.tier": tier["tier"],
                    "continuum.reuse.hit": tier["hit"],
                    "continuum.reuse.tokens_saved": tier["tokens_saved"],
                    "continuum.reuse.match_len": tier["match_len"],
                    "continuum.reuse.similarity": tier["similarity"],
                },
            )
            child.end(end_time=tier["end_unix_ns"])
        self._pending.clear()
        span.end(end_time=ev["end_unix_ns"])


def instrument(
    target: _Observable, tracer_provider: Any = None, meter_provider: Any = None
) -> OpenTelemetryObserver:
    """Attach an :class:`OpenTelemetryObserver` to a ``Session`` or ``DurableAgent``."""
    observer = OpenTelemetryObserver(tracer_provider, meter_provider)
    target.set_observer(observer)
    return observer


def enabled_from_env() -> bool:
    """True when ``CONTINUUM_OTEL`` is ``1`` / ``true`` / ``yes`` / ``on``."""
    return os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def auto_instrument(target: _Observable) -> OpenTelemetryObserver | None:
    """Instrument ``target`` if ``CONTINUUM_OTEL`` is set; otherwise do nothing."""
    if not enabled_from_env():
        return None
    try:
        return instrument(target)
    except ImportError:
        warnings.warn(
            f"{ENV_VAR} is set but opentelemetry is not installed; "
            'install "continuum-ai[otel]" to export reuse telemetry',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
