"""OpenTelemetry export of per-tier reuse events (issue #18)."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from continuum import DurableAgent
from continuum._native import (
    BackendRegistry,
    BruteForceEmbeddingProvider,
    LayerKVCacheIndex,
    MemoryGraphStore,
    MemoTable,
    SemanticCacheIndex,
    Session,
)
from continuum.telemetry import (
    ENV_VAR,
    CallbackObserver,
    OpenTelemetryObserver,
    auto_instrument,
    enabled_from_env,
    instrument,
)


def _full_session() -> Session:
    reg = BackendRegistry()
    reg.register_fake_llm()
    s = Session("otel", reg)
    s.set_memo_table(MemoTable(64, 0))
    s.set_semantic_cache(SemanticCacheIndex(64, 0.99))
    s.set_embedding_provider(BruteForceEmbeddingProvider(64))
    s.set_layer_cache(LayerKVCacheIndex(64, 1 << 20))
    s.set_memory_graph(MemoryGraphStore(64))
    return s


def test_no_observer_emits_nothing() -> None:
    events: list[dict[str, Any]] = []
    s = _full_session()
    obs = CallbackObserver(events.append)
    s.set_observer(obs)
    s.set_observer(None)
    s.generate(["hello"], "fake/m", 4)
    assert events == []


def test_events_cover_every_tier_and_node() -> None:
    events: list[dict[str, Any]] = []
    s = _full_session()
    s.set_observer(CallbackObserver(events.append))
    s.generate(["the shared system prompt. question one"], "fake/m", 4)
    first = list(events)
    tiers = [e["tier"] for e in first if e["kind"] == "tier_lookup"]
    assert tiers == ["memo", "semantic", "prefix_kv", "layer_kv", "memory_graph"]
    assert not any(e["hit"] for e in first if e["kind"] == "tier_lookup")
    token_node = [
        e for e in first if e["kind"] == "node_execution" and e["node_kind"] == "TokenOp"
    ][0]
    assert token_node["served_by"] == "backend"
    assert token_node["backend"] == "fake"
    assert token_node["model_id"] == "fake/m"
    for e in first:
        assert e["end_unix_ns"] >= e["start_unix_ns"] > 0

    events.clear()
    s.generate(["the shared system prompt. question one"], "fake/m", 4)  # exact repeat
    memo = [e for e in events if e["kind"] == "tier_lookup"]
    assert [e["tier"] for e in memo] == ["memo"] and memo[0]["hit"]
    token_node = [
        e for e in events if e["node_kind"] == "TokenOp" and e["kind"] == "node_execution"
    ][0]
    assert token_node["served_by"] == "memo"
    assert token_node["tokens_saved"] == token_node["total_tokens"] > 0


def test_callback_errors_do_not_break_execution() -> None:
    def boom(_: dict[str, Any]) -> None:
        raise ValueError("observer bug")

    s = _full_session()
    s.set_observer(CallbackObserver(boom))
    assert s.generate(["x"], "fake/m", 4) is not None


def _providers() -> tuple[
    TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
]:
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    return tp, exporter, MeterProvider(metric_readers=[reader]), reader


def _metric_points(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    data = reader.get_metrics_data()
    assert data is not None
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out[metric.name] = list(metric.data.data_points)
    return out


def test_spans_and_metrics() -> None:
    tp, exporter, mp, reader = _providers()
    s = _full_session()
    obs = instrument(s, tracer_provider=tp, meter_provider=mp)
    assert isinstance(obs, OpenTelemetryObserver)
    s.generate(["repeat me"], "fake/m", 4)
    s.generate(["repeat me"], "fake/m", 4)

    spans = exporter.get_finished_spans()
    nodes = [sp for sp in spans if sp.name == "continuum.node"]
    token_spans = [sp for sp in nodes if sp.attributes["continuum.node.kind"] == "TokenOp"]
    assert [sp.attributes["continuum.served_by"] for sp in token_spans] == ["backend", "memo"]
    assert token_spans[0].attributes["continuum.backend"] == "fake"
    assert token_spans[0].attributes["gen_ai.request.model"] == "fake/m"

    children = [sp for sp in spans if sp.name.startswith("continuum.reuse.")]
    by_parent: dict[int, list[str]] = {}
    for sp in children:
        assert sp.parent is not None
        by_parent.setdefault(sp.parent.span_id, []).append(sp.name)
    assert by_parent[token_spans[1].context.span_id] == ["continuum.reuse.memo"]
    assert len(by_parent[token_spans[0].context.span_id]) == 5

    points = _metric_points(reader)
    lookups = {
        p.attributes["continuum.reuse.tier"]: p.value for p in points["continuum.reuse.lookups"]
    }
    assert lookups["memo"] == 2 and lookups["prefix_kv"] == 1
    hits = {p.attributes["continuum.reuse.tier"]: p.value for p in points["continuum.reuse.hits"]}
    assert hits == {"memo": 1}
    assert sum(p.value for p in points["continuum.reuse.tokens_saved"]) > 0
    assert sum(p.count for p in points["continuum.reuse.lookup.duration"]) == 6
    # Prompt values arrive as graph inputs, so only the two TokenOps execute.
    assert sum(p.value for p in points["continuum.node.executions"]) == 2


def test_env_var_instruments_durable_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry import trace

    tp, exporter, _, _ = _providers()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tp)

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert not enabled_from_env()
    assert auto_instrument(_full_session()) is None

    monkeypatch.setenv(ENV_VAR, "1")
    assert enabled_from_env()
    agent = DurableAgent()
    agent.begin(["a", "b"])
    agent.resume_from(agent.run_until_step(0))
    spans = exporter.get_finished_spans()
    kinds = {sp.attributes["continuum.node.kind"] for sp in spans if sp.name == "continuum.node"}
    assert kinds == {"TokenOp"}
    assert any(sp.name == "continuum.reuse.memo" for sp in spans)


def test_env_var_without_otel_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    import continuum.telemetry as tel

    def missing(*_: Any, **__: Any) -> Any:
        raise ImportError("no otel")

    monkeypatch.setenv(ENV_VAR, "true")
    monkeypatch.setattr(tel, "instrument", missing)
    with pytest.warns(RuntimeWarning, match="opentelemetry is not installed"):
        assert tel.auto_instrument(_full_session()) is None


def test_exporter_errors_are_swallowed() -> None:
    class Broken:
        def get_tracer(self, *a: Any, **k: Any) -> Any:
            raise_on = self

            class T:
                def start_span(self, *a: Any, **k: Any) -> Any:
                    raise RuntimeError(f"exporter down {raise_on!r}")

            return T()

    _, _, mp, _ = _providers()
    s = _full_session()
    instrument(s, tracer_provider=Broken(), meter_provider=mp)
    assert s.generate(["x"], "fake/m", 4) is not None
