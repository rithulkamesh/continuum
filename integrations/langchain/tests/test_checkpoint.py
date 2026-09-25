from __future__ import annotations

import asyncio
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from continuum.checkpoints import LocalDirectoryStore
from continuum_langchain import ContinuumCheckpointSaver, ContinuumLLM
from continuum_langchain.llm import default_session


class State(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]


def _graph(llm: ContinuumLLM, saver: ContinuumCheckpointSaver, interrupt: bool = False) -> Any:
    def research(state: State) -> dict[str, Any]:
        return {"notes": [f"research:{llm.invoke('research ' + state['topic'])[:20]}"]}

    def outline(state: State) -> dict[str, Any]:
        return {"notes": [f"outline:{llm.invoke('outline ' + state['topic'])[:20]}"]}

    def draft(state: State) -> dict[str, Any]:
        return {
            "notes": [f"draft:{llm.invoke('draft ' + state['topic'] + ' ' + state['notes'][-1])}"]
        }

    g = StateGraph(State)
    g.add_node("research", research)
    g.add_node("outline", outline)
    g.add_node("draft", draft)
    g.add_edge(START, "research")
    g.add_edge("research", "outline")
    g.add_edge("outline", "draft")
    g.add_edge("draft", END)
    return g.compile(checkpointer=saver, interrupt_before=["draft"] if interrupt else None)


def test_resume_in_fresh_process_starts_warm(tmp_path: Path) -> None:
    store = LocalDirectoryStore(tmp_path)
    llm = ContinuumLLM()
    saver = ContinuumCheckpointSaver(store, session=llm.session)
    cfg = {"configurable": {"thread_id": "t1"}}
    _graph(llm, saver, interrupt=True).invoke({"topic": "kv caches", "notes": []}, cfg)
    assert llm.session.cache_size() > 0

    # "New process": new session with a cold cache, new saver on the same store.
    llm2 = ContinuumLLM(session=default_session("revived"))
    assert llm2.session.cache_size() == 0
    saver2 = ContinuumCheckpointSaver(store, session=llm2.session)
    final = _graph(llm2, saver2, interrupt=True).invoke(None, cfg)
    assert [n.split(":")[0] for n in final["notes"]] == ["research", "outline", "draft"]
    assert llm2.session.cache_size() > 0  # KV index came back with the checkpoint

    history = list(saver2.list(cfg))
    assert len(history) >= 4
    assert history[0].checkpoint["id"] > history[-1].checkpoint["id"]  # newest first
    assert list(saver2.list(cfg, limit=2)).__len__() == 2
    assert all(
        t.metadata.get("source") == "loop" for t in saver2.list(cfg, filter={"source": "loop"})
    )
    assert saver2.get_tuple({"configurable": {"thread_id": "missing"}}) is None


def test_fork_keeps_lineage_and_branches_independently(tmp_path: Path) -> None:
    store = LocalDirectoryStore(tmp_path)
    llm = ContinuumLLM()
    saver = ContinuumCheckpointSaver(store, session=llm.session)
    graph = _graph(llm, saver)
    cfg = {"configurable": {"thread_id": "t"}}
    original = graph.invoke({"topic": "tries", "notes": []}, cfg)

    # Rewind to the checkpoint before "draft" ran and change the outline.
    before_draft = next(s for s in graph.get_state_history(cfg) if s.next == ("draft",))
    fork_cfg = graph.update_state(before_draft.config, {"notes": ["outline:EDITED"]})
    forked = graph.invoke(None, fork_cfg)

    assert forked["notes"][-1] != original["notes"][-1]  # draft saw the edit
    assert forked["notes"][:2] == original["notes"][:2]
    fork_tuple = saver.get_tuple(fork_cfg)
    assert fork_tuple is not None and fork_tuple.parent_config is not None
    assert (
        fork_tuple.parent_config["configurable"]["checkpoint_id"]
        == before_draft.config["configurable"]["checkpoint_id"]
    )
    # The original branch is untouched.
    orig_final = saver.get_tuple(
        {
            "configurable": {
                "thread_id": "t",
                "checkpoint_id": list(graph.get_state_history(cfg))[-1].config["configurable"][
                    "checkpoint_id"
                ],
            }
        }
    )
    assert orig_final is not None
    ids = [t.checkpoint["id"] for t in saver.list(cfg)]
    assert len(ids) == len(set(ids))


def test_writes_threads_and_async(tmp_path: Path) -> None:
    saver = ContinuumCheckpointSaver(LocalDirectoryStore(tmp_path))
    graph = _graph(ContinuumLLM(), saver)
    graph.invoke({"topic": "a", "notes": []}, {"configurable": {"thread_id": "one"}})

    async def run() -> list[Any]:
        await graph.ainvoke({"topic": "b", "notes": []}, {"configurable": {"thread_id": "two"}})
        return [t async for t in saver.alist({"configurable": {"thread_id": "two"}})]

    two = asyncio.run(run())
    assert two
    assert {t.config["configurable"]["thread_id"] for t in saver.list(None)} == {"one", "two"}
    latest = asyncio.run(saver.aget_tuple({"configurable": {"thread_id": "one"}}))
    assert latest is not None and latest.checkpoint["channel_values"]["topic"] == "a"
    asyncio.run(saver.aput_writes(latest.config, [("notes", ["x"])], "task-1"))
    assert saver.get_tuple(latest.config).pending_writes[-1] == ("task-1", "notes", ["x"])
    asyncio.run(saver.adelete_thread("one"))
    assert saver.get_tuple({"configurable": {"thread_id": "one"}}) is None
    assert saver.get_tuple({"configurable": {"thread_id": "two", "checkpoint_id": "nope"}}) is None


def test_special_writes_overwrite(tmp_path: Path) -> None:
    saver = ContinuumCheckpointSaver(LocalDirectoryStore(tmp_path))
    graph = _graph(ContinuumLLM(), saver)
    cfg = {"configurable": {"thread_id": "w"}}
    graph.invoke({"topic": "a", "notes": []}, cfg)
    latest = saver.get_tuple(cfg)
    assert latest is not None
    saver.put_writes(latest.config, [("__error__", "first")], "t")
    saver.put_writes(latest.config, [("__error__", "second")], "t")
    errors = [w for w in saver.get_tuple(latest.config).pending_writes if w[1] == "__error__"]
    assert errors == [("t", "__error__", "second")]


@pytest.mark.parametrize("version", [None, 3, "00000000000000000000000000000007.123"])
def test_next_version_monotonic(version: Any) -> None:
    saver = ContinuumCheckpointSaver(LocalDirectoryStore("/tmp/unused-continuum-lc"))
    nxt = saver.get_next_version(version, None)
    base = 0 if version is None else int(str(version).split(".")[0])
    assert int(nxt.split(".")[0]) == base + 1
