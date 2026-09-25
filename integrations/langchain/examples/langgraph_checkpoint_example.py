"""ContinuumCheckpointSaver: pause a LangGraph run, resume it warm in a fresh
session, then fork it from a past step. Offline, on Continuum's FakeLLM.

    pip install -e integrations/langchain langgraph
    python integrations/langchain/examples/langgraph_checkpoint_example.py
"""

from __future__ import annotations

import operator
import os
import shutil
import tempfile
from typing import Annotated, Any, TypedDict

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from continuum.checkpoints import LocalDirectoryStore  # noqa: E402
from continuum_langchain import ContinuumCheckpointSaver, ContinuumLLM  # noqa: E402
from continuum_langchain.llm import default_session  # noqa: E402


class State(TypedDict):
    task: str
    steps: Annotated[list[str], operator.add]


def build(llm: ContinuumLLM, saver: ContinuumCheckpointSaver) -> Any:
    def plan(state: State) -> dict[str, Any]:
        return {"steps": [f"plan -> {llm.invoke('plan: ' + state['task'])}"]}

    def act(state: State) -> dict[str, Any]:
        return {"steps": [f"act -> {llm.invoke('act on: ' + state['steps'][-1])}"]}

    def report(state: State) -> dict[str, Any]:
        return {"steps": [f"report -> {llm.invoke('report: ' + state['steps'][-1])}"]}

    g = StateGraph(State)
    for name, fn in (("plan", plan), ("act", act), ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=saver, interrupt_before=["report"])


def main() -> None:
    root = tempfile.mkdtemp(prefix="continuum-langgraph-")
    store = LocalDirectoryStore(root)  # or S3Store / GCSStore
    cfg = {"configurable": {"thread_id": "ticket-42"}}

    llm = ContinuumLLM(max_tokens=6)
    build(llm, ContinuumCheckpointSaver(store, session=llm.session)).invoke(
        {"task": "fix the flaky test", "steps": []}, cfg
    )
    print(f"paused before 'report'; warm KV entries: {llm.session.cache_size()}")

    # A fresh session (a new process, another machine) resumes the thread.
    llm2 = ContinuumLLM(max_tokens=6, session=default_session("worker-2"))
    graph = build(llm2, ContinuumCheckpointSaver(store, session=llm2.session))
    final = graph.invoke(None, cfg)
    print(f"resumed: {len(final['steps'])} steps; KV restored: {llm2.session.cache_size()} entries")
    assert llm2.session.cache_size() > 0

    # Fork: rewind to before 'act', replace the plan, run the alternate branch.
    before_act = next(s for s in graph.get_state_history(cfg) if s.next == ("act",))
    fork = graph.update_state(before_act.config, {"steps": ["plan -> quarantine the test"]})
    graph.invoke(None, fork)  # runs 'act' on the new plan, pauses before 'report'
    alt = graph.invoke(None, fork)  # runs 'report'
    print(f"fork: {alt['steps'][-2][:40]}...")
    assert alt["steps"][-1] != final["steps"][-1]
    shutil.rmtree(root)
    print("langgraph checkpointer: OK")


if __name__ == "__main__":
    main()
