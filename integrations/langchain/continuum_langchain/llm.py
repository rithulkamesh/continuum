"""A LangChain LLM that runs prompts through a Continuum ``Session``."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict, Field

from continuum._native import BackendRegistry, MemoTable, Session


def default_session(session_id: str = "langchain") -> Session:
    """FakeLLM session (offline, deterministic), or the vLLM / Ollama shim
    when ``VLLM_BASE_URL`` is set. The memo tier is attached."""
    registry = BackendRegistry()
    registry.register_fake_llm()
    if os.environ.get("VLLM_BASE_URL"):
        registry.register_vllm()
    session = Session(session_id, registry)
    session.set_memo_table(MemoTable(4096, 0))
    return session


class ContinuumLLM(LLM):
    """Text-completion LLM backed by the Continuum reuse stack.

    Every call is one ``TokenOp`` through ``session``: exact repeats hit the
    memo tier and shared prefixes hit the prefix-KV tier. With the default
    FakeLLM backend, output is the generated token ids joined by spaces.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = "fake/model"
    max_tokens: int = 32
    temperature: float = 0.0
    session: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        if self.session is None:
            self.session = default_session()

    @property
    def _llm_type(self) -> str:
        return "continuum"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens, "temperature": self.temperature}

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        out = self.session.generate([prompt], self.model_id, self.max_tokens, self.temperature)
        text = out if isinstance(out, str) else " ".join(str(t) for t in out)
        if stop:
            for s in stop:
                cut = text.find(s)
                if cut != -1:
                    text = text[:cut]
        return text
