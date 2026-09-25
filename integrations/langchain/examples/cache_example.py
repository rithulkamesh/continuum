"""ContinuumCache as LangChain's LLM cache, against Continuum's offline FakeLLM.

pip install -e integrations/langchain
python integrations/langchain/examples/cache_example.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from langchain_core.globals import set_llm_cache  # noqa: E402

from continuum_langchain import ContinuumCache, ContinuumLLM  # noqa: E402

QUESTIONS = [
    "how do I reset my password",
    "what is the refund policy",
    "how do I reset my password",  # exact repeat -> memo hit
    "what is the refund policy",  # exact repeat -> memo hit
]


def main() -> None:
    cache = ContinuumCache()
    set_llm_cache(cache)
    llm = ContinuumLLM(max_tokens=8)
    for q in QUESTIONS:
        llm.invoke(q)
    print(f"cache stats: {cache.stats}")
    assert cache.stats["memo_hits"] == 2 and cache.stats["misses"] == 2
    print("langchain cache: OK")


if __name__ == "__main__":
    main()
