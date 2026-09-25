"""Hit verifiers for the semantic cache tier.

Similarity search finds the cached prompt *closest* to a new one; a verifier
then decides whether that cached answer is actually *correct* for it. This
second stage is what keeps near-miss edits ("enable" vs "disable" two-factor
auth, "Australia" vs "Austria", "5" vs "50" euros) from being served the wrong
answer: embedders score those pairs as high as true paraphrases.

- :class:`~continuum._native.LexicalNearMissVerifier` (the default on every
  ``SemanticCacheIndex``) rejects minimal edits: swapped numbers, swapped
  content words with everything else unchanged, flipped polarity or
  negation. Fast and deterministic, but it cannot tell a synonym swap
  ("cancel" vs "end") from an antonym swap, so it also turns those down.
- :class:`LLMJudgeVerifier` asks a chat model whether both prompts have the
  same answer. It understands synonyms and antonyms, at the cost of one
  model call per candidate hit (verdicts are cached).
- :class:`AllOf` requires every verifier in a chain to accept. Note that
  chaining the lexical verifier in front of a judge keeps the lexical
  verifier's synonym rejections; use the judge alone when recall on
  rewordings matters.

Use one with ``SemanticCacheIndex.set_verifier(...)``; ``None`` disables
verification. Measurements: ``benchmarks/reports/semantic-false-hits.md``.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence

from continuum._native import HitVerifier, LexicalNearMissVerifier

__all__ = ["AllOf", "HitVerifier", "LLMJudgeVerifier", "LexicalNearMissVerifier"]

JUDGE_PROMPT = (
    "You decide whether a cached answer can be reused.\n"
    "Question A: {a}\n"
    "Question B: {b}\n"
    "Would one correct answer to A also be a correct and complete answer to B? "
    "Answer with exactly one word: YES or NO."
)


class LLMJudgeVerifier(HitVerifier):
    """Verify candidate hits with a chat model over an OpenAI-compatible API.

    Works with OpenAI, vLLM, and Ollama (``base_url="http://localhost:11434"``).
    Any failure (network, unexpected reply) rejects the hit: a miss is safe, a
    wrong answer is not.

    Args:
        base_url: Server root, e.g. ``"http://localhost:11434"``.
        model: Chat model name, e.g. ``"gemma4"``.
        api_key: Sent as a bearer token when given.
        timeout: Per-request timeout in seconds.
        prompt: Judge prompt with ``{a}`` and ``{b}`` placeholders.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        prompt: str = JUDGE_PROMPT,
    ) -> None:
        super().__init__()
        base = base_url.rstrip("/")
        self._url = (base if base.endswith("/v1") else base + "/v1") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._prompt = prompt
        self.calls = 0

    def verify(self, cached_prompt: str, new_prompt: str, similarity: float = 1.0) -> bool:
        self.calls += 1
        body = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 3,
            "messages": [
                {"role": "user", "content": self._prompt.format(a=cached_prompt, b=new_prompt)}
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                reply = json.load(resp)["choices"][0]["message"]["content"] or ""
        except Exception:
            return False
        return reply.strip().upper().startswith("YES")

    def name(self) -> str:
        return f"llm-judge/{self._model}"


class AllOf(HitVerifier):
    """Accept a hit only if every verifier accepts it (evaluated in order)."""

    def __init__(self, *verifiers: HitVerifier) -> None:
        super().__init__()
        if not verifiers:
            raise ValueError("AllOf needs at least one verifier")
        self._verifiers: Sequence[HitVerifier] = verifiers

    def verify(self, cached_prompt: str, new_prompt: str, similarity: float = 1.0) -> bool:
        return all(v.verify(cached_prompt, new_prompt, similarity) for v in self._verifiers)

    def name(self) -> str:
        return "all-of(" + ",".join(v.name() for v in self._verifiers) + ")"
