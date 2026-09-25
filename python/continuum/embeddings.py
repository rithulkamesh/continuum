"""Embedding providers for the semantic cache and memory-graph tiers.

Every provider subclasses :class:`continuum._native.EmbeddingProvider`, so it
can be handed straight to ``Session.set_embedding_provider``. Each one reports
an :meth:`identity`; the semantic cache stores it with every entry and only
compares vectors that share it, so switching embedders never produces a false
hit against vectors from another embedding space.

Three shapes cover the common cases:

- :class:`CallableEmbeddingProvider` wraps any ``text -> vector`` function,
  e.g. a local sentence-transformers model.
- :class:`OpenAICompatibleEmbeddingProvider` calls a hosted ``/v1/embeddings``
  endpoint (OpenAI, vLLM, Ollama, LM Studio, ...).
- :class:`PrecomputedEmbeddingProvider` serves vectors computed ahead of time,
  for fully reproducible runs and offline evaluation.
- :class:`WordLlamaEmbeddingProvider` is a small semantic model that runs
  locally on CPU with no download (``pip install "continuum-ai[semantic]"``).
  It is the recommended starting point for the semantic tier.

Pair any of them with the semantic tier's hit verifier (on by default, see
:mod:`continuum.verifiers`): similarity alone cannot tell a paraphrase from a
near-miss edit such as "enable" vs "disable".

The built-in :class:`continuum._native.BruteForceEmbeddingProvider` (character
n-gram hashing, identity ``continuum/char-ngram-v1:<dim>``) stays the default
in examples: dependency-free and deterministic, but lexical rather than
semantic.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from continuum._native import EmbeddingProvider

__all__ = [
    "CallableEmbeddingProvider",
    "EmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "PrecomputedEmbeddingProvider",
    "WordLlamaEmbeddingProvider",
]


def _checked(vector: Sequence[float], dimension: int, source: str) -> list[float]:
    out = [float(x) for x in vector]
    if len(out) != dimension:
        raise ValueError(f"{source} returned {len(out)} dims, expected {dimension}")
    return out


class CallableEmbeddingProvider(EmbeddingProvider):
    """Adapt a ``text -> vector`` callable (a local model, a cached lookup, ...).

    Args:
        fn: Returns the embedding of one string.
        dimension: Length of every vector ``fn`` returns; checked on each call.
        identity: Stable name of the embedding space, e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``. Change it whenever
            the vectors would change (new model, new normalization, ...).
    """

    def __init__(self, fn: Callable[[str], Sequence[float]], dimension: int, identity: str) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not identity:
            raise ValueError("identity must be non-empty")
        self._fn = fn
        self._dimension = dimension
        self._identity = identity

    def embed(self, text: str) -> list[float]:
        return _checked(self._fn(text), self._dimension, self._identity)

    def dimension(self) -> int:
        return self._dimension

    def identity(self) -> str:
        return self._identity


class PrecomputedEmbeddingProvider(EmbeddingProvider):
    """Serve vectors computed ahead of time, keyed by the exact prompt text.

    Args:
        vectors: Prompt text to embedding. All vectors must share one length.
        identity: Name of the embedder that produced ``vectors``.
        fallback: Provider for texts not in ``vectors``. It must report the
            same identity and dimension; without one, unknown text raises
            ``KeyError``.
    """

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        identity: str,
        fallback: EmbeddingProvider | None = None,
    ) -> None:
        super().__init__()
        if not vectors:
            raise ValueError("vectors must be non-empty")
        dims = {len(v) for v in vectors.values()}
        if len(dims) != 1:
            raise ValueError(f"vectors have mixed dimensions: {sorted(dims)}")
        self._dimension = dims.pop()
        self._vectors = {k: [float(x) for x in v] for k, v in vectors.items()}
        self._identity = identity
        if fallback is not None and (
            fallback.identity() != identity or fallback.dimension() != self._dimension
        ):
            raise ValueError("fallback must share the identity and dimension")
        self._fallback = fallback

    def embed(self, text: str) -> list[float]:
        vec = self._vectors.get(text)
        if vec is not None:
            return list(vec)
        if self._fallback is None:
            raise KeyError(f"no precomputed embedding for {text!r}")
        return _checked(self._fallback.embed(text), self._dimension, self._identity)

    def dimension(self) -> int:
        return self._dimension

    def identity(self) -> str:
        return self._identity


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Call an OpenAI-format ``POST {base_url}/v1/embeddings`` endpoint.

    Works with OpenAI, vLLM (``--task embed``), Ollama, and any server that
    speaks the same wire format.

    Args:
        base_url: Server root, e.g. ``"http://localhost:11434"``.
        model: Embedding model name sent in the request.
        dimension: Expected vector length. ``None`` probes the server once.
        api_key: Sent as a bearer token when given.
        identity: Defaults to ``"openai-compatible/<model>"``; the host is
            left out on purpose, since one model yields one embedding space
            wherever it is served.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int | None = None,
        api_key: str | None = None,
        identity: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._identity = identity or f"openai-compatible/{model}"
        self._dimension = dimension if dimension is not None else len(self._request("probe"))

    def _request(self, text: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = json.dumps({"model": self._model, "input": text}).encode()
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = json.load(resp)
        return [float(x) for x in payload["data"][0]["embedding"]]

    def embed(self, text: str) -> list[float]:
        return _checked(self._request(text), self._dimension, self._identity)

    def dimension(self) -> int:
        return self._dimension

    def identity(self) -> str:
        return self._identity


class WordLlamaEmbeddingProvider(EmbeddingProvider):
    """Local semantic embeddings from WordLlama (static token embeddings
    distilled from an LLM's input layer; ~16 MB, CPU, sub-millisecond).

    The model files ship inside the ``wordllama`` wheel, so this works with no
    network access. Requires ``pip install "continuum-ai[semantic]"``.

    Args:
        config: WordLlama model config (``"l2_supercat"`` ships in the wheel).
        dim: Embedding width (the bundled model is 256; smaller truncates).
    """

    def __init__(self, config: str = "l2_supercat", dim: int = 256) -> None:
        super().__init__()
        try:
            import wordllama
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                'WordLlamaEmbeddingProvider needs wordllama: pip install "continuum-ai[semantic]"'
            ) from exc
        self._model = self._load_bundled(wordllama, config, dim)
        self._dimension = dim
        self._identity = f"wordllama/{config}:{dim}"

    @staticmethod
    def _load_bundled(wordllama: Any, config: str, dim: int) -> Any:
        from pathlib import Path

        pkg = Path(wordllama.__file__).parent
        tokenizer_file = pkg / "tokenizers" / f"{config}_tokenizer_config.json"
        weights_file = pkg / "weights" / f"{config}_256.safetensors"
        if tokenizer_file.is_file() and weights_file.is_file():
            from safetensors import safe_open
            from tokenizers import Tokenizer
            from wordllama.inference import WordLlamaInference

            with safe_open(str(weights_file), framework="np", device="cpu") as f:
                embedding = f.get_tensor("embedding.weight")[:, :dim]
            return WordLlamaInference(embedding, Tokenizer.from_file(str(tokenizer_file)))
        return wordllama.WordLlama.load(config=config, dim=dim)  # pragma: no cover - downloads

    def embed(self, text: str) -> list[float]:
        vec = [float(x) for x in self._model.embed([text])[0]]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def dimension(self) -> int:
        return self._dimension

    def identity(self) -> str:
        return self._identity
