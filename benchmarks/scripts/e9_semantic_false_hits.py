"""E9: semantic-cache false hits vs threshold, per embedder and hit verifier.

Sweeps the similarity threshold over labeled paraphrase / near-miss /
unrelated prompt pairs (see ``continuum.benchmarks.semantic_eval``) for every
combination of:

- embedder: the bundled char-n-gram provider, WordLlama (if installed), and
  optionally any OpenAI-compatible embeddings endpoint;
- hit verifier: none, the default ``LexicalNearMissVerifier``, and optionally
  an ``LLMJudgeVerifier`` on a chat endpoint;
- dataset: ``dev`` (used to design the verifier), ``validation`` (used once to
  diagnose it), ``test`` (never used for tuning; the headline numbers).

Run offline::

    pip install "continuum-ai[semantic]"
    PYTHONPATH=python python benchmarks/scripts/e9_semantic_false_hits.py

Add an Ollama embedder and judge::

    ... e9_semantic_false_hits.py --base-url http://localhost:11434 \\
        --embed-model nomic-embed-text --judge-model gemma4

Writes benchmarks/data/e9_semantic_false_hits.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from continuum._native import BruteForceEmbeddingProvider, EmbeddingProvider, HitVerifier
from continuum.benchmarks.semantic_eval import evaluate, load_pairs, summarize
from continuum.verifiers import LexicalNearMissVerifier, LLMJudgeVerifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "data"
DATASETS = {
    "dev": DATA / "semantic_pairs.json",
    "validation": DATA / "semantic_pairs_validation.json",
    "test": DATA / "semantic_pairs_test.json",
}


def embedders(args: argparse.Namespace) -> list[EmbeddingProvider]:
    out: list[EmbeddingProvider] = [BruteForceEmbeddingProvider(64)]
    try:
        from continuum.embeddings import WordLlamaEmbeddingProvider

        out.append(WordLlamaEmbeddingProvider())
    except ImportError:
        print('(skipping WordLlama: pip install "continuum-ai[semantic]")')
    if args.base_url and args.embed_model:
        from continuum.embeddings import OpenAICompatibleEmbeddingProvider

        out.append(
            OpenAICompatibleEmbeddingProvider(args.base_url, args.embed_model, api_key=args.api_key)
        )
    return out


def verifiers(args: argparse.Namespace) -> list[HitVerifier | None]:
    out: list[HitVerifier | None] = [None, LexicalNearMissVerifier()]
    if args.base_url and args.judge_model:
        out.append(LLMJudgeVerifier(args.base_url, args.judge_model, api_key=args.api_key))
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", help="OpenAI-compatible server (Ollama, vLLM, OpenAI)")
    ap.add_argument("--embed-model", help="embedding model on --base-url")
    ap.add_argument("--judge-model", help="chat model on --base-url for the LLM-judge verifier")
    ap.add_argument("--api-key")
    ap.add_argument("--out", default=str(DATA / "e9_semantic_false_hits.json"))
    args = ap.parse_args(argv)

    results = []
    for embedder in embedders(args):
        for verifier in verifiers(args):
            for name, path in DATASETS.items():
                full = evaluate(load_pairs(path), embedder, verifier=verifier)
                results.append(
                    {
                        "dataset": name,
                        **{k: full[k] for k in ("embedder", "verifier", "pairs")},
                        **summarize(full),
                        "sweep": full["thresholds"],
                        "similarity": full["similarity"],
                    }
                )

    print(
        f"{'embedder':<30} {'verifier':<24} {'dataset':<10} {'zero-FH thr':>11} {'recall@0FH':>10} "
        f"{'FHR@0.85':>8} {'recall@0.85':>11}"
    )
    for r in results:
        thr = (
            "-" if r["zero_false_hit_threshold"] is None else f"{r['zero_false_hit_threshold']:.2f}"
        )
        rec = (
            "-"
            if r["recall_at_zero_false_hits"] is None
            else f"{r['recall_at_zero_false_hits']:.2f}"
        )
        print(
            f"{r['embedder']:<30} {str(r['verifier'] or 'none'):<24} {r['dataset']:<10} {thr:>11} {rec:>10} "
            f"{r['false_hit_rate_at_default']:>8.3f} {r['recall_at_default']:>11.2f}"
        )
    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"experiment": "E9: semantic false hits by embedder and verifier", "results": results},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return {"results": results}


if __name__ == "__main__":
    main()
