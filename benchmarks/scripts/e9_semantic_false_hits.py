"""E9: semantic-cache false-hit rate vs similarity threshold (issue #10).

Sweeps the similarity threshold over a labeled set of paraphrase / near-miss /
unrelated prompt pairs and reports precision, recall, and false-hit rate. See
``continuum.benchmarks.semantic_eval`` for the method.

Default embedder is the bundled char-n-gram provider (offline, deterministic).
Evaluate a real one against any OpenAI-compatible embeddings endpoint:

    PYTHONPATH=python python benchmarks/scripts/e9_semantic_false_hits.py \\
        --embed-base-url http://localhost:11434 --embed-model nomic-embed-text

Writes benchmarks/data/e9_semantic_false_hits[_<slug>].json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from continuum._native import BruteForceEmbeddingProvider, EmbeddingProvider
from continuum.benchmarks.semantic_eval import evaluate, load_pairs, safest_threshold
from continuum.embeddings import OpenAICompatibleEmbeddingProvider

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "data" / "semantic_pairs.json"
DEFAULT_RUNTIME_THRESHOLD = 0.85


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--dim", type=int, default=64, help="n-gram embedder dimension")
    ap.add_argument("--embed-base-url", help="OpenAI-compatible server for embeddings")
    ap.add_argument("--embed-model", help="embedding model name on that server")
    ap.add_argument("--api-key")
    args = ap.parse_args()

    embedder: EmbeddingProvider
    if args.embed_base_url:
        if not args.embed_model:
            ap.error("--embed-model is required with --embed-base-url")
        embedder = OpenAICompatibleEmbeddingProvider(
            args.embed_base_url, args.embed_model, api_key=args.api_key
        )
    else:
        embedder = BruteForceEmbeddingProvider(args.dim)

    result = evaluate(load_pairs(args.dataset), embedder)
    result["dataset"] = str(Path(args.dataset).resolve().relative_to(ROOT))
    result["zero_false_hit_threshold"] = safest_threshold(result, 0.0)

    default_id = BruteForceEmbeddingProvider(64).identity()
    slug = "" if result["embedder"] == default_id else "_" + re.sub(r"[^a-z0-9]+", "-", result["embedder"].lower()).strip("-")
    out = ROOT / "benchmarks" / "data" / f"e9_semantic_false_hits{slug}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"E9: semantic false hits  embedder={result['embedder']}  pairs={result['pairs']}")
    for label, s in result["similarity"].items():
        print(f"  similarity {label:<10} min={s['min']:.3f} mean={s['mean']:.3f} max={s['max']:.3f}")
    print("  thr   prec   recall  false-hit  near-miss  unrelated  cache-wrong")
    for r in result["thresholds"]:
        prec = "  -  " if r["precision"] is None else f"{r['precision']:.3f}"
        cw = "  -" if r["cache_wrong_answer_rate"] is None else f"{r['cache_wrong_answer_rate']:.3f}"
        mark = "  <- runtime default" if r["threshold"] == DEFAULT_RUNTIME_THRESHOLD else ""
        print(
            f"  {r['threshold']:.2f}  {prec}  {r['recall']:.3f}   {r['false_hit_rate']:.3f}"
            f"      {r['near_miss_false_hit_rate']:.3f}      {r['unrelated_false_hit_rate']:.3f}"
            f"      {cw}{mark}"
        )
    print(f"  lowest threshold with zero false hits: {result['zero_false_hit_threshold']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
