"""End-to-end checks against a real Ollama server.

Exercises the paths that talk to an OpenAI-compatible model server:

1. DurableAgent on Ollama (VLLM_BASE_URL): generated text per step, a
   checkpoint resume that replays completed steps without regenerating them,
   and a fork whose edit cascades to later steps.
2. The OpenAI-compatible proxy with Ollama upstream, through the official
   ``openai`` SDK: a repeat is served from cache with identical content,
   streaming works both ways, and the metrics endpoint counts it.
3. The semantic tier with real Ollama embeddings: a reworded question is
   served, a near-miss edit is refused by the hit verifier.
4. Prefix reuse through the vLLM backend: cold vs warm latency (reported,
   not asserted: tiny CPU models make timing noisy).

    ollama pull smollm2:135m && ollama pull all-minilm
    PYTHONPATH=python python scripts/ollama_e2e.py --model smollm2:135m --embed-model all-minilm

Thinking models need ``--max-tokens 512`` or the durable steps come back empty.

Exits non-zero if any check fails; ``--json`` writes a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import Any

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

RESULTS: list[dict[str, Any]] = []


def check(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., None]]:
    def wrap(fn: Callable[..., dict[str, Any]]) -> Callable[..., None]:
        def run(*args: Any) -> None:
            t0 = time.perf_counter()
            try:
                detail = fn(*args)
                RESULTS.append(
                    {
                        "check": name,
                        "ok": True,
                        "seconds": round(time.perf_counter() - t0, 2),
                        **detail,
                    }
                )
                print(f"PASS  {name}  {json.dumps(detail)[:300]}", flush=True)
            except Exception as exc:  # report every check, then fail at the end
                RESULTS.append(
                    {"check": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAIL  {name}  {type(exc).__name__}: {exc}", flush=True)

        return run

    return wrap


def wait_for_server(base: str, timeout: float = 60.0) -> list[str]:
    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=5) as r:
                return [m["name"] for m in json.load(r).get("models", [])]
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(1)


class _Recorder:
    """Pass-through between the agent and Ollama that records request bodies,
    so checks assert what reached the server rather than what a small model
    happened to answer."""

    def __init__(self, upstream: str) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(json.loads(raw))
                req = urllib.request.Request(
                    upstream + self.path,
                    data=raw,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"


@check("durable agent on ollama: text, replay, fork cascade")
def durable_agent(base: str, model: str, max_tokens: int) -> dict[str, Any]:
    rec_server = _Recorder(base)
    os.environ["VLLM_BASE_URL"] = rec_server.url
    try:
        from continuum import DurableAgent

        edit = "Name a vegetable of that color."
        prompts = [
            "Name a primary color.",
            "Name a fruit of that color.",
            "Name a dish with that fruit.",
        ]
        rec = DurableAgent()
        assert rec.backend == "vllm", rec.backend
        rec.begin(prompts, model_id=f"vllm/{model}", max_tokens=max_tokens)
        ckpt = rec.run_until_step(0)
        assert len(rec_server.requests) == 1

        forked = DurableAgent.fork(ckpt, rec.prompt_node_ids[1], edit)
        b = DurableAgent()
        b.resume_from(forked)
        sent = [r["prompt"] for r in rec_server.requests[1:]]
        outs = b.step_outputs()
        assert all(isinstance(x, str) and x.strip() for x in outs), outs
        # Step 1 replays from the checkpoint: only steps 2 and 3 reach the server.
        assert len(sent) == 2, sent
        assert sent[0].startswith(edit), "the edit must reach the server"
        assert " ".join(outs[0].split()) in sent[0], "step 2 must see step 1's output"
        assert " ".join(outs[1].split()) in sent[1], "step 3 must see the forked step 2's output"
        return {"model": model, "steps": [x[:60] for x in outs], "requests_after_resume": len(sent)}
    finally:
        os.environ.pop("VLLM_BASE_URL", None)
        rec_server.httpd.shutdown()


@check("openai sdk -> continuum proxy -> ollama: cache hit, streaming, metrics")
def proxy(base: str, model: str) -> dict[str, Any]:
    import openai

    from continuum.proxy import ContinuumProxy, ProxyConfig

    p = ContinuumProxy(ProxyConfig(upstream=base + "/v1"), port=0).start()
    try:
        client = openai.OpenAI(base_url=p.address + "/v1", api_key="ollama", max_retries=0)
        msgs = [{"role": "user", "content": "In one short sentence, what is a cache?"}]
        first = client.chat.completions.with_raw_response.create(
            model=model, messages=msgs, temperature=0
        )
        t0 = time.perf_counter()
        second = client.chat.completions.with_raw_response.create(
            model=model, messages=msgs, temperature=0
        )
        hit_ms = (time.perf_counter() - t0) * 1000
        assert first.headers["x-continuum-cache"] == "miss"
        assert second.headers["x-continuum-cache"] == "hit"
        text = first.parse().choices[0].message.content
        assert text and second.parse().choices[0].message.content == text

        smsgs = [{"role": "user", "content": "Count from one to five."}]
        streamed = "".join(
            c.choices[0].delta.content or ""
            for c in client.chat.completions.create(
                model=model, messages=smsgs, temperature=0, stream=True
            )
            if c.choices
        )
        replay = "".join(
            c.choices[0].delta.content or ""
            for c in client.chat.completions.create(
                model=model, messages=smsgs, temperature=0, stream=True
            )
            if c.choices
        )
        assert streamed and replay == streamed
        with urllib.request.urlopen(p.address + "/continuum/metrics") as r:
            metrics = json.load(r)
        assert metrics["requests"]["hit"] >= 2, metrics
        return {
            "answer": text[:80],
            "hit_latency_ms": round(hit_ms, 1),
            "requests": metrics["requests"],
        }
    finally:
        p.close()


@check("semantic tier with ollama embeddings + hit verifier")
def semantic(base: str, model: str, embed_model: str) -> dict[str, Any]:
    from continuum._native import SemanticCacheIndex
    from continuum.embeddings import OpenAICompatibleEmbeddingProvider

    emb = OpenAICompatibleEmbeddingProvider(base, embed_model)
    cached = "how do I reset my password"
    rewording = "I forgot my password and need to reset it"
    near_miss = "how do I reset my username"
    vecs = {q: emb.embed(q) for q in (cached, rewording, near_miss)}
    sims = {
        q: SemanticCacheIndex.cosine_similarity(vecs[cached], vecs[q])
        for q in (rewording, near_miss)
    }
    # Threshold just under both similarities: both clear it, so only the
    # hit verifier decides what is served.
    threshold = min(sims.values()) - 0.01
    idx = SemanticCacheIndex(8, threshold)
    idx.insert(vecs[cached], "m", b"answer", embedder_id=emb.identity(), prompt=cached)
    served_rewording = idx.lookup(
        vecs[rewording], "m", embedder_id=emb.identity(), query_prompt=rewording
    )
    served_near_miss = idx.lookup(
        vecs[near_miss], "m", embedder_id=emb.identity(), query_prompt=near_miss
    )
    assert served_rewording["above_threshold"], "the rewording should be served"
    assert not served_near_miss["above_threshold"], "the near-miss must be refused"
    assert served_near_miss["verifier_rejections"] == 1
    return {
        "embedder": emb.identity(),
        "dims": emb.dimension(),
        "similarity_rewording": round(sims[rewording], 3),
        "similarity_near_miss": round(sims[near_miss], 3),
        "threshold": round(threshold, 3),
    }


@check("prefix reuse via vllm backend: cold vs warm latency")
def prefix(base: str, model: str) -> dict[str, Any]:
    os.environ["VLLM_BASE_URL"] = base
    try:
        from continuum._native import BackendRegistry, Session

        doc = "Continuum reuses prefix state across requests and checkpoints agent runs. " * 40
        reg = BackendRegistry()
        reg.register_vllm()
        s = Session("ollama-prefix", reg)

        def timed(prompt: str) -> float:
            t0 = time.perf_counter()
            s.generate([prompt], f"vllm/{model}", 4, 0.0)
            return (time.perf_counter() - t0) * 1000

        timed("Prime. " + doc)  # load the model
        cold = [timed(f"[{i}{time.time_ns()}] " + doc + " Question: summarize.") for i in range(3)]
        timed(doc + " Question: warm up.")
        warm = [timed(doc + f" Question: summarize point {i}.") for i in range(3)]
        return {
            "cold_p50_ms": round(statistics.median(cold), 1),
            "warm_p50_ms": round(statistics.median(warm), 1),
            "speedup": round(statistics.median(cold) / statistics.median(warm), 2),
        }
    finally:
        os.environ.pop("VLLM_BASE_URL", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ap.add_argument("--model", default="smollm2:135m")
    ap.add_argument("--embed-model", default="all-minilm")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="per-step budget for the durable agent; thinking models (e.g. gemma4) "
        "spend it on hidden reasoning, so give them ~512",
    )
    ap.add_argument("--json", help="write a JSON summary here")
    args = ap.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")
    models = wait_for_server(base)
    print(f"ollama at {base}; models: {models}", flush=True)

    durable_agent(base, args.model, args.max_tokens)
    proxy(base, args.model)
    semantic(base, args.model, args.embed_model)
    prefix(base, args.model)

    ok = all(r["ok"] for r in RESULTS)
    print(f"\n{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks passed", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "base_url": base,
                    "model": args.model,
                    "embed_model": args.embed_model,
                    "results": RESULTS,
                },
                f,
                indent=2,
            )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
