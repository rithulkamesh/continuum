"""E10: wall-clock prefix reuse against a real vLLM (or Ollama) server (issue #7).

Measures end-to-end latency of Continuum's vLLM backend when requests share a
long prefix (the server skips recomputing it) versus when they do not.

- **cold**: every prompt starts with a unique nonce, so no request shares a
  prefix with any other and the server prefills the whole prompt.
- **warm**: every prompt starts with the same long prefix; after one priming
  request the server serves the prefix from its KV cache.

Both arms send identical prompt lengths and ``max_tokens`` through a Continuum
``Session`` on the vLLM backend, so the difference is prefill work the server
skipped. The server's ``usage.prompt_tokens_details.cached_tokens`` is recorded
when it reports it (vLLM: ``--enable-prompt-tokens-details``).

Run against a server started with prefix caching (on by default in vLLM V1)::

    vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name bench \\
        --enable-prompt-tokens-details
    PYTHONPATH=python python benchmarks/scripts/e10_vllm_prefix_reuse.py \\
        --base-url http://localhost:8000 --model bench

``--self-test`` runs the same code against an in-process stub that
*simulates* prefill cost (2 ms per uncached character). It checks the
plumbing and produces no results; it never writes the data file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "data" / "e10_vllm_prefix_reuse.json"

QUESTIONS = [
    "Summarize the document in one sentence.",
    "List three risks the document mentions.",
    "What is the main recommendation?",
    "Who is the intended audience?",
    "Name one metric the document tracks.",
    "What should happen first?",
    "Which section is the longest?",
    "Give a title for the document.",
    "What is left undecided?",
    "Rewrite the conclusion in plain words.",
]


def shared_prefix(chars: int) -> str:
    para = (
        "Continuum design review. The runtime executes token and tensor programs, "
        "reuses prefix state across requests, checkpoints long agent runs, and forks "
        "them from any step. Operators track hit rate, tokens saved, and latency. "
    )
    text = "System: answer from the document below.\nDocument:\n"
    while len(text) < chars:
        text += para
    return text[:chars] + "\nQuestion: "


def _session(model: str) -> tuple[Any, list[dict[str, Any]]]:
    from continuum._native import BackendRegistry, Session
    from continuum.telemetry import CallbackObserver

    reg = BackendRegistry()
    reg.register_vllm()
    session = Session("e10", reg)
    events: list[dict[str, Any]] = []
    session.set_observer(CallbackObserver(events.append))
    return session, events


def _run(prompts: list[str], model: str, max_tokens: int, prime: str | None) -> dict[str, Any]:
    session, events = _session(model)
    if prime is not None:
        session.generate([prime], f"vllm/{model}", max_tokens, 0.0)
    events.clear()
    latencies = []
    for p in prompts:
        t0 = time.perf_counter()
        session.generate([p], f"vllm/{model}", max_tokens, 0.0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    nodes = [e for e in events if e["kind"] == "node_execution" and e["node_kind"] == "TokenOp"]
    latencies_sorted = sorted(latencies)
    return {
        "latency_ms": [round(x, 2) for x in latencies],
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))], 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "server_cached_tokens": sum(e["tokens_saved"] for e in nodes),
    }


def run_benchmark(model: str, prefix_chars: int, max_tokens: int, trials: int) -> dict[str, Any]:
    prefix = shared_prefix(prefix_chars)
    questions = (QUESTIONS * (trials // len(QUESTIONS) + 1))[:trials]
    # Same total prompt length in both arms: the nonce replaces the first bytes.
    cold = [f"[{uuid.uuid4().hex}] " + prefix[35:] + q for q in questions]
    warm = [prefix + q for q in questions]
    cold_res = _run(cold, model, max_tokens, prime=None)
    warm_res = _run(warm, model, max_tokens, prime=prefix + "Say OK.")
    return {
        "experiment": "E10: vLLM prefix reuse, wall clock",
        "model": model,
        "prefix_chars": prefix_chars,
        "max_tokens": max_tokens,
        "trials": trials,
        "cold": cold_res,
        "warm": warm_res,
        "p50_speedup": round(cold_res["p50_ms"] / warm_res["p50_ms"], 3) if warm_res["p50_ms"] else None,
    }


class _SimulatedServer:
    """Stand-in vLLM for --self-test: sleeps 2 ms per character the previous
    request did not share (a crude automatic-prefix-cache model)."""

    def __init__(self) -> None:
        self.previous = ""
        self.lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["prompt"]
                with outer.lock:
                    shared = len(os.path.commonprefix([prompt, outer.previous]))
                    outer.previous = prompt
                time.sleep(0.002 * (len(prompt) - shared) / 10)
                raw = json.dumps({
                    "choices": [{"index": 0, "text": " ok", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt) // 4,
                              "prompt_tokens_details": {"cached_tokens": shared // 4}},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "bench"))
    ap.add_argument("--prefix-chars", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--self-test", action="store_true", help="simulated server; plumbing check only")
    args = ap.parse_args(argv)

    stub = None
    if args.self_test:
        stub = _SimulatedServer()
        args.base_url = stub.url
        args.prefix_chars = min(args.prefix_chars, 2000)
        args.trials = min(args.trials, 5)
    if not args.base_url:
        ap.error("--base-url (or VLLM_BASE_URL) is required unless --self-test")
    previous = os.environ.get("VLLM_BASE_URL")
    os.environ["VLLM_BASE_URL"] = args.base_url.rstrip("/").removesuffix("/v1")
    os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")
    try:
        result = run_benchmark(args.model, args.prefix_chars, args.max_tokens, args.trials)
    finally:  # the backend reads the variable per call; do not leak it
        if previous is None:
            os.environ.pop("VLLM_BASE_URL", None)
        else:
            os.environ["VLLM_BASE_URL"] = previous
    result["simulated"] = bool(args.self_test)
    print(f"E10 vLLM prefix reuse  model={args.model} prefix={args.prefix_chars} chars  trials={args.trials}"
          + ("  [SIMULATED SERVER: plumbing check, not a result]" if args.self_test else ""))
    for arm in ("cold", "warm"):
        r = result[arm]
        print(f"  {arm:<5} p50={r['p50_ms']:>9.1f} ms  p95={r['p95_ms']:>9.1f} ms  "
              f"server cached tokens={r['server_cached_tokens']}")
    print(f"  p50 speedup (cold / warm): {result['p50_speedup']}")
    if stub is None:
        DATA.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {DATA.relative_to(ROOT)}")
    else:
        stub.httpd.shutdown()
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
