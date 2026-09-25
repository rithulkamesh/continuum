# Integrations

Ways to put Continuum's reuse stack behind code you already have.

## OpenAI-compatible proxy

`continuum.proxy` is a local HTTP server that speaks the OpenAI wire format.
Any client that lets you set a base URL gains Continuum's caching with no code
changes.

```bash
python -m continuum.proxy --upstream https://api.openai.com/v1 --port 8787
export OPENAI_BASE_URL=http://localhost:8787/v1
```

That covers the OpenAI SDK, LangChain's `ChatOpenAI`, LlamaIndex's `OpenAI`,
and plain HTTP. The upstream can be any OpenAI-compatible server: OpenAI,
vLLM, Ollama (`--upstream http://localhost:11434/v1`), LM Studio, and so on.
The client's `Authorization` header is forwarded. If the client sends none,
`--api-key` (or `CONTINUUM_PROXY_API_KEY` / `OPENAI_API_KEY`) is used.

**Reused.** Each `POST /v1/chat/completions` or `/v1/completions` becomes one
Continuum `TokenOp`, keyed on its messages (or prompt) and sampling
parameters, with the upstream call as the backend:

- **Exact repeats** hit the memo tier and return the stored response without
  an upstream call.
- **Shared prefixes** (for example a common system prompt) hit the prefix-KV
  tier. The upstream still gets the full request, since it has no view of
  Continuum's state. The shared prefix is reported as savings, preferring the
  upstream's own `usage.prompt_tokens_details.cached_tokens`.
- **Paraphrases** can hit the semantic tier, but only with
  `--semantic-threshold` and `--embed-model`. It is off by default. The
  measured starting point is `--embed-model wordllama --semantic-threshold 0.7`
  (a local model; `pip install "continuum-ai[semantic]"`). Every candidate
  hit also passes the near-miss verifier; add `--judge-model <chat model>` to
  have an LLM judge instead. See
  [the false-hit report](../benchmarks/reports/semantic-false-hits.md).

Transport-only fields (`stream`, `stream_options`, `user`, `metadata`,
`store`, `service_tier`) are not part of the key, so a streamed and a
non-streamed request for the same content share one entry.

**Never cached.** Requests with `tools` / `functions` / `tool_choice`,
conversations containing tool calls or tool results, `n > 1`, and requests
sent with `Cache-Control: no-cache` or `no-store` are forwarded untouched, as
are upstream errors. Every other path (`/v1/models`, `/v1/embeddings`, ...) is
passed straight through.

**Streaming.** On a miss, the upstream's server-sent events stream through as
they arrive and the assembled response is cached. On a hit, the stored
response is replayed as a server-sent-event stream.

**Metrics.**

- Response headers: `x-continuum-cache` (`hit`, `miss`, `bypass`),
  `x-continuum-served-by` (`memo`, `semantic`, `backend`), and
  `x-continuum-tokens-saved`.
- `GET /metrics`: Prometheus text (requests by outcome, lookups and hits by
  tier, tokens saved).
- `GET /continuum/metrics`: the same counters as JSON.
- `GET /health`

All server threads share the memo, prefix-KV, and semantic tiers.

From Python:

```python
from continuum.proxy import ContinuumProxy, ProxyConfig

proxy = ContinuumProxy(ProxyConfig(upstream="http://localhost:11434/v1"), port=8787).start()
...
proxy.close()
```

## LangChain and LangGraph

[`integrations/langchain`](../integrations/langchain/) is a separate package,
`continuum-langchain`, so LangChain never becomes a dependency of the engine.
It contains:

- `ContinuumCache`: a LangChain `BaseCache` on the memo tier, plus the
  semantic tier if you opt in. Tool calls are never cached.
- `ContinuumCheckpointSaver`: a LangGraph `BaseCheckpointSaver` that writes to
  any Continuum `CheckpointStore` (local, S3, GCS). It snapshots a session's
  prefix-KV index with each checkpoint so a resumed thread starts warm, and it
  keeps fork lineage from `update_state`.
- `ContinuumLLM`: a LangChain LLM on a Continuum `Session`.

Both adapters have a runnable example, and CI runs them against the FakeLLM
backend. See the package README.

## Python backends

`BackendRegistry.register_python(name, fn)` puts any Python callable behind
the reuse stack as a token backend. `fn` gets a dict (`prompt_parts`,
`model_id`, `max_tokens`, `temperature`, `has_prefix_state`,
`remaining_tokens`, `prompt_len`) and returns the output text, or a dict with
`output` and optional `tokens_sent` / `tokens_saved`. The proxy and the
LangChain adapter are both built on it.

```python
from continuum._native import BackendRegistry, MemoTable, Session

reg = BackendRegistry()
reg.register_python("mine", lambda req: my_model("\n".join(req["prompt_parts"])), priority=100)
session = Session("app", reg)
session.set_memo_table(MemoTable())
session.generate(["You are terse.", "What is a KV cache?"], "mine/model", 128)
```
