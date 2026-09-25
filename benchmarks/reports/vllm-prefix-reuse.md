# E10 — vLLM Prefix Reuse, Wall Clock

**Status**: harness ready, **results pending**. This needs a GPU vLLM server
(or an Ollama / llama.cpp server), which the environment that built the
harness did not have. No numbers are reported until a real run exists.

**Runner**: [`benchmarks/scripts/e10_vllm_prefix_reuse.py`](../scripts/e10_vllm_prefix_reuse.py)
**Raw data (after a run)**: `benchmarks/data/e10_vllm_prefix_reuse.json`

## What is measured

End-to-end latency of `Session.generate` on Continuum's vLLM backend in two
arms with identical prompt lengths and `max_tokens`:

- **cold**: each prompt starts with a unique nonce, so no request shares a
  prefix and the server prefills everything;
- **warm**: every prompt starts with the same ~8,000-character document; after
  one priming request the server serves that prefix from its KV cache.

The difference is prefill work the server skipped. The server's
`usage.prompt_tokens_details.cached_tokens` is recorded as corroboration.

## How prefix reuse works with vLLM

The KV blocks stay inside the vLLM server. Continuum never copies them.

1. The backend always sends the **full** prompt. vLLM's automatic prefix
   caching hashes the prompt's token blocks and skips prefill for blocks it
   already holds.
2. Continuum's prefix-KV tier records which prefix each request left warm on
   the server (the backend's state handle). Hits are reported as
   `tokens_saved`, using the server's `cached_tokens` when present.
3. **Checkpoint / resume.** The state handle is portable
   (`export_state` / `import_state`), so a checkpoint carries it and a resumed
   process knows which prefixes are warm. If the server may have restarted,
   set `VLLM_REWARM_ON_IMPORT=1`: each imported prefix is re-warmed with a
   1-token request before the resumed run needs it.

## Run it

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name bench --enable-prompt-tokens-details
PYTHONPATH=python python benchmarks/scripts/e10_vllm_prefix_reuse.py \
    --base-url http://localhost:8000 --model bench --trials 20
```

Ollama works too (`--base-url http://localhost:11434 --model <name>`),
although it does not report `cached_tokens`.

`--self-test` runs against a simulated server and only checks the plumbing.
Its latencies are simulated and must not be reported.

## Results

_Pending a run on real hardware. Record GPU, vLLM version, model, prefix
size, and the p50 / p95 table from the script output here._
