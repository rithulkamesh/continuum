# continuum-langchain

LangChain and LangGraph adapters for [Continuum](../../README.md). LangChain is
a dependency of this package only, never of `continuum-ai` itself.

```bash
pip install -e integrations/langchain           # plus `langgraph` for graphs
```

## `ContinuumCache`: LangChain LLM cache

```python
from langchain_core.globals import set_llm_cache
from continuum_langchain import ContinuumCache

set_llm_cache(ContinuumCache())                               # memo tier (exact repeats)
# set_llm_cache(ContinuumCache(semantic_threshold=0.92, embedder=my_embedder))
```

- Exact repeats (same prompt, same model configuration) hit Continuum's memo
  tier, which is LRU-bounded (`memo_entries`).
- The semantic tier is opt-in and needs a real embedder
  (`continuum.embeddings`). Measure its false-hit rate on your traffic first:
  `benchmarks/scripts/e9_semantic_false_hits.py`.
- **Tool calls are never cached**: neither responses that contain tool calls
  nor calls from models bound to tools.
- `cache.stats` counts memo hits, semantic hits, misses, and skipped tool calls.

## `ContinuumCheckpointSaver`: LangGraph checkpointer

```python
from continuum.checkpoints import LocalDirectoryStore     # or S3Store / GCSStore
from continuum_langchain import ContinuumCheckpointSaver, ContinuumLLM

llm = ContinuumLLM()                                        # a Continuum Session under the hood
saver = ContinuumCheckpointSaver(LocalDirectoryStore("ckpt"), session=llm.session)
graph = builder.compile(checkpointer=saver)
```

- **Durable, portable.** Checkpoints live in any Continuum `CheckpointStore`,
  so a thread paused on one machine resumes on another.
- **Warm-KV resume.** With a `session`, each checkpoint also stores that
  session's prefix-KV index. When a thread is loaded into a session whose
  cache is empty (a fresh process), the index is restored, so the resumed run
  gets prefix hits right away instead of starting cold.
- **Fork.** `graph.update_state(past_config, values)` creates a branch whose
  checkpoint records its parent (`parent_config`), and branches never
  overwrite each other.
- **Incremental.** Each channel value is stored once per version, so a
  checkpoint only writes the channels that changed. Objects are
  create-if-absent, so several workers can share one store.
- `get_tuple`, `list` (with `filter` / `before` / `limit`), `put`,
  `put_writes`, `delete_thread`, and their async variants.

## `ContinuumLLM`

A LangChain `LLM` that runs each prompt through a Continuum `Session`. It uses
the offline, deterministic FakeLLM backend by default (output is the
generated token ids), or the vLLM / Ollama shim when `VLLM_BASE_URL` is set.
Pass `session=` to use your own registry and tiers.

## Examples (run in CI against FakeLLM)

```bash
python integrations/langchain/examples/cache_example.py
python integrations/langchain/examples/langgraph_checkpoint_example.py
```

## Tests

```bash
cd integrations/langchain && pytest
```
