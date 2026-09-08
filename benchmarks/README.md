# Benchmarks

| Path | Contents |
|---|---|
| `scripts/` | One runner per experiment (`e1`-`e7`), `generate_plots.py`, and the example reproducibility harness (`run_examples.py`, `validate_outputs.py`). |
| `data/` | JSON output from each runner. |
| `plots/` | PNGs produced by `generate_plots.py`. |
| `reports/` | Written analysis. `isolated-mechanisms.md` and `mixed-workload.md` are current. `archive/` holds superseded drafts, kept for provenance. |

## Running

Live-backend experiments need Azure OpenAI credentials in the environment.
From the repo root:

```bash
PYTHONPATH=python python benchmarks/scripts/e1_shared_prefix.py
PYTHONPATH=python python benchmarks/scripts/e7_mixed_workload.py
PYTHONPATH=python python benchmarks/scripts/generate_plots.py
```

Deterministic FakeLLM versions of every mechanism run offline through
`examples/` and `tests/python/`, and are the ones checked in CI.

## Example reproducibility harness

```bash
PYTHONPATH=python python benchmarks/scripts/run_examples.py \
  | python benchmarks/scripts/validate_outputs.py
```

## Versioning

Results are dated inside each report rather than pinned to a directory name.
The suite that produced the published v1.1 numbers is tagged in git history.
