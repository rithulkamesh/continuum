# Roadmap

This is the planning counterpart to [`CHANGELOG.md`](CHANGELOG.md): the changelog
records what shipped, this file records what is planned. Work is grouped into
release trains, each tracked by a GitHub milestone. Individual tasks live as
issues labelled `area:*` and `P0`/`P1`/`P2`.

Milestones: [v1.1](https://github.com/rithulkamesh/continuum/milestone/1) ·
[v1.2](https://github.com/rithulkamesh/continuum/milestone/2) ·
[v1.3](https://github.com/rithulkamesh/continuum/milestone/3) ·
[v2.0](https://github.com/rithulkamesh/continuum/milestone/4) ·
[Backlog](https://github.com/rithulkamesh/continuum/milestone/5)

## v1.1 — Build & test hardening

The v1.0 readiness report records the C++ suite passing locally (`28/28`), but
`.github/workflows/ci.yml` has only a `python` job and a `docs` job. The C++
engine is the product and needs the same CI rigour the Python layer already has:
build matrix, coverage gate, and a real serialization conformance corpus.

| Issue | Title | Priority |
|---|---|---|
| [#1](https://github.com/rithulkamesh/continuum/issues/1) | CI: add a C++ build + `ctest` job | P0 |
| [#2](https://github.com/rithulkamesh/continuum/issues/2) | CI: enforce a C++ coverage gate | P1 |
| [#3](https://github.com/rithulkamesh/continuum/issues/3) | Expand the C++ unit suite beyond graph + smoke | P1 |
| [#4](https://github.com/rithulkamesh/continuum/issues/4) | CIR schema conformance corpus + cross-version test | P1 |
| [#8](https://github.com/rithulkamesh/continuum/issues/8) | Harden the MLX backend and cover it on macOS arm64 | P2 |

## v1.2 — Reuse-stack maturity

The five-tier reuse stack works in the benchmarks, but several tiers have thin
public semantics: the semantic cache's embedding source is undocumented, no tier
documents eviction despite taking a capacity argument, the memory-graph recall
tier is absent from `docs/design/cache.md`, and cache keys have no tenant
dimension. This train makes every tier configurable, bounded, measured, and safe
for multi-tenant use.

| Issue | Title | Priority |
|---|---|---|
| [#9](https://github.com/rithulkamesh/continuum/issues/9) | Semantic cache: pluggable embedding provider + config | P1 |
| [#10](https://github.com/rithulkamesh/continuum/issues/10) | Semantic cache: false-hit-rate eval harness | P1 |
| [#11](https://github.com/rithulkamesh/continuum/issues/11) | Cache eviction and memory bounds across all tiers | P1 |
| [#12](https://github.com/rithulkamesh/continuum/issues/12) | Document and benchmark the memory-graph recall tier | P2 |
| [#13](https://github.com/rithulkamesh/continuum/issues/13) | Multi-tenant cache isolation / key namespacing | P0 |
| [#14](https://github.com/rithulkamesh/continuum/issues/14) | On-disk cache + checkpoint format versioning and migration | P1 |

## v1.3 — Durability & scale

Checkpoints are byte blobs today. This train adds a pluggable store (local
directory, S3, GCS), incremental checkpoints so `run_until_step` on long runs
does not re-serialize everything, and a shared checkpoint store that multiple
workers can resume and fork from.

| Issue | Title | Priority |
|---|---|---|
| [#15](https://github.com/rithulkamesh/continuum/issues/15) | Checkpoint to a pluggable object store | P1 |
| [#16](https://github.com/rithulkamesh/continuum/issues/16) | Incremental and distributed checkpoint resume | P2 |

## v2.0 — Extensibility & ecosystem

`backend_abi.h` defines a C vtable "suitable for dynamic loading later"
(`docs/design/abi.md`) — v2 actually loads plugins through it and ships a
conformance kit so third parties can build backends out of tree. This train also
delivers the stable typed Python API, OpenTelemetry export, and the first-party
LangChain / OpenAI-proxy integrations.

| Issue | Title | Priority |
|---|---|---|
| [#5](https://github.com/rithulkamesh/continuum/issues/5) | Dynamic backend plugin loading via the C vtable | P1 |
| [#6](https://github.com/rithulkamesh/continuum/issues/6) | Third-party backend conformance test kit | P2 |
| [#7](https://github.com/rithulkamesh/continuum/issues/7) | Promote `vllm_shim` to a real vLLM backend with KV export | P1 |
| [#17](https://github.com/rithulkamesh/continuum/issues/17) | Stable typed public API for durable agents | P1 |
| [#18](https://github.com/rithulkamesh/continuum/issues/18) | OpenTelemetry export for per-tier reuse events | P2 |
| [#19](https://github.com/rithulkamesh/continuum/issues/19) | `continuum-langchain`: LangGraph checkpointer + LangChain cache | P1 |
| [#20](https://github.com/rithulkamesh/continuum/issues/20) | OpenAI-compatible proxy shim | P2 |

## Open questions

- Milestone naming: keep `v1.1 … v2.0`, or move to date-based trains.
- Is [#13](https://github.com/rithulkamesh/continuum/issues/13) genuinely P0, or
  is Continuum single-tenant-per-process by design for now?
- [#19](https://github.com/rithulkamesh/continuum/issues/19) and
  [#20](https://github.com/rithulkamesh/continuum/issues/20): same repo under an
  `integrations/` path, or separate repos.
