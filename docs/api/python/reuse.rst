The reuse stack
===============

Most of an agent run is work it has already done: the same system prompt
re-sent every step, the same sub-question asked twice, a suite of near-identical
eval prompts. Continuum removes that redundancy at the layer where the tokens
are actually spent.

Every ``TokenOp`` checks the tiers below in order and stops at the first hit.
Only a cold call reaches the backend.

The five tiers
--------------

===============  ======================================================  ==========================
Tier             When it fires                                            Cost of a hit
===============  ======================================================  ==========================
Memo             The exact same call, seen before.                        O(1) hash lookup, 0 tokens
Semantic         Different wording, same intent, matched by embedding.    1 vector lookup, 0 tokens
Prefix KV        A shared prompt prefix is already tokenized.             Send only the new suffix
Layer KV         Warm attention state can be carried forward.             No prefill
Memory graph     A prior run holds relevant context to surface.           1 graph query
===============  ======================================================  ==========================

If none match, the backend runs and its result populates the tiers for next
time.

How a key is built
------------------

A cache key is derived from:

- backend and model identity,
- decode parameters (``op_name``, ``temperature``, ``max_tokens``),
- the canonicalized input token sequence.

The interpreter canonicalizes textual and token inputs *before* lookup, and
prefix length is measured against that canonical form, so a match survives
formatting differences.

Token reuse and state reuse
---------------------------

A prefix hit has two halves that must agree:

- **Token reuse.** The runtime finds the longest matching token prefix and
  sends only the suffix.
- **State reuse.** The runtime hands the backend the ``BackendState`` handle
  from the matched entry.

Reuse is valid only when both align. If a state handle was derived from a
different canonical prefix length than the one being claimed, it is not used.
On the Azure path, prefix savings are currently approximated by sending
suffix-only requests and reporting ``tokens_sent`` / ``tokens_saved``. On the
vLLM path the same prefix-hit mechanism forwards real KV state, so the shared
prefix is not recomputed. Both paths emit the same metrics, so benchmark
numbers stay backend-agnostic.

Keeping reuse correct
---------------------

- **Side effects are never cached.** ``ToolOp`` results always run.
- **Policy-gated.** Every tier respects a per-session
  :class:`~continuum._native.ReusePolicy`: ``always``, ``never``, or a
  minimum-prefix-length threshold. One switch, no stale reads.
- **Resume bumps versions.** Memoized results are version-bumped when a run
  resumes, so a stale entry cannot leak across the process boundary.

Setting a policy
----------------

.. code-block:: python

   from continuum._native import ReusePolicy, Session, BackendRegistry

   session = Session("agent-fleet", BackendRegistry())
   session.policy = ReusePolicy.always()             # reuse whenever a tier matches
   session.policy = ReusePolicy.never()              # bypass every tier
   session.policy = ReusePolicy.threshold(min_len=64)  # only reuse prefixes >= 64 tokens

Cross-session persistence
-------------------------

``Session.save_cache_metadata(path)`` writes the cache index to disk;
``load_cache_metadata(path)`` reloads it in a new process. On the first warm
run after a restart the hit rate is at least 80 percent in the bundled
benchmark.

Measured, per tier
------------------

Isolated benchmarks against a live Azure OpenAI backend:

- Trie prefix KV cache: about 99 percent fewer tokens on a 3,000-character
  shared prefix, roughly 30 tokens sent per call.
- Memo table: 5 of 5 exact-repeat calls served from cache, 0 ms each.
- Mixed 20-step workflow: 92.5 percent fewer tokens, 4 of 20 backend calls
  eliminated.
- Prefix-hit latency: median 5.4 s to 3.7 s. The network round-trip stays, so
  wall-clock time drops less than token cost.

Full tables and the scripts behind every number are in
`docs/benchmarks.md <https://github.com/rithulkamesh/continuum/blob/master/docs/benchmarks.md>`_.
