"""06 - An agent fleet that sends its system prompt once.

Hundreds of chat sessions, every one prefixed with the same multi-thousand-token
policy / persona / tool-schema preamble. Naively that preamble is tokenized and
billed once per session. With a shared trie prefix KV cache, the first session
pays for it and the rest ride the cache -- only their unique user turn is new
work.

This runs 64 "sessions" through one persistent ``Session`` cache (FakeLLM,
deterministic), each with a 3,000-char shared preamble and a distinct suffix,
then projects the input-token bill against a sample price.

    PYTHONPATH=python python examples/06_agent_fleet_prefix.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CONTINUUM_LOG_LEVEL", "warn")

from continuum._native import run_session_benchmark  # noqa: E402

BAR = "=" * 64

SESSIONS = 64
PREFIX_CHARS = 3000
# Illustrative only: USD per 1M input tokens, ~4 chars/token.
PRICE_PER_MTOK = 0.15
CHARS_PER_TOK = 4.0


def main() -> None:
    print(BAR)
    print(" Continuum - Agent Fleet (shared system prompt)")
    print(BAR)
    print(f"sessions={SESSIONS}  shared preamble={PREFIX_CHARS} chars  "
          f"(~{PREFIX_CHARS / CHARS_PER_TOK:.0f} tokens)")
    print("-" * 64)

    res = run_session_benchmark(num_steps=SESSIONS, prefix_tokens=PREFIX_CHARS, suffix_tokens=16)
    runs = res["runs"]

    saved_chars = sum(r["total_tokens_saved"] for r in runs)
    sent_chars = sum(r["total_tokens_processed"] for r in runs)
    naive_chars = saved_chars + sent_chars

    first, last = runs[0], runs[-1]
    print(f"  session   1 : token_reduction={first['token_reduction'] * 100:5.1f}%")
    print(f"  session {SESSIONS:>3} : token_reduction={last['token_reduction'] * 100:5.1f}%")
    print(f"  shared-prefix cache entries: {res['final_cache_size']}")
    print("-" * 64)

    naive_tok = naive_chars / CHARS_PER_TOK
    sent_tok = sent_chars / CHARS_PER_TOK
    naive_cost = naive_tok / 1e6 * PRICE_PER_MTOK
    real_cost = sent_tok / 1e6 * PRICE_PER_MTOK
    print(f"input tokens  without reuse : {naive_tok:>12,.0f}")
    print(f"input tokens  with Continuum: {sent_tok:>12,.0f}")
    if naive_chars:
        print(f"reduction                   : {saved_chars / naive_chars * 100:>11.1f}%")
    print(f"illustrative cost @ ${PRICE_PER_MTOK}/Mtok : "
          f"${naive_cost:.4f} -> ${real_cost:.4f}")

    assert last["token_reduction"] > first["token_reduction"], "fleet should converge on prefix reuse"
    print(BAR)
    print(" agent fleet prefix: OK")
    print(BAR)


if __name__ == "__main__":
    main()
