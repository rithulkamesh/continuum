"""Backend conformance check: ``python -m continuum.backend_check <target>``.

``target`` is a built-in backend name (``fake``, ``libtorch``, ``vllm``, ...),
a path to a plugin shared library, or a plugin name listed in
``CONTINUUM_BACKEND_PLUGINS``. Prints one line per check and exits non-zero if
any check fails. The checks are the bar every backend must clear; see
"Backend conformance" in ``docs/design/abi.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from continuum._native import check_backend


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m continuum.backend_check", description=__doc__)
    ap.add_argument("target", help="built-in name, plugin path, or CONTINUUM_BACKEND_PLUGINS name")
    ap.add_argument(
        "--nondeterministic",
        action="store_true",
        help="skip the determinism checks (for sampling or remote backends)",
    )
    ap.add_argument("--prompt", default="", help="override the token-path prompt")
    ap.add_argument("--tensor-op", default="identity", help="tensor op that must echo its input")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    report = check_backend(
        args.target,
        expect_deterministic=not args.nondeterministic,
        prompt=args.prompt,
        tensor_op=args.tensor_op,
    )
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2))
    else:
        print(report["summary"], end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
