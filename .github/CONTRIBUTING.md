# Contributing to Continuum

Thanks for contributing to Continuum. We welcome bug fixes, docs updates, tests, and new features.

## Before You Start

- Read the `README.md` for project context and docs.
- Search existing issues and pull requests before opening a new one.
- For larger changes, open an issue first to align on scope.

## Development Setup

```bash
git clone https://github.com/rithulkamesh/continuum.git
cd continuum
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev,test,docs]
pip install pre-commit
pre-commit install
```

## Local Validation

Run the main checks before opening a pull request:

```bash
pre-commit run --all-files
pytest
```

Optional reproducibility check:

```bash
PYTHONPATH=python python benchmarks/scripts/run_examples.py | python benchmarks/scripts/validate_outputs.py
```

## Pull Request Guidelines

- Keep PRs focused and reasonably small.
- Include tests for behavioral changes.
- Update docs when API or behavior changes.
- Use clear commit messages that explain intent.
- Fill out the PR template completely.

## Commit Hygiene

- Do not add `Co-Authored-By` trailers for AI assistants, or any
  `Claude-Session`, `Generated-with`, or similar tool-attribution lines, to
  commit messages or pull request descriptions.
- Do not commit assistant or editor state: `.claude/settings.local.json`,
  `CLAUDE.local.md`, `.cursor/`, `.aider*`, scratch planning files, or session
  transcripts. The tracked `.claude/skills/` and `CLAUDE.md` are the only
  intentional exceptions.
- Keep generated output out of git (docs builds, `build/`, `dist/`).

## Coding Guidelines

- Prefer explicit behavior over hidden magic.
- Keep backend interoperability explicit and type-safe.
- Avoid introducing silent cross-backend conversions.
- Maintain consistency with existing style and architecture.
- C++: follow [`.clang-format`](../.clang-format) and
  [`.clang-tidy`](../.clang-tidy), and the checklist in
  [`.claude/skills/cpp-standards/SKILL.md`](../.claude/skills/cpp-standards/SKILL.md).
- Python: `ruff` and `mypy` must pass; both run in pre-commit and CI.

## Reporting Security Issues

Do not open public issues for vulnerabilities.
See `SECURITY.md` for the private reporting process.
