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
PYTHONPATH=python python scripts/benchmarks/run_examples.py | python scripts/benchmarks/validate_outputs.py
```

## Pull Request Guidelines

- Keep PRs focused and reasonably small.
- Include tests for behavioral changes.
- Update docs when API or behavior changes.
- Use clear commit messages that explain intent.
- Fill out the PR template completely.

## Coding Guidelines

- Prefer explicit behavior over hidden magic.
- Keep backend interoperability explicit and type-safe.
- Avoid introducing silent cross-backend conversions.
- Maintain consistency with existing style and architecture.

## Reporting Security Issues

Do not open public issues for vulnerabilities.
See `SECURITY.md` for the private reporting process.
