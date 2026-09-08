# Building the Docs

Hosted:

- Python API docs: <https://ct.rithul.dev/python/>
- C++ API docs: <https://ct.rithul.dev/cpp/>

## Local build

```bash
# Python docs
python -m venv .venv-docs
. .venv-docs/bin/activate
pip install sphinx furo breathe
PYTHONPATH=python sphinx-build -b html docs/api/python docs/api/python/_build

# C++ docs
doxygen docs/Doxyfile
```

Outputs:

- `docs/api/python/_build/index.html`
- `docs/api/cpp/html/index.html`

Both output directories are generated and git-ignored.
