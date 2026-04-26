# API Docs

Continuum ships dual API docs:

- Python API docs via Sphinx (`docs/api/python`)
- C++ API docs via Doxygen (`Doxyfile`)

## Build Python docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/api/python docs/api/python/_build
```

## Build C++ docs

```bash
doxygen Doxyfile
```
