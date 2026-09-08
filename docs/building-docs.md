# Building the Docs

Hosted:

- Python API docs: <https://ct.rithul.dev/python/>
- C++ API docs: <https://ct.rithul.dev/cpp/>

## Local build

With [`just`](https://github.com/casey/just):

```bash
just docs        # both sets
just docs-py     # Python only  -> docs/api/python/_build
just docs-cpp    # C++ only     -> docs/api/cpp/html
just docs-serve  # build Python docs and serve at http://localhost:8000
```

`just docs-cpp` vendors `doxygen-awesome-css` into `docs/doxygen-awesome-css/`
on first run. `just docs-py` runs Sphinx with warnings treated as errors.

Without `just`:

```bash
# Python docs
uv sync --extra docs
PYTHONPATH=python uv run sphinx-build -b html docs/api/python docs/api/python/_build

# C++ docs
git clone --depth 1 --branch v2.3.4 \
  https://github.com/jothepro/doxygen-awesome-css.git docs/doxygen-awesome-css
doxygen docs/Doxyfile
```

Outputs:

- `docs/api/python/_build/index.html`
- `docs/api/cpp/html/index.html`

Both output directories are generated and git-ignored.

## What lives where

| File | Purpose |
|---|---|
| `docs/api/python/*.rst` | Python guide + API reference (Sphinx, Furo theme). |
| `docs/api/python/conf.py` | Sphinx config: theme, dark palette, logo, autodoc mocks. |
| `docs/api/python/_static/continuum.css` | Dark brand theme layered on Furo. |
| `docs/api/cpp-overview.md` | Doxygen main page. |
| `docs/Doxyfile` | Doxygen config. `HTML_COLORSTYLE = DARK`; pulls in `docs/design/*.md` as pages. |
| `docs/continuum-doxygen.css` | Dark brand theme + webfonts, layered on doxygen-awesome-css. |
| `web/logo.svg` | Brand mark, copied into both doc builds. |
