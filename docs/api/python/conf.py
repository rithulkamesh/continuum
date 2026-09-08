import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "python"))

# Keep the brand mark in one place (web/logo.svg) and copy it in at build time.
_static = HERE / "_static"
_static.mkdir(exist_ok=True)
shutil.copyfile(ROOT / "web" / "logo.svg", _static / "logo.svg")

project = "Continuum"
author = "Rithul Kamesh"
copyright = "Rithul Kamesh, MIT licensed"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# --- HTML / theme ---------------------------------------------------------
html_theme = "furo"
html_title = "Continuum"
html_static_path = ["_static"]
html_css_files = ["continuum.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

pygments_style = "monokai"
pygments_dark_style = "monokai"

_continuum_vars = {
    "color-brand-primary": "#e8623c",
    "color-brand-content": "#e8623c",
    "color-background-primary": "#100f0d",
    "color-background-secondary": "#17150f",
    "color-foreground-primary": "#ece9e0",
    "color-foreground-secondary": "#b6b2a6",
    "color-foreground-muted": "#8a8579",
    "font-stack": '"Hanken Grotesk", ui-sans-serif, system-ui, sans-serif',
    "font-stack--monospace": '"IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace',
}
html_theme_options = {
    "light_css_variables": _continuum_vars,
    "dark_css_variables": _continuum_vars,
    "source_repository": "https://github.com/rithulkamesh/continuum/",
    "source_branch": "master",
    "source_directory": "docs/api/python/",
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/rithulkamesh/continuum",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# --- autodoc -----------------------------------------------------------
# The hosted docs build has no compiled extension and no torch, so these
# stay mocked. The native runtime API is documented by hand in reference.rst.
autodoc_mock_imports = ["torch", "_continuum", "continuum._native"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
add_module_names = False
python_use_unqualified_type_names = True
