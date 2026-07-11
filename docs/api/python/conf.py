import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

project = "Continuum"
author = "Continuum Authors"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "furo"
html_title = "Continuum Python API"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#5b4bdb",
        "color-brand-content": "#5b4bdb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7c6cff",
        "color-brand-content": "#7c6cff",
    },
}
autodoc_mock_imports = ["torch", "_continuum", "continuum._native"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
