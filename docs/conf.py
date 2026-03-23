# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "FURAX-CS"
copyright = "2024, FURAX Team"
author = "FURAX Team"
release = "0.1.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
myst_heading_anchors = 3

# nbsphinx configuration
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

suppress_warnings = ["toc.no_title"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/09_*",
    "notebooks/80-*",
    "notebooks/build_*",
    "notebooks/README*",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "FURAX-CS"
html_static_path = ["_static"]
