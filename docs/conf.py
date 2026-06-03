import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Decider"
copyright = "2024-2026, Capitec"
author = "Sholto Armstrong"

import decider
version = decider.__version__
release = version

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_sitemap",
]

autosummary_generate = True
autoclass_content = "both"
html_show_sourcelink = False
autodoc_inherit_docstrings = True
add_module_names = False
autodoc_mock_imports = ["hamilton", "streamlit"]
nb_execution_mode = "off"   # don't execute notebooks during build

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_static_path = ["_static"]
html_theme = "furo"
html_title = "Decider"
html_theme_options = {
    "source_repository": "https://github.com/capitecbankltd/dsp_north-polrs",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-announcement-background": "#ffba00",
        "color-announcement-text": "#091E42",
    },
    "dark_css_variables": {
        "color-announcement-background": "#ffba00",
        "color-announcement-text": "#091E42",
    },
}

# anchor links inside {include}'d CONTRIBUTING.md don't resolve cross-doc
suppress_warnings = ["myst.xref_missing"]

language = "en"
html_baseurl = "https://capitecbankltd.github.io/dsp_north-polrs"
html_extra_path = ["robots.txt"]
