# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpockFlow'
copyright = '2024, Sholto Armstrong'
author = 'Sholto Armstrong'
import spockflow
version = str(spockflow.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    'sphinx.ext.napoleon', 
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinxcontrib.confluencebuilder',
    "sphinx_sitemap", # Welcome robots to the website
]

# AutoDoc Conf (Thanks https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/blob/master/docs/conf.py)
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
#autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False # Remove namespaces from class/method signatures


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']

html_theme = "furo"
html_title = "SpockFlow"
html_theme_options = {
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


# for the sitemap extension ---
# check if the current commit is tagged as a release (vX.Y.Z) and set the version
import subprocess
is_latest = subprocess.check_output(["bump-my-version", "show", "scm_info.distance_to_latest_tag"]) == b"0"
if is_latest:
    version = "latest"
else:
    version = str(subprocess.check_output(["bump-my-version", "show", "current_version"]))
language = "en"
html_baseurl = "https://spockflow.capinet/"
html_extra_path = ["robots.txt"]

confluence_config_path = os.path.split(__file__)[0]+"/confluence.json"
if os.path.isfile(confluence_config_path):
    import json
    with open(confluence_config_path) as fp:
        conf_config = json.load(fp)
    confluence_publish = conf_config['confluence_publish']
    confluence_parent_page = conf_config['confluence_parent_page']
    confluence_space_key = conf_config['confluence_space_key']
    confluence_ask_password = conf_config['confluence_ask_password']
    confluence_server_url = conf_config['confluence_server_url']
    confluence_server_user = conf_config['confluence_server_user']
    confluence_server_cookies = conf_config['confluence_server_cookies']
