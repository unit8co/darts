# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

# -- Project information -----------------------------------------------------

project = "darts"
copyright = f"2020 - {datetime.now().year}, Unit8 SA (Apache 2.0 License)"
author = "Unit8 SA"
version = "0.38.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension"]

templates_path = ["_templates"]
exclude_patterns = []

autoapi_dirs = ["../../darts"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "static/darts-logo-trim.png"
html_favicon = "static/docs-favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/unit8co/darts",
    "twitter_url": "https://twitter.com/unit8co",
}
