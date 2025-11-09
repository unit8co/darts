# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "darts"
copyright = f"2020 - {datetime.now().year}, Unit8 SA (Apache 2.0 License)"
author = "Unit8 SA"
version = "0.38.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    # "sphinx.ext.autodoc",
    # "sphinx_autodoc_typehints",
]

templates_path = ["templates"]
exclude_patterns = []

# trail 2: type hints
autodoc_typehints = "description"


# -- Sphinx-AutoAPI Configuration Options ------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_options

autoapi_dirs = ["../../darts"]
# autoapi_template_dir
# autoapi_file_patterns
# autoapi_generate_api_docs

autoapi_options = [
    "members",  # validated
    "undoc-members",  # validated
    # 'private-members',  # validated
    "show-inheritance",  # validated
    "show-module-summary",  # validated
    # 'special-members',  # validated
    # 'imported-members',  # validated
]
# autoapi_ignore  # populate
autoapi_root = "generated_api"  # validated
# autoapi_add_toctree_entry=True  # validated
autoapi_python_class_content = "both"  # validated
# autoapi_member_order
# autoapi_python_use_implicit_namespaces
# autoapi_prepare_jinja_env
# autoapi_own_page_level = "class"  # this could be nice on "class-level" instead of default "module"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "static/darts-logo-trim.png"
html_favicon = "static/docs-favicon.ico"
html_static_path = ["static"]

html_theme_options = {
    "github_url": "https://github.com/unit8co/darts",
    "twitter_url": "https://twitter.com/unit8co",
}
