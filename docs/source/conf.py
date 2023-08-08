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
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'SeisMIC'
copyright = '2023, The SeisMIC development team.'
author = \
    'Peter Makus, Christoph Sens-Schönfelder, and the SeisMIC development Team'

# The full version, including alpha/beta/rc tags
release = '0.4.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinxcontrib.mermaid',
              'sphinx_copybutton',
              'sphinx_gallery.gen_gallery',
              ]

# --------------------------------
# Sphinx Gallery config
# sphinx_gallery_conf = {
#     # path to your example scripts
#     'examples_dirs': ['../examples/tutorials'],
#     # path to where to save gallery generated output
#     'gallery_dirs': ["tutorials"],
#     # Checks matplotlib for figure creation
#     'image_scrapers': ('matplotlib'),
#     # Which files to include
#     'filename_pattern': r"\.py"
# }

# ---------------------------------

# autosummary_generate = True

# For docstring __init__ documentation
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    # 'members': 'var1, var2',
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__init__',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'  # 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_title = ""
html_logo = "source/figures/seismic_logo_small.svg"
# html_favicon = "chapters/figures/favicon.ico"

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise
    "github_user": 'PeterMakus',
    "github_repo": 'SeisMIC',
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "github_url": "https://github.com/PeterMakus/SeisMIC",
    "use_edit_page_button": True,
    # "show_toc_level": 1,

    "use_edit_page_button": True,

    # "collapse_navigation": True,
    "navigation_depth": 2,
    # "navbar_align": "left"

}
