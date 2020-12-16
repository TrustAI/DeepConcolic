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
sys.path.insert(0, os.path.abspath('../../src/'))


# -- Project information -----------------------------------------------------

project = 'deepconcolic'
copyright = '2020, TrustAI'
author = 'TrustAI'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',               # in case
    # 'sphinx_autodoc_typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bizstyle'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for Autodoc -----------------------------------------------------

autodoc_mock_imports = ['tensorflow', 'sklearn', 'pomegranate',
                        'cv2', 'typing']
autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': 'maketrans'
}



# -- Options for Napoleon ----------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# # https://stackoverflow.com/questions/49540656/how-to-automatically-add-parameter-types-in-sphinx-documentation/51312475#51312475
# napoleon_use_param = True

# -- Options for Autosummary -------------------------------------------------

autosummary_generate = False
autosummary_imported_members = True


# -- Options for Intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'tensorflow': ("https://www.tensorflow.org/api_docs/python",
                   "https://raw.githubusercontent.com/mr-ubik/tensorflow-intersphinx/master/tf2_py_objects.inv"),
    'pomegranate': ('https://pomegranate.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org', None)
}
