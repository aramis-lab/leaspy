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
import ast
import os
import re
from datetime import date
from distutils.version import LooseVersion

import sphinx

# -- Project information -----------------------------------------------------

project = "Leaspy"


def find_var(varname: str, *py_file_paths):
    with open(os.path.join(*py_file_paths), "r") as f:
        for line in f:
            if re.match(rf"^\s*{varname}\s*=", line):
                return ast.parse(line.strip()).body[0].value
    raise RuntimeError("Unable to find `{var_name}` definition.")


# The full version, including alpha/beta/rc tags
release = find_var("__version__", "../src", "leaspy", "__init__.py").s
copyright = f"2017-{date.today().year}, Leaspy contributors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # 'sphinx.ext.doctest',
    "sphinx.ext.intersphinx",
    # 'sphinx.ext.viewcode',
    "numpydoc",
    # 'rinoh.frontend.sphinx',
    "sphinx_rtd_theme",
    # 'sphinx.ext.napoleon',
    # 'pytsdtwdoc',
    # 'sphinx_gallery.gen_gallery',
    "myst_nb",
]

# -- autoapi configuration ---------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = ["../src"]
autoapi_root = "reference/api"


# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = True
# class_members_toctree = False

# to remove leaspy. *** in index
modindex_common_prefix = ["leaspy."]  # , 'leaspy.algo.', 'leaspy.models.'

# to remove leaspy in front of all objects
add_module_names = False

# If true, suppress the module name of the python reference if it can be resolved. The default is False. (experimental)
python_use_unqualified_type_names = True

# primary domain for references
primary_domain = "py"

# - From Johann conf.py
# Use svg images for math stuff
imgmath_image_format = "svg"
# pngmath / imgmath compatibility layer for different sphinx versions

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# autoclass_content = 'class' # no __init__ method
# autodoc_inherit_docstrings = False

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "private-members": False,  # _methods are not shown
    "undoc-members": False,
    #'member-order': 'bysource', # 'groupwise' # 'alphabetical'
    #'special-members': '__init__',
    "exclude-members": "__init__,__weakref__",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
highlight_language = "python3"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
# ---sphinx-themes-----
# html_theme = 'neo_rtd_theme'
# html_theme_path = [sphinx_theme.get_html_theme_path()]

# html_theme = 'alabaster'
# html_theme = 'sphinx-theme'
html_theme = "sphinx_rtd_theme"

add_function_parentheses = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "display_version": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": None,
    # Logo and description
    # 'description': 'LEArning Spatiotemporal Patterns in Python',
    # 'logo_name': 'false',
    # 'logo_text_align': 'center',
    # GitHub stuff
    # 'github_banner': 'true',
    # 'github_repo': 'pyts',
    # 'github_type': 'star',
    # 'github_user': 'johannfaouzi',
    # Page and sidebar widths
    # 'page_width': '1300px',
    "body_max_width": "1000px",
    # 'sidebar_width': '250px',
    # Related links
    # 'show_related': 'true',
    # 'show_relbar_bottom': 'true',
    # Font sizes
    # 'font_size': '15px',
    # 'code_font_size': '13px'
}

html_context = {
    "display_gitlab": True,  # Integrate Github
    "gitlab_user": "icm-institute/aramislab",  # Username
    "gitlab_repo": "leaspy",  # Repo name
    "gitlab_version": "dev",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# Custom CSS files
# html_css_files = [
#     'custom.css',
# ]# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/images/leaspy_logo.png"

html_title = "Leaspy"
# A shorter title for the navigation bar. Default is the same as html_title.
html_short_title = "Leaspy documentation"

# -- Options for LaTeX output ---------------------------------------------
latex_engine = "pdflatex"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
    # Additional stuff for the LaTeX preamble.
}

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    # 'seaborn': ('https://seaborn.pydata.org/', None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}
