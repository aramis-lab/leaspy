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

import sphinx
from sphinx_gallery.sorting import FileNameSortKey

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
copyright = f"2017-{date.today().year}, Juliette Ortholand, Nicolas Gensollen, Etienne Maheux, Caglayan Tuna, Raphael Couronne, Arnaud Valladier, Sofia Kaisaridi, Pierre-Emmanuel Poulet, Nemo Fournier, Léa Aguilhon, Maylis Tran, Gabrielle Casimiro, Igor Koval, Stanley Durrleman, Sophie Tezenas Du Montcel"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# sphinx-copybutton: strip shell prompts from copied text
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

bibtex_bibfiles = ["references.bib"]

# -- autoapi configuration ---------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
]
nb_execution_timeout = 600
autoapi_dirs = ["../src"]
autoapi_root = "reference/api"
autodoc_typehints = "description"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],  # path to your example scripts
    "gallery_dirs": ["auto_examples"],  # path to where to save gallery generated output
    "notebook_images": "https://leaspy.readthedocs.io/en/stable/_images/",
    "plot_gallery": True,
    "within_subsection_order": FileNameSortKey,
}

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

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/*.py",
    "auto_examples/*.ipynb",
    "auto_examples/*.py.md5",
    "auto_examples/*.codeobj.json",
    "data_summary.ipynb",
]

show_warning_types = True

suppress_warnings = [
    "config.cache",  # sphinx_gallery_conf unpicklable
    # "autoapi",      # only if you decide to silence AutoAPI warnings too
]

# The name of the Pygments (syntax highlighting) style to use.
# PyData supports separate styles for light and dark mode.
highlight_language = "python3"
pygments_style = "friendly"        # light mode
pygments_dark_style = "monokai"    # dark mode

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

add_function_parentheses = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_js_files = ["custom.js"]

# Favicon
html_favicon = "_static/favicon.png"

# Theme options
html_theme_options = {
    # Navbar
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 6,
    # Secondary (right) sidebar
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    # Logo
    "logo": {
        "image_light": "_static/images/leaspy_logo.png",
        "image_dark": "_static/images/leaspy_logo.png",
    },
    # GitHub icon in navbar
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/aramis-lab/leaspy",
            "icon": "fa-brands fa-github",
        }
    ],
}

html_context = {
    "github_user": "aramis-lab",
    "github_repo": "leaspy",
    "github_version": "v2",
    "doc_path": "docs",
}

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
