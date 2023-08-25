"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import collections.abc as cabc
import configparser
import datetime as dt
import os
import pathlib as plib
import re


def setup_config() -> configparser.ConfigParser:
    """
    Load information contained in `setup.cfg`.
    """
    sphinx_src_dir = plib.Path(__file__).parent
    setup_path = sphinx_src_dir / ".." / "setup.cfg"
    setup_path = setup_path.resolve(strict=True)

    with setup_path.open(mode="r") as f:
        cfg = configparser.ConfigParser()
        cfg.read_file(f)
    return cfg


def pkg_info() -> cabc.Mapping:
    """
    Load information contained in `PKG-INFO`.
    """
    sphinx_src_dir = plib.Path(__file__).parent
    info_path = sphinx_src_dir / ".." / "src" / "pyxu.egg-info" / "PKG-INFO"
    info_path = info_path.resolve(strict=True)

    # Pattern definitions
    pat_version = r"Version: (.+)$"

    with info_path.open(mode="r") as f:
        info = dict(version=None)
        for line in f:
            if (m := re.match(pat_version, line)) is not None:
                info["version"] = m.group(1)
    return info


# -- Project information -----------------------------------------------------
cfg, info = setup_config(), pkg_info()
project = cfg.get("metadata", "name")
author = cfg.get("metadata", "author")
copyright = f"{dt.date.today().year}, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = info["version"]
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"

# The full version, including alpha/beta/rc tags.
release = version

master_doc = "index"
default_role = "code"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_togglebutton",
    # "_extension.gallery_directive",
    # For extension examples and demos
    # "ablog",
    # "jupyter_sphinx",
    "matplotlib.sphinxext.plot_directive",
    # "myst_nb",
    # "sphinxcontrib.youtube",
    # "nbsphinx",  # Uncomment and comment-out MyST-NB for local testing purposes.
    "numpydoc",
    # "jupyterlite_sphinx",
    # "sphinx_favicon",
    # "notfound.extension",
]

nitpicky = False
add_module_names = False
maximum_signature_line_length = 140


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for Python domain -----------------------------------------------
python_display_short_literal_types = True

# -- autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- Ablog options -----------------------------------------------------------

# blog_path = "examples/blog/index"
# blog_authors = {
#    "pydata": ("PyData", "https://pydata.org"),
#    "jupyter": ("Jupyter", "https://jupyter.org"),
# }

# codeautolink
codeautolink_autodoc_inject = False
codeautolink_search_css_classes = ["highlight-default"]
codeautolink_concat_default = True

# Plot options
plot_include_source = True
plot_html_show_source_link = True
# plot_formats = [("png", 90)]

# copybutton config: strip console characters
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": None,
    "member-order": "bysource",
    "undoc-members": None,
    # "private-members": None,
    # "special-members": None,
    # "inherited-members": None,
    "show-inheritance": None,
    "imported-members": None,
    # "exclude-members": "__module__,",
    # "class-doc-from": None,
    # "no-value": None,
}
autodoc_typehints = "description"
autodoc_type_aliases = {}  # works only if `__futures__.annotations` imported
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True


# numpydoc configuration

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "of",
    "or",
    "optional",
    "default",
    "1D",
    "2D",
    "3D",
    "n-dimensional",
    "M",
    "N",
    "K",
}
numpydoc_xref_aliases = {
    "ndarray": ":class:`~numpy.ndarray`",
    "matplotlib_axes": ":class:`matplotlib Axes <matplotlib.axes.Axes>`",
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Docstrings reference annotations from pyxu.info via short-hands to improve legibility.
    # All shorthands here must be kept in sync manually with pyxu.info sub-modules.
    "NDArrayInfo": "~pyxu.info.deps.NDArrayInfo",
    "NDArrayShape": "~pyxu.info.ptype.NDArrayShape",
    "NDArray": "~pyxu.info.ptype.NDArray",
    "ArrayModule": "~pyxu.info.ptype.ArrayModule",
    "OpShape": "~pyxu.info.ptype.OpShape",
    "OpT": "~pyxu.info.ptype.OpT",
    "OpC": "~pyxu.info.ptype.OpC",
    "Integer": "~pyxu.info.ptype.Integer",
    "Real": "~pyxu.info.ptype.Real",
    "DType": "~pyxu.info.ptype.DType",
    "Path": "~pyxu.info.ptype.Path",
    "VarName": "~pyxu.info.ptype.VarName",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Pyxu Documentation"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_sourcelink_suffix = ""
html_short_title = "Pyxu"

html_theme_options = {
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/matthieumeo/pycsou",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pycsou/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Contact",
            "url": "mailto: matthieu.simeoni@gmail.com",
            "icon": "fa-brands fa-telegram",
        },
        {
            "name": "EPFL Center for Imaging",
            "url": "https://imaging.epfl.ch/",
            "icon": "_static/imaging.png",
            "type": "local",
        },
    ],
    # alternative way to set twitter and github header icons
    # "github_url": "https://github.com/pydata/pydata-sphinx-theme",
    # "twitter_url": "https://twitter.com/PyData",
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["navbar-nav"],
    # "announcement": "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/main/docs/_templates/custom-template.html",
    # "show_nav_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-version", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "article_footer_items": ["test.html", "test.html"],
    # "content_footer_items": ["test.html", "test.html"],
    # "footer_start": ["test.html", "test.html"],
    "secondary_sidebar_items": ["page-toc", "searchbox", "edit-this-page", "sourcelink"],
    # "google_analytics_id": "G-W1G68W77YV",
    # "switcher": {
    #    "json_url": json_url,
    #    "version_match": version_match,
    # },
    "pygment_light_style": "tango",
}

# html_sidebars = {
# "community/index": [
#    "sidebar-nav-bs",
#    "custom-template",
# ],  # This ensures we test for custom sidebars
# "examples/no-sidebar": [],  # Test what page looks like with no sidebar items
# "examples/persistent-search-field": ["search-field"],
# Blog sidebars
# ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
# "examples/blog/*": [
#    "ablog/postcard.html",
#    "ablog/recentposts.html",
#    "ablog/tagcloud.html",
#    "ablog/categories.html",
#    "ablog/authors.html",
#    "ablog/languages.html",
#    "ablog/locations.html",
#    "ablog/archives.html",
# ],
# }

html_context = {
    "github_user": "matthieumeo",
    "github_repo": "pycsou",
    "github_version": "main",
    "doc_path": "doc",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_js_files = ["custom-icon.js"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
# htmlhelp_basename = "Pyxudoc"

# -- favicon options ---------------------------------------------------------

# see https://sphinx-favicon.readthedocs.io for more information about the
# sphinx-favicon extension
# favicons = [
#     # generic icons compatible with most browsers
#     "favicon-32x32.png",
#     "favicon-16x16.png",
#     {"rel": "shortcut icon", "sizes": "any", "href": "favicon.ico"},
#     # chrome specific
#     "android-chrome-192x192.png",
#     # apple icons
#     {"rel": "mask-icon", "color": "#459db9", "href": "safari-pinned-tab.svg"},
#     {"rel": "apple-touch-icon", "href": "apple-touch-icon.png"},
#     # msapplications
#     {"name": "msapplication-TileColor", "content": "#459db9"},
#     {"name": "theme-color", "content": "#ffffff"},
#     {"name": "msapplication-TileImage", "content": "mstile-150x150.png"},
# ]

# Example configuration for intersphinx
intersphinx_mapping = {
    "mpl": ("https://matplotlib.org/stable", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "finufft": ("https://finufft.readthedocs.io/en/latest/", None),
    "sphinx-primer": ("https://sphinx-primer.readthedocs.io/en/latest/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3", None),
    "NumPy [stable]": ("https://numpy.org/doc/stable/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/", None),
    "Dask [stable]": ("https://docs.dask.org/en/stable/", None),
    "Sparse [latest]": ("https://sparse.pydata.org/en/latest/", None),
    "Pytest [latest]": ("https://docs.pytest.org/en/latest/", None),
    "Matplotlib [stable]": ("https://matplotlib.org/stable/", None),
}
