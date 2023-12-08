"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options.
For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime as dt
import importlib.metadata as im
import pathlib as plib


def load_nitpick_ignore() -> list:
    # Load references to be ignored by `nitpick`.
    sphinx_src_dir = plib.Path(__file__).parent
    f_path = sphinx_src_dir / "nitpick-exceptions"

    data = set()
    for line in open(f_path, mode="r"):
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        data.add((dtype, target.strip()))
    return list(data)


# -- Project information -----------------------------------------------------
cfg = im.metadata("pyxu")
author = cfg["Author"]
copyright = f"{dt.date.today().year}, {author}"

# Compute legible version info.
version = cfg["Version"]  # <semver>[.devXXX][+<git-hash>]
version = version.strip().split("+")[0]  # restrict to <semver>[.devXXX]
release = version

# -- General configuration ---------------------------------------------------
root_doc = "index"  # legacy term = "master_doc"
default_role = "code"  # None
pygments_style = "sphinx"  # The name of the Pygments (syntax highlighting) style to use.

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source
# files.  This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

nitpicky = True
nitpick_ignore_regex = load_nitpick_ignore()
add_module_names = False
maximum_signature_line_length = 140
language = "en"

# -- Options for Python domain -----------------------------------------------
python_display_short_literal_types = True

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
            "url": "https://github.com/pyxu-org/pyxu",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyxu/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Contact",
            "url": "mailto: contact@pyxu.org",
            "icon": "fa-brands fa-telegram",
        },
        {
            "name": "EPFL Center for Imaging",
            "url": "https://imaging.epfl.ch/",
            "icon": "_static/imaging.png",
            "type": "local",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["navbar-nav"],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-version", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "searchbox", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright", "sphinx-chatgpt"],
    "pygment_light_style": "tango",
}

html_context = {
    "github_user": "pyxu-org",
    "github_repo": "pyxu",
    "github_version": "main",
    "doc_path": "doc",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here, relative to this directory. They are
# copied after the builtin static files, so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

## EXTENSION CONFIGURATION ===================================================
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# -- Options for plot_directive extension ------------------------------------
plot_include_source = True
plot_html_show_source_link = True
# plot_formats = [("png", 90)]

# -- Options for nbsphinx extension ------------------------------------------
# If the notebooks take a long time to run, pre-run them and save the outputs. The following line tells nbsphinx not to
# re-run them during the build process.
nbsphinx_execute = "never"

# -- Options for codeautolink extension --------------------------------------
codeautolink_autodoc_inject = False
codeautolink_search_css_classes = ["highlight-default"]
codeautolink_concat_default = True

# -- Option for copybutton extension -----------------------------------------
# copybutton config: strip console characters
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for sphinx_design extension -------------------------------------

# -- Options for sphinx_gallery extension ------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],  # Path to your Jupyter notebooks
    "gallery_dirs": ["examples"],  # Path where the gallery should be placed
}

# -- Options for togglebutton extension --------------------------------------

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

# -- Options for autosummary extension ---------------------------------------
autosummary_context = {}  # for template engine
autosummary_generate = False
autosummary_generate_overwrite = True
autosummary_mock_imports = []
autosummary_imported_members = True
autosummary_ignore_module_all = False

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {  # We only include most useful doc-sets.
    "python": ("https://docs.python.org/3", None),
    "NumPy [stable]": ("https://numpy.org/doc/stable/", None),
    "CuPy [latest]": ("https://docs.cupy.dev/en/latest/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/", None),
    "Dask [stable]": ("https://docs.dask.org/en/stable/", None),
    "Pytest [latest]": ("https://docs.pytest.org/en/latest/", None),
    "Matplotlib [stable]": ("https://matplotlib.org/stable/", None),
    "JAX [latest]": ("https://jax.readthedocs.io/en/latest/", None),
    "PyTorch [stable]": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for mathjax extension -------------------------------------------

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Docstrings reference annotations from pyxu.info via short-hands to improve legibility.
    # All shorthands here must be kept in sync manually with pyxu.info sub-modules.
    "ArrayModule": "~pyxu.info.ptype.ArrayModule",
    "DType": "~pyxu.info.ptype.DType",
    "Integer": "~pyxu.info.ptype.Integer",
    "NDArray": "~pyxu.info.ptype.NDArray",
    "NDArrayAxis": "~pyxu.info.ptype.NDArrayAxis",
    "NDArrayInfo": "~pyxu.info.deps.NDArrayInfo",
    "NDArrayShape": "~pyxu.info.ptype.NDArrayShape",
    "OpC": "~pyxu.info.ptype.OpC",
    "OpT": "~pyxu.info.ptype.OpT",
    "Path": "~pyxu.info.ptype.Path",
    "Real": "~pyxu.info.ptype.Real",
    "SparseArray": "~pyxu.info.ptype.SparseArray",
    "VarName": "~pyxu.info.ptype.VarName",
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for viewcode extension ------------------------------------------
viewcode_line_numbers = True
