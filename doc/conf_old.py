import collections.abc as cabc
import configparser
import datetime as dt
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
    info_path = sphinx_src_dir / ".." / "src" / "pycsou.egg-info" / "PKG-INFO"
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
version = release = info["version"]

# -- General configuration ---------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "notfound.extension",
    "sphinx_copybutton",
]


templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = []
pygments_style = "sphinx"
add_module_names = False
plot_include_source = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": -1, "titles_only": False}

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = "Pycsou"
html_context = {
    "menu_links_name": "Repository",
    "menu_links": [
        ('<i class="fa fa-github fa-fw"></i> Source Code', "https://github.com/matthieumeo/pycsou"),
    ],
    "doc_path": "docs/source",
    "github_project": "matthieumeo",
    "github_repo": "pycsou",
    "github_version": "master",
}

# -- Extension configuration -------------------------------------------------
# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    # 'inherited-members',
    "show-inheritance",
]
autodoc_inherit_docstrings = False

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
    "dask [latest]": ("https://docs.dask.org/en/latest", None),
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# Aliases (only works with from __future__ import annotations)
autodoc_type_aliases = {"ArrayLike": "NDArray"}


def skip(app, what, name, obj, would_skip, options):
    if name in {
        "__init__",
        "__add__",
        "__mul__",
        "__rmul__",
        "__pow__",
        "__truediv__",
        "__sub__",
        "__neg__",
    }:
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
