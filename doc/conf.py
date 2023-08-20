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
version = release = info["version"]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
]

root_doc = "index"
exclude_patterns = []
templates_path = []
nitpicky = True
add_module_names = False
maximum_signature_line_length = 72

# -- Options for HTML output -------------------------------------------------
html_theme = "classic"  # temporary

# -- Options for Python domain -----------------------------------------------
python_display_short_literal_types = True

# -- Extension configuration -------------------------------------------------

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

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {  # We only include most useful doc-sets.
    "python": ("https://docs.python.org/3", None),
    "NumPy [stable]": ("https://numpy.org/doc/stable/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/", None),
    "Dask [stable]": ("https://docs.dask.org/en/stable/", None),
    "Sparse [latest]": ("https://sparse.pydata.org/en/latest/", None),
    "Pytest [latest]": ("https://docs.pytest.org/en/latest/", None),
    "Matplotlib [stable]": ("https://matplotlib.org/stable/", None),
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

# -- Options for viewcode extension ------------------------------------------
# viewcode_line_numbers = True  # sphinx 7.2+

# -- Options for plot_directive extension ------------------------------------
plot_html_show_source_link = True
