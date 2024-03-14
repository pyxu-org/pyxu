"""
This module initializes Pyxu-wide config information which may be needed by some sub-modules.
Sub-modules which need it should load this module at their top level.
"""

# To handle dask-distributed and potential machine config differences, paths to system directories should be queried
# through functions.

import os
import pathlib as plib
import sys

__all__ = [
    "config_dir",
    "data_dir",
    "cache_dir",
]

resolve = lambda p: plib.Path(p).expanduser().resolve()


def xdg_config_root() -> plib.Path:
    return resolve(os.getenv("XDG_CONFIG_HOME", "~/.config"))


def xdg_data_root() -> plib.Path:
    return resolve(os.getenv("XDG_DATA_HOME", "~/.local/share"))


def xdg_cache_root() -> plib.Path:
    return resolve(os.getenv("XDG_CACHE_HOME", "~/.cache"))


def config_dir() -> plib.Path:
    # config files (if any)
    return xdg_config_root() / "pyxu"


def data_dir() -> plib.Path:
    # pyxu-shipped data (if any)
    return xdg_data_root() / "pyxu"


def cache_dir(load: bool = False) -> plib.Path:
    # runtime-generated stuff (if any)
    cdir = xdg_cache_root() / "pyxu"

    if load and (str(cdir) not in sys.path):
        sys.path.append(str(cdir))

    return cdir
