"""
This module initializes Pyxu-wide config information which may be needed by some sub-modules.
Sub-modules which need it should load this module at their top level.
"""

import os
import pathlib as plib
import sys

__all__ = [
    "config_dir",
    "data_dir",
    "cache_dir",
]

resolve = lambda p: plib.Path(p).expanduser().resolve()

xdg_config_root: plib.Path = resolve(os.getenv("XDG_CONFIG_HOME", "~/.config"))
xdg_data_root: plib.Path = resolve(os.getenv("XDG_DATA_HOME", "~/.local/share"))
xdg_cache_root: plib.Path = resolve(os.getenv("XDG_CACHE_HOME", "~/.cache"))

config_dir: plib.Path = xdg_config_root / "pyxu"  # config files (if any)
data_dir: plib.Path = xdg_data_root / "pyxu"  # pyxu-shipped data (if any)
cache_dir: plib.Path = xdg_cache_root / "pyxu"  # runtime-generated stuff (if any)

for folder in [config_dir, data_dir, cache_dir]:
    folder.mkdir(parents=True, exist_ok=True)

sys.path.append(str(cache_dir))  # runtime-generated Python modules will lie here
