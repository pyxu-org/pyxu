#!/usr/bin/env python3

"""
Setup script.
"""

import configparser
import pathlib as plib

import setuptools


def read_file(path: plib.Path) -> str:
    with open(path, mode="r") as f:
        txt = f.read()
    return txt


def write_file(path: plib.Path, txt: str):
    with open(path, mode="w") as f:
        f.write(txt)


def update_extra_requires(cfg_path: plib.Path):
    # overwrite .CFG file with added targets `complete_gpu`, `complete_no_gpu`.
    cfg = configparser.ConfigParser()
    with open(cfg_path, mode="r") as f:
        cfg.read_file(f)

    # Add aggregate targets
    #   * complete_no_gpu
    #   * complete_gpu
    pkg = {"no_gpu": set(), "gpu": set()}
    pkg_blacklist = {
        "dev",
        "complete_no_gpu",
        "complete_gpu",
    }
    xtra = cfg["options.extras_require"]
    for xtra_name, xtra_values in xtra.items():
        if xtra_name not in pkg_blacklist:
            pkg["gpu"].add(xtra_values)
            if not xtra_name.endswith("_gpu"):
                pkg["no_gpu"].add(xtra_values)
    xtra["complete_no_gpu"] = "".join(pkg["no_gpu"])
    xtra["complete_gpu"] = "".join(pkg["gpu"])

    with open(cfg_path, mode="w") as f:
        cfg.write(f)


cfg_path = plib.Path(__file__).parent / "setup.cfg"
cfg_init = read_file(cfg_path)  # Save setup.cfg original state
update_extra_requires(cfg_path)

setuptools.setup(setup_requires=["pbr"], pbr=True)

write_file(cfg_path, cfg_init)  # Restore setup.cfg to original state
