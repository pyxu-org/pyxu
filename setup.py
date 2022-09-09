#!/usr/bin/env python3

"""
Setup script.
"""

import configparser
import pathlib as plib

import setuptools


def update_extra_requires():
    cfg_file = plib.Path(__file__).parent / "setup.cfg"

    cfg = configparser.ConfigParser()
    with open(cfg_file, mode="r") as f:
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

    with open(cfg_file, mode="w") as f:
        cfg.write(f)


update_extra_requires()
setuptools.setup(setup_requires=["pbr"], pbr=True)
