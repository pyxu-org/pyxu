import collections
import collections.abc as cabc
import functools
import itertools
import types

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.info.plugin import _load_entry_points

__all__ = [
    "stack",
    "block_diag",
]

__all__ = _load_entry_points(globals(), group="pyxu.opt.blocks", names=__all__)


def stack(
    ops: cabc.Sequence[pxt.OpT],
    axis: pxt.Integer,
) -> pxt.OpT:
    pass

def block_diag(ops: cabc.Sequence[pxt.OpT]) -> pxt.OpT:
    pass