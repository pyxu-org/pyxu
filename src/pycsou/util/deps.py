import collections.abc as cabc
import importlib.util
import types

import dask.array as da
import numpy as np

CUPY_ENABLED = importlib.util.find_spec("cupy") is not None


def array_modules() -> cabc.Sequence[types.ModuleType]:
    """
    List of all (known) installed NumPy-compatible APIs.

    This function is most useful for testing purposes.
    """
    xp = [np, da]

    if CUPY_ENABLED:
        import cupy as cp

        xp.append(cp.ndarray)

    return xp
