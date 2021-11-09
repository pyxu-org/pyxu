import functools
import importlib.util
import typing as typ

import dask.array as da
import numpy as np

CUPY_ENABLED = importlib.util.find_spec("cupy") is not None


def array_backend_info():
    """
    List of all (known) installed NumPy-compatible (array, API) pairs.

    This function is most useful for testing purposes.
    """
    info = {
        np.ndarray: np,
        da.core.Array: da,
    }
    if CUPY_ENABLED:
        import cupy as cp

        info[cp.ndarray] = cp
    return info


def supported_array_types():
    return tuple(array_backend_info().keys())


def supported_array_modules():
    return tuple(array_backend_info().values())


NDArray = functools.reduce(
    lambda a, b: typ.Union[a, b],
    supported_array_types()[1:],
    supported_array_types()[0],
)
