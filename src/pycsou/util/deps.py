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


NDArray = typ.TypeVar("NDArray", *supported_array_types())
supported_array_modules_literals = [typ.Literal[mod] for mod in supported_array_modules()]
ArrayModule = typ.TypeVar("ArrayModule", *supported_array_modules_literals)
