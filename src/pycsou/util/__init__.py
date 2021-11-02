import collections.abc as cabc
import types
import typing as typ

import dask.array as da
import numpy as np

import pycsou.util.deps as pycd

if pycd.CUPY_ENABLED:
    import cupy as cp


def broadcast_sum_shapes(
    shape1: typ.Tuple[int, ...],
    shape2: typ.Tuple[int, ...],
) -> typ.Tuple[int, ...]:
    return np.broadcast_shapes(shape1, shape2)


def get_array_module(x: cabc.Sequence[typ.Any], fallback: types.ModuleType = None) -> types.ModuleType:
    """
    Get the array namespace corresponding to a given object.

    Parameters
    ----------
    x: cabc.Sequence[typ.Any]
        Any object which is a NumPy/CuPy/Dask array, or that can be converted to one.
    fallback: types.ModuleType
        Fallback module if `x` is not a NumPy/CuPy/Dask array.
        Default behaviour: raise error if fallback is required.

    Returns
    -------
    namespace: types.ModuleType
        The namespace to use to manipulate `x`, or `fallback`.
    """

    def infer_api(y):
        if isinstance(y, np.ndarray):
            return np
        elif isinstance(y, da.core.Array):
            return da
        elif pycd.CUPY_ENABLED and isinstance(y, cp.ndarray):
            return cp
        else:
            return None

    if (xp := infer_api(x)) is not None:
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer array module for {type(x)}.")
