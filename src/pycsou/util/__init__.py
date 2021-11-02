import collections.abc as cabc
import types
import typing as typ

import dask.array as da
import numpy as np


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
        else:
            # At this point infer_api() should return `cupy` or `None`, with fallback on `None` if
            # the former is unavailable.
            try:
                import cupy as cp

                if isinstance(y, cp.ndarray):
                    return cp
            except ImportError:
                pass
            finally:
                return None

    if xp := infer_api(x):
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer array API for {type(x)}.")
