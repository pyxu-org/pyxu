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


def broadcast_matmul_shapes(
    shape1: typ.Tuple[int, ...],
    shape2: typ.Tuple[int, ...],
) -> typ.Tuple[int, ...]:
    sh1_was_1d = False
    if len(shape1) == 1:
        sh1_was_1d = True
        shape1 = (1,) + shape1

    sh2_was_1d = False
    if len(shape2) == 1:
        sh2_was_1d = True
        shape2 = shape2 + (1,)

    if shape1[-1] != shape2[-2]:
        raise ValueError(f"Cannot @-multiply shapes {shape1}, {shape2}.")

    sh = np.broadcast_shapes(shape1[:-2], shape2[:-2])
    if sh1_was_1d and sh2_was_1d:
        pass
    elif (not sh1_was_1d) and sh2_was_1d:
        sh = sh + (shape1[-2],)
    elif sh1_was_1d and (not sh2_was_1d):
        sh = sh + (shape2[-1],)
    else:
        sh = sh + (shape1[-2], shape2[-1])
    return sh


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
        if (t := type(y)) is np.ndarray:
            return np
        elif t is da.core.Array:
            return da
        else:
            # At this point infer_api() should return `cupy` or `None`, with fallback on `None` if
            # the former is unavailable.
            try:
                import cupy as cp

                if t is cp.ndarray:
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
