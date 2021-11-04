import collections.abc as cabc
import types
import typing as typ

import dask.array as da
import numpy as np

import pycsou.util.deps as pycd

if pycd.CUPY_ENABLED:
    import cupy as cp


Shape = typ.Tuple[int, typ.Union[int, None]]


def infer_sum_shape(shape1: Shape, shape2: Shape) -> Shape:
    A, B, C, D = *shape1, *shape2
    if None in (A, C):
        raise ValueError("Addition of codomain-dimension-agnostic operators is not supported.")
    try:
        domain_None = (B is None, D is None)
        if all(domain_None):
            return np.broadcast_shapes((A,), (C,)) + (None,)
        elif any(domain_None):
            fill = lambda _: [1 if (k is None) else k for k in _]
            return np.broadcast_shapes(fill(shape1), fill(shape2))
        elif domain_match := (B == D):
            return np.broadcast_shapes((A,), (C,)) + (B,)
        else:
            raise
    except:
        raise ValueError(f"Addition of {shape1} and {shape2} operators forbidden.")


def infer_composition_shape(shape1: Shape, shape2: Shape) -> Shape:
    A, B, C, D = *shape1, *shape2
    if None in (A, C):
        raise ValueError("Composition of codomain-dimension-agnostic operators is not supported.")
    elif (B == C) or (B is None):
        return (A, D)
    else:
        raise ValueError(f"Composition of {shape1} and {shape2} operators forbidden.")


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
