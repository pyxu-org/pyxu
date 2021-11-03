import collections.abc as cabc
import types
import typing as typ

import dask.array as da
import numpy as np

import pycsou.util.deps as pycd

if pycd.CUPY_ENABLED:
    import cupy as cp


def infer_sum_shapes(
    shape1: typ.Tuple[int, typ.Union[int, None]],
    shape2: typ.Tuple[int, typ.Union[int, None]],
) -> typ.Tuple[int, typ.Union[int, None]]:
    if None in (shape1[0], shape2[0]):
        raise ValueError(f"Shapes with agnostic codomain dimensions are not supported.")
    elif None in (shape1[1], shape2[1]):
        out_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if None in (dim1, dim2):
                out_shape.append(dim1 if dim2 is None else dim2)
            else:
                if dim1 == dim2 or (1 in (t := (dim1, dim2))):
                    out_shape.append(np.amax(t))
                else:
                    raise ValueError(f"Cannot infer output shape for input shapes: {shape1} and {shape2}.")
    else:
        out_shape = np.broadcast_shapes(shape1, shape2)
    return tuple(out_shape)


def infer_composition_shapes(
    shape1: typ.Tuple[int, typ.Union[int, None]],
    shape2: typ.Tuple[int, typ.Union[int, None]],
) -> typ.Tuple[int, typ.Union[int, None]]:
    if None in (shape1[0], shape2[0]):
        raise ValueError(f"Shapes with agnostic codomain dimensions are not supported.")
    else:
        return (shape1[0], shape2[1])


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
