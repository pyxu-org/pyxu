import collections.abc as cabc
import functools
import inspect
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


def infer_array_module(i: str) -> cabc.Callable:
    """
    Decorator to auto-fill the `_xp` parameter of the called function.

    Parameters
    ----------
    i: str
        Parameter from which array module must be inferred. Must have a NumPy API.

    Example
    -------
    >>> import pycsou.util as pycu
    >>> @pycu.infer_array_module('x')
    ... def f(x, y, _xp=None):
    ...     print(_xp.__name__)
    ...     return _xp.ones(len(x)) + x, y
    ... x, y = np.arange(5), 2
    ... out = f(x, y)  # -> numpy
    """

    def decorator(func: cabc.Callable) -> cabc.Callable:
        @functools.wraps(func)
        def wrapper(*ARGS, **KWARGS):
            sig = inspect.Signature.from_callable(func)
            func_args = sig.bind(*ARGS, **KWARGS)
            func_args.apply_defaults()
            func_args = func_args.arguments
            for k in (i, "_xp"):
                if k not in func_args:
                    error_msg = f"Parameter[{k}] not part of {func.__qualname__}() parameter list."
                    raise ValueError(error_msg)

            func_args["_xp"] = get_array_module(func_args[i])
            return func(**func_args)

        return wrapper

    return decorator
