import dask
import numpy as np

import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def infer_sum_shape(sh1: pyct.Shape, sh2: pyct.Shape) -> pyct.Shape:
    A, B, C, D = *sh1, *sh2
    if None in (A, C):
        raise ValueError("Addition of codomain-dimension-agnostic operators is not supported.")
    try:
        domain_None = (B is None, D is None)
        if all(domain_None):
            return np.broadcast_shapes((A,), (C,)) + (None,)
        elif any(domain_None):
            fill = lambda _: [1 if (k is None) else k for k in _]
            return np.broadcast_shapes(fill(sh1), fill(sh2))
        elif domain_match := (B == D):
            return np.broadcast_shapes((A,), (C,)) + (B,)
        else:
            raise
    except:
        raise ValueError(f"Addition of {sh1} and {sh2} operators forbidden.")


def infer_composition_shape(sh1: pyct.Shape, sh2: pyct.Shape) -> pyct.Shape:
    A, B, C, D = *sh1, *sh2
    if None in (A, C):
        raise ValueError("Composition of codomain-dimension-agnostic operators is not supported.")
    elif (B == C) or (B is None):
        return (A, D)
    else:
        raise ValueError(f"Composition of {sh1} and {sh2} operators forbidden.")


def get_array_module(x, fallback: pyct.ArrayModule = None) -> pyct.ArrayModule:
    """
    Get the array namespace corresponding to a given object.

    Parameters
    ----------
    x: object
        Any object compatible with the interface of NumPy arrays.
    fallback: pycsou.util.ptype.ArrayModule
        Fallback module if `x` is not a NumPy-like array.
        Default behaviour: raise error if fallback used.

    Returns
    -------
    namespace: pycsou.util.ptype.ArrayModule
        The namespace to use to manipulate `x`, or `fallback` if provided.
    """

    def infer_api(y):
        for array_t, api in pycd.array_backend_info().items():
            if isinstance(y, array_t):
                return api
        return None

    if (xp := infer_api(x)) is not None:
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer array module for {type(x)}.")


def compute(*args, mode: str = "compute", **kwargs):
    """
    Force computation of Dask collections.

    Parameters
    ----------
    *args: object | sequence(object)
        Any number of objects. If it is a dask object, it is evaluated and the result is returned.
        Non-dask arguments are passed through unchanged.
    mode: str
        Dask evaluation strategy: compute or persist.
    kwargs: dict
        Extra keyword parameters forwarded to `dask.[compute, persist]`.

    Returns
    -------
    *cargs: object | sequence(object)
        Evaluated objects. Non-dask arguments are passed through unchanged.
    """
    try:
        func = dict(compute=dask.compute, persist=dask.persist)[mode.lower()]
    except:
        raise ValueError(f"mode: expected compute/persist, got {mode}.")

    cargs = func(*args, **kwargs)
    if len(args) == 1:
        cargs = cargs[0]
    return cargs
