import importlib.util

import dask.array as da
import numpy as np
import scipy.sparse as scisp
import sparse as sp

CUPY_ENABLED: bool = importlib.util.find_spec("cupy") is not None


def array_backend_info():
    """
    List of all (known) installed NumPy-compatible (array, API, short-name) triplets.

    This function is most useful for testing purposes.
    """
    info = [
        (np.ndarray, np, "NUMPY"),
        (da.core.Array, da, "DASK"),
    ]
    if CUPY_ENABLED:
        import cupy as cp

        info.append((cp.ndarray, cp, "CUPY"))
    return tuple(info)


def supported_array_types():
    return tuple(_[0] for _ in array_backend_info())


def supported_array_modules():
    return tuple(_[1] for _ in array_backend_info())


def sparse_backend_info():
    r"""
    List of all (known) installed sparse array modules.
    """
    info = {
        sp.SparseArray: sp,
        scisp.spmatrix: scisp,
    }
    if CUPY_ENABLED:
        import cupyx.scipy.sparse as cpxsp

        info[cpxsp.spmatrix] = cpxsp
    return info


def supported_sparse_types():
    return tuple(sparse_backend_info().keys())


def supported_sparse_modules():
    return tuple(sparse_backend_info().values())
