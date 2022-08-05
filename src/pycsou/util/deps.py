import importlib.util

CUPY_ENABLED: bool = importlib.util.find_spec("cupy") is not None
if CUPY_ENABLED:
    try:
        import cupy
    except ImportError:
        # CuPy is installed, but GPU drivers probably missing.
        CUPY_ENABLED = False


def array_backend_info():
    """
    List of all (known) installed NumPy-compatible (array, API, short-name) triplets.

    This function is most useful for testing purposes.
    """
    import dask.array as da
    import numpy as np

    info = []
    info.append((np.ndarray, np, "NUMPY"))
    info.append((da.core.Array, da, "DASK"))

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
    List of all (supported & installed) sparse (array, API, short-name) triplets.
    """
    import scipy.sparse as sp
    import sparse

    info = []
    info.append((sp.spmatrix, sp, "SCIPY_SPARSE"))
    info.append((sparse.SparseArray, sparse, "PYDATA_SPARSE"))

    if CUPY_ENABLED:
        import cupyx.scipy.sparse as csp

        info.append((csp.spmatrix, csp, "CUPY_SPARSE"))
    return info


def supported_sparse_types():
    return tuple(_[0] for _ in sparse_backend_info())


def supported_sparse_modules():
    return tuple(_[1] for _ in sparse_backend_info())
