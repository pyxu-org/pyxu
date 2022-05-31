import enum

import numpy as np

import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    "view_as_real",
    "view_as_complex",
]


@enum.unique
class _CWidth(enum.Enum):
    """
    Machine-dependent complex-valued floating-point types.
    """

    # HALF = np.dtype(np.chalf)  # unsupported by NumPy
    SINGLE = np.dtype(np.csingle)
    DOUBLE = np.dtype(np.cdouble)
    QUAD = np.dtype(np.clongdouble)


def view_as_complex(x: pyct.NDArray) -> pyct.NDArray:
    """
    View real-valued array as its complex-valued bijection.

    Parameters
    ----------
    x: NDArray
        (..., 2N) real-valued array.

    Returns
    -------
    y: NDArray
        (..., N) complex-valued array.
    """
    try:
        r_dtype = x.dtype
        r_width = pycrt.Width(r_dtype)
        c_width = _CWidth[r_width.name]
        c_dtype = c_width.value
    except:
        raise ValueError(f"Unsupported dtype {r_dtype}.")
    assert x.shape[-1] % 2 == 0, "Last array dimension should be even-valued."

    c_sh = (*x.shape[:-1], x.shape[-1] // 2)
    y = x.view(c_dtype).reshape(c_sh)
    return y


def view_as_real(x: pyct.NDArray) -> pyct.NDArray:
    """
    View complex-valued array as its real-valued bijection.

    Parameters
    ----------
    x: NDArray
        (..., N) complex-valued array

    Returns
    -------
    y: NDArray
        (..., 2N) real-valued array.
    """
    try:
        c_dtype = x.dtype
        c_width = _CWidth(c_dtype)
        r_width = pycrt.Width[c_width.name]
        r_dtype = r_width.value
    except:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    r_sh = (*x.shape[:-1], 2 * x.shape[-1])
    y = x.view(r_dtype).reshape(r_sh)
    return y
