import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt

__all__ = [
    "view_as_real",
    "view_as_complex",
]


def view_as_complex(x: pxt.NDArray) -> pxt.NDArray:
    r"""
    View real-valued array as its complex-valued bijection.  (Inverse of :py:func:`~pyxu.util.view_as_real`.)

    Parameters
    ----------
    x: NDArray
        (..., N, 2) real-valued array.

    Returns
    -------
    y: NDArray
        (..., N) complex-valued array.

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real, view_as_complex
       x = np.array([[0., 1],
                     [2 , 3],
                     [4 , 5]])
       y = view_as_complex(x)  # array([0.+1.j, 2.+3.j, 4.+5.j])
       view_as_real(y) == x    # True

    Notes
    -----
    Complex-valued inputs are returned unchanged.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_real`,
    :py:func:`~pyxu.util.view_as_real_mat`,
    :py:func:`~pyxu.util.view_as_complex_mat`
    """
    assert x.ndim >= 2
    if _is_complex(x):
        return x

    try:
        r_dtype = x.dtype
        r_width = pxrt.Width(r_dtype)
        c_dtype = r_width.complex.value
    except Exception:
        raise ValueError(f"Unsupported dtype {r_dtype}.")
    assert x.shape[-1] == 2, "Last array dimension should contain real/imaginary terms only."

    y_sh = x.shape[:-1]
    y = x.view(c_dtype).reshape(y_sh)
    return y


def view_as_real(x: pxt.NDArray) -> pxt.NDArray:
    r"""
    View complex-valued array as its real-valued bijection.  (Inverse of :py:func:`~pyxu.util.view_as_complex`.)

    Parameters
    ----------
    x: NDArray
        (..., N) complex-valued array.

    Returns
    -------
    y: NDArray
        (..., N, 2) real-valued array.

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real, view_as_complex
       x = np.r_[0+1j, 2+3j, 4+5j]
       y = view_as_real(x)               # array([[0., 1.],
                                         #        [2., 3.],
                                         #        [4., 5.]])
       view_as_complex(y) == x           # True

    Notes
    -----
    Real-valued inputs are returned unchanged.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_complex`,
    :py:func:`~pyxu.util.view_as_real_mat`,
    :py:func:`~pyxu.util.view_as_complex_mat`
    """
    assert x.ndim >= 1
    if _is_real(x):
        return x

    try:
        c_dtype = x.dtype
        c_width = pxrt.CWidth(c_dtype)
        r_dtype = c_width.real.value
    except Exception:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    y_sh = (*x.shape, 2)
    y = x.view(r_dtype).reshape(y_sh)
    return y


def _is_real(x: pxt.NDArray) -> bool:
    try:
        return bool(pxrt.Width(x.dtype))
    except Exception:
        return False


def _is_complex(x: pxt.NDArray) -> bool:
    try:
        return bool(pxrt.CWidth(x.dtype))
    except Exception:
        return False
