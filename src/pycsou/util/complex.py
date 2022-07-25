import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    "view_as_real",
    "view_as_complex",
]


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

    Note
    ----
    This method is a no-op if the input is complex-valued.
    """
    if _is_complex(x):
        return x

    try:
        r_dtype = x.dtype
        r_width = pycrt.Width(r_dtype)
        c_dtype = r_width.complex.value
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

    Note
    ----
    This method is a no-op if the input is real-valued.
    """
    if _is_real(x):
        return x

    try:
        c_dtype = x.dtype
        c_width = pycrt._CWidth(c_dtype)
        r_dtype = c_width.real.value
    except:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    r_sh = (*x.shape[:-1], 2 * x.shape[-1])
    y = x.view(r_dtype).reshape(r_sh)
    return y


def _is_real(x: pyct.NDArray) -> bool:
    try:
        return bool(pycrt.Width(x.dtype))
    except:
        return False


def _is_complex(x: pyct.NDArray) -> bool:
    try:
        return bool(pycrt._CWidth(x.dtype))
    except:
        return False
