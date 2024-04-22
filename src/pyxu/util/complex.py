import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt

__all__ = [
    "as_real_op",
    "require_viewable",
    "view_as_real",
    "view_as_complex",
]


def require_viewable(x: pxt.NDArray) -> pxt.NDArray:
    """
    Copy array if required to do real/complex view manipulations.

    Real/complex view manipulations are feasible if the last axis is contiguous.

    Parameters
    ----------
    x: NDArray

    Returns
    -------
    y: NDArray
    """
    N = pxd.NDArrayInfo
    ndi = N.from_obj(x)
    if ndi == N.DASK:
        # No notion of contiguity for Dask graphs -> always safe.
        y = x
    elif ndi in (N.NUMPY, N.CUPY):
        if x.strides[-1] == x.dtype.itemsize:
            y = x
        else:
            y = x.copy(order="C")
    else:
        msg = f"require_viewable() not yet defined for {ndi}."
        raise NotImplementedError(msg)
    return y


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
    :py:func:`~pyxu.util.as_real_op`,
    :py:func:`~pyxu.util.view_as_real`
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

    y = x.view(c_dtype)  # (..., N, 1)
    y = y[..., 0]  # (..., N)
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
    :py:func:`~pyxu.util.as_real_op`,
    :py:func:`~pyxu.util.view_as_complex`
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

    y = x.view(r_dtype)  # (..., 2N)

    ndi = pxd.NDArrayInfo.from_obj(x)
    if ndi == pxd.NDArrayInfo.DASK:
        y = y.map_blocks(  # (..., N, 2)
            lambda blk: blk.reshape(
                *blk.shape[:-1],
                blk.shape[-1] // 2,
                2,
            ),
            chunks=(*x.chunks, 2),
            new_axis=x.ndim,
            meta=y._meta,
        )
    else:
        y = y.reshape(*x.shape, 2)  # (..., N, 2)
    return y


def as_real_op(A: pxt.NDArray, dim_rank: pxt.Integer = None) -> pxt.NDArray:
    r"""
    View complex-valued linear operator as its real-valued equivalent.

    Useful to transform complex-valued matrix/vector products to their real-valued counterparts.

    Parameters
    ----------
    A: NDArray
        (N1...,NK, M1,...,MD) complex-valued matrix.
    dim_rank: Integer
        Dimension rank :math:`D`. (Can be omitted if `A` is 2D.)

    Returns
    -------
    A_r: NDArray
        (N1,...,NK,2, M1,...,MD,2) real-valued equivalent.

    Examples
    --------

    .. code-block:: python3

       import numpy as np
       import pyxu.util.complex as cpl

       codim_shape = (1,2,3)
       dim_shape = (4,5,6,7)
       dim_rank = len(dim_shape)

       rng = np.random.default_rng(0)
       A =      rng.standard_normal((*codim_shape, *dim_shape)) \
         + 1j * rng.standard_normal((*codim_shape, *dim_shape))    # (1,2,3  |4,5,6,7  )
       A_r = cpl.as_real_op(A, dim_rank=dim_rank)                  # (1,2,3,2|4,5,6,7,2)

       x =      rng.standard_normal(dim_shape) \
         + 1j * rng.standard_normal(dim_shape)                     # (4,5,6,7  )
       x_r = cpl.view_as_real(x)                                   # (4,5,6,7,2)

       y = np.tensordot(A, x, axes=dim_rank)                       # (1,2,3  )
       y_r = np.tensordot(A_r, x_r, axes=dim_rank+1)               # (1,2,3,2)

       np.allclose(y, cpl.view_as_complex(y_r))                    # True


    Notes
    -----
    Real-valued matrices are returned unchanged.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_real`,
    :py:func:`~pyxu.util.view_as_complex`
    """
    if _is_real(A):
        return A

    try:
        c_dtype = A.dtype
        c_width = pxrt.CWidth(c_dtype)
        r_dtype = c_width.real.value
    except Exception:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    if A.ndim == 2:
        dim_rank = 1  # doesn't matter what the user specified.
    else:  # rank > 2
        # if ND -> mandatory supplied & (1 <= dim_rank < A.ndim)
        assert dim_rank is not None, "Dimension rank must be specified for ND operators."
        assert 1 <= dim_rank < A.ndim
    dim_shape = A.shape[-dim_rank:]
    codim_shape = A.shape[:-dim_rank]
    codim_rank = len(codim_shape)

    xp = pxd.NDArrayInfo.from_obj(A).module()
    A_r = xp.zeros((*codim_shape, 2, *dim_shape, 2), dtype=r_dtype)

    codim_sel = [*(slice(None),) * codim_rank, 0]
    dim_sel = [*(slice(None),) * dim_rank, 0]
    A_r[*codim_sel, *dim_sel] = A.real

    codim_sel = [*(slice(None),) * codim_rank, 1]
    dim_sel = [*(slice(None),) * dim_rank, 1]
    A_r[*codim_sel, *dim_sel] = A.real

    codim_sel = [*(slice(None),) * codim_rank, 0]
    dim_sel = [*(slice(None),) * dim_rank, 1]
    A_r[*codim_sel, *dim_sel] = -A.imag

    codim_sel = [*(slice(None),) * codim_rank, 1]
    dim_sel = [*(slice(None),) * dim_rank, 0]
    A_r[*codim_sel, *dim_sel] = A.imag
    return A_r


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
