import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt

__all__ = [
    "view_as_real",
    "view_as_complex",
    "view_as_real_mat",
    "view_as_complex_mat",
]


def view_as_complex(x: pxt.NDArray) -> pxt.NDArray:
    r"""
    View real-valued array as its complex-valued bijection.  (Inverse of :py:func:`~pyxu.util.view_as_real`.)

    Parameters
    ----------
    x: NDArray
        (..., 2N) real-valued array.

    Returns
    -------
    y: NDArray
        (..., N) complex-valued array.

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real, view_as_complex
       x = np.arange(6.0)      # array([0., 1., 2., 3., 4., 5.])
       y = view_as_complex(x)  # array([0.+1.j, 2.+3.j, 4.+5.j])
       view_as_real(y) == x    # True

    Notes
    -----
    Complex-valued inputs are returned unchanged.  For real-valued inputs, this function acts on the last axis as:

    .. math::

       y_n = x_{2n-1}+j \, x_{2n}, \qquad 1\leq n\leq N.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_real`,
    :py:func:`~pyxu.util.view_as_real_mat`,
    :py:func:`~pyxu.util.view_as_complex_mat`
    """
    if _is_complex(x):
        return x

    try:
        r_dtype = x.dtype
        r_width = pxrt.Width(r_dtype)
        c_dtype = r_width.complex.value
    except Exception:
        raise ValueError(f"Unsupported dtype {r_dtype}.")
    assert x.shape[-1] % 2 == 0, "Last array dimension should be even-valued."

    c_sh = (*x.shape[:-1], x.shape[-1] // 2)
    y = x.view(c_dtype).reshape(c_sh)
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
        (..., 2N) real-valued array.

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real, view_as_complex
       x = np.r_[:3] + 1j * np.r_[2:5]   # array([0.+2.j, 1.+3.j, 2.+4.j])
       y = view_as_real(x)               # array([0., 2., 1., 3., 2., 4.])
       view_as_complex(y) == x           # True

    Notes
    -----
    Real-valued inputs are returned unchanged.  For complex-valued inputs, this function acts on the last axis as:

    .. math::

        y_{2n-1} = \mathcal{R}(x_n),
        \quad
        y_{2n} = \mathcal{I}(x_n),
        \quad 1\leq n\leq N,

    where :math:`\mathcal{R}, \mathcal{I}` denote the real/imaginary parts respectively.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_complex`,
    :py:func:`~pyxu.util.view_as_real_mat`,
    :py:func:`~pyxu.util.view_as_complex_mat`
    """
    if _is_real(x):
        return x

    try:
        c_dtype = x.dtype
        c_width = pxrt.CWidth(c_dtype)
        r_dtype = c_width.real.value
    except Exception:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    r_sh = (*x.shape[:-1], 2 * x.shape[-1])
    y = x.view(r_dtype).reshape(r_sh)
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


def view_as_real_mat(
    cmat: pxt.NDArray,
    real_input: bool = False,
    real_output: bool = False,
) -> pxt.NDArray:
    r"""
    View complex-valued matrix as its real-valued equivalent.  (Inverse of :py:func:`~pyxu.util.view_as_complex_mat`.)

    Useful to transform complex-valued matrix/vector products to their real-valued counterparts.

    Parameters
    ----------
    cmat: NDArray
        (M, N) complex-valued matrix.

    Returns
    -------
    rmat: NDArray
        The output shape depends on the values of `real_input` and `real_output`::

           | real_input | real_output | rmat.shape |
           |------------|-------------|------------|
           | False      | False       | (2M, 2N)   |
           | False      | True        | ( M, 2N)   |
           | True       | False       | (2M,  N)   |
           | True       | True        | ( M,  N)   |

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real_mat, view_as_complex_mat

       A = np.reshape(
           np.r_[:6] + 1j * np.r_[2:8], # array([[0.+2.j, 1.+3.j, 2.+4.j],
           newshape=(2, 3),             #        [3.+5.j, 4.+6.j, 5.+7.j]])
       )
       B = view_as_real_mat(A)          # array([[ 0., -2.,  1., -3.,  2., -4.],
                                        #        [ 2.,  0.,  3.,  1.,  4.,  2.],
                                        #        [ 3., -5.,  4., -6.,  5., -7.],
                                        #        [ 5.,  3.,  6.,  4.,  7.,  5.]])

    Notes
    -----
    * Real-valued matrices are returned unchanged.
    * Complex-valued matrices :math:`A\in\mathbb{C}^{M\times N}` are mapped into a real-valued matrix
      :math:`\hat{A}\in\mathbb{R}^{2M\times 2N}` defined, for :math:`1\leq n \leq N`, :math:`1\leq m\leq M` as

      .. math::

          \hat{A}_{2m-1,2n-1} = \mathcal{R}(A_{m,n}),
          & \quad
          \hat{A}_{2m-1,2n} = -\mathcal{I}(A_{m,n}),\\
          \hat{A}_{2m,2n-1} = \mathcal{I}(A_{m,n}),
          & \quad
          \hat{A}_{2m,2n}=\mathcal{R}(A_{m,n}).

      If ``real_[in|out]put=True``, then even columns/rows (or both) are furthermore dropped.  We have ``view_as_real(A
      @ x) = view_as_real_mat(A) @ view_as_real(x)``.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_real`,
    :py:func:`~pyxu.util.view_as_complex`,
    :py:func:`~pyxu.util.view_as_complex_mat`
    """
    assert cmat.ndim == 2, f"cmat: expected a 2D array, got {cmat.ndim}-D."
    if _is_real(cmat):
        return cmat

    try:
        c_dtype = cmat.dtype
        c_width = pxrt.CWidth(c_dtype)
        r_dtype = c_width.real.value
    except Exception:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    xp = pxd.NDArrayInfo.from_obj(cmat).module()
    rmat = xp.zeros((2 * cmat.shape[0], 2 * cmat.shape[1]), dtype=r_dtype)
    rmat[::2, ::2], rmat[1::2, 1::2] = cmat.real, cmat.real
    rmat[::2, 1::2], rmat[1::2, ::2] = -cmat.imag, cmat.imag
    if real_input:
        rmat = rmat[:, ::2]
    if real_output:
        rmat = rmat[::2, :]
    return rmat


def view_as_complex_mat(
    rmat: pxt.NDArray,
    real_input: bool = False,
    real_output: bool = False,
) -> pxt.NDArray:
    r"""
    View real-valued matrix as its complex-valued equivalent.  (Inverse of :py:func:`~pyxu.util.view_as_real_mat`.)

    Useful to transform real-valued matrix/vector products to their complex-valued counterparts.

    Parameters
    ----------
    rmat: NDArray
        Real-valued matrix.  Accepted dimensions depend on the values of `real_input` and `real_output`::

           | real_input | real_output | rmat.shape |
           |------------|-------------|------------|
           | False      | False       | (2M, 2N)   |
           | False      | True        | ( M, 2N)   |
           | True       | False       | (2M,  N)   |
           | True       | True        | ( M,  N)   |

    Returns
    -------
    cmat: NDArray
        (M, N) complex-valued matrix.

    Examples
    --------

    .. code-block:: python3

       from pyxu.util import view_as_real_mat, view_as_complex_mat

       A = np.array([[ 0., -2.,  1., -3.,  2., -4.],
                     [ 2.,  0.,  3.,  1.,  4.,  2.],
                     [ 3., -5.,  4., -6.,  5., -7.],
                     [ 5.,  3.,  6.,  4.,  7.,  5.]])

       B = view_as_complex(A)  # array([[0.+2.j, 1.+3.j, 2.+4.j],
                               #        [3.+5.j, 4.+6.j, 5.+7.j]])

    Notes
    -----
    * Complex-valued matrices are returned unchanged.
    * Real-valued matrices are mapped into a complex-valued matrix :math:`{A}\in\mathbb{C}^{M\times N}` as follows:

      * :math:`\hat{A}\in\mathbb{R}^{2M\times 2N}`:  :math:`A_{m,n} = \hat{A}_{2m-1,2n-1} + j \hat{A}_{2m,2n-1}`,
      * :math:`\hat{A}\in\mathbb{R}^{M\times 2N}`:  :math:`A_{m,n} = \hat{A}_{m,2n-1} - j \hat{A}_{m,2n}`,
      * :math:`\hat{A}\in\mathbb{R}^{2M\times N}`:  :math:`A_{m,n} = \hat{A}_{2m-1,n} + j \hat{A}_{2m,n}`,
      * :math:`\hat{A}\in\mathbb{R}^{M\times N}`:  :math:`A_{m,n} = \hat{A}_{m,n} + 0j`,

      for :math:`1\leq n \leq N`, :math:`1\leq m\leq M`.

    See Also
    --------
    :py:func:`~pyxu.util.view_as_real`,
    :py:func:`~pyxu.util.view_as_complex`,
    :py:func:`~pyxu.util.view_as_real_mat`
    """
    assert rmat.ndim == 2, f"rmat: expected a 2D array, got {rmat.ndim}-D."
    if _is_complex(rmat):
        return rmat

    try:
        r_dtype = rmat.dtype
        r_width = pxrt.Width(r_dtype)
        c_dtype = r_width.complex.value
    except Exception:
        raise ValueError(f"Unsupported dtype {r_dtype}.")

    N_row, N_col = rmat.shape
    error_msg = lambda _: f"{_} array dimension should be even-valued."
    if real_input and real_output:
        cmat = rmat.astype(c_dtype)
    elif real_input and (not real_output):
        assert N_row % 2 == 0, error_msg("First")
        cmat = rmat[::2, :] + (1j * rmat[1::2, :])
    elif (not real_input) and real_output:
        assert N_col % 2 == 0, error_msg("Last")
        cmat = rmat[:, ::2] - (1j * rmat[:, 1::2])
    else:  # (not real_input) and (not real_output)
        assert N_row % 2 == 0, error_msg("First")
        assert N_col % 2 == 0, error_msg("Last")
        cmat = rmat[::2, ::2] + (1j * rmat[1::2, ::2])
    return cmat
