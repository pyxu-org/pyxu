import typing as typ

import dask.array as da

import pycsou.runtime as pycrt
import pycsou.util.array_module as pyca
import pycsou.util.ptype as pyct

__all__ = [
    "view_as_real",
    "view_as_complex",
    "view_as_real_mat",
    "view_as_complex_mat",
]


def view_as_complex(x: pyct.NDArray) -> pyct.NDArray:
    r"""
    View real-valued array as its complex-valued bijection (inverse of :py:func:`~pycsou.util.complex.view_as_real`).

    Parameters
    ----------
    x: pyct.NDArray
        (..., 2N) real-valued array.

    Returns
    -------
    y: pyct.NDArray
        (..., N) complex-valued array.

    Examples
    --------
    >>> from pycsou.util import view_as_real, view_as_complex
    >>> x = np.arange(6).astype(float)
    array([0., 1., 2., 3., 4., 5.])
    >>> y = view_as_complex(x)
    array([0.+1.j, 2.+3.j, 4.+5.j])
    >>> view_as_real(y) == x
    True

    Notes
    -----
    Complex-valued input arrays are returned unchanged. For real-valued inputs, this function acts on the last axis as:

    .. math::

        y_n = x_{2n-1}+jx_{2n}, \qquad 1\leq n\leq N.

    See Also
    --------
    view_as_real, view_as_real_mat, view_as_complex_mat
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
    r"""
    View complex-valued array as its real-valued bijection (inverse of :py:func:`~pycsou.util.complex.view_as_complex`).

    Parameters
    ----------
    x: pyct.NDArray
        (..., N) complex-valued array

    Returns
    -------
    y: pyct.NDArray
        (..., 2N) real-valued array.

    Examples
    --------
    >>> from pycsou.util import view_as_real, view_as_complex
    >>> x = np.arange(3) + 1j * (np.arange(3) + 2)
    array([0.+2.j, 1.+3.j, 2.+4.j])
    >>> y = view_as_real(x)
    array([0., 2., 1., 3., 2., 4.])
    >>> view_as_complex(y) == x
    True

    Notes
    -----
    Real-valued array inputs are returned unchanged. For complex-valued inputs, this function acts on the last axis as:

    .. math::

        y_{2n-1}=\mathcal{R}(x_n),\quad  y_{2n}=\mathcal{I}(x_n), \quad 1\leq n\leq N,

    where :math:`\mathcal{R}, \mathcal{I}` denote the real/imaginary parts respectively.

    See Also
    --------
    view_as_real_mat, view_as_complex, view_as_complex_mat
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


def view_as_real_mat(cmat: pyct.NDArray, real_input: bool = False, real_output: bool = False) -> pyct.NDArray:
    r"""
    View complex-valued matrix as its real-valued equivalent (inverse of :py:func:`~pycsou.util.complex.view_as_complex_mat`).

    Useful to transform complex-valued matrix/vector products in their real counterparts.

    Parameters
    ----------
    cmat: NDArray
        (M, N) complex-valued matrix.

    Returns
    -------
    rmat: NDArray
        * (2M, 2N) real-valued matrix if ``real_input=real_output=False``,
        * (M, 2N) real-valued matrix if ``real_output=True`` and ``real_input=False``,
        * (2M, N) real-valued matrix if ``real_output=False`` and ``real_input=True``,
        * (M, N) real-valued matrix if ``real_input=real_output=True``.

    Examples
    --------
    >>> from pycsou.util import view_as_real_mat, view_as_complex_mat
    >>> A = (np.arange(6) + 1j * (np.arange(6)+2)).reshape(2,3)
    array([[0.+2.j, 1.+3.j, 2.+4.j],
           [3.+5.j, 4.+6.j, 5.+7.j]])
    >>> B = view_as_real_mat(A)
    array([[ 0., -2.,  1., -3.,  2., -4.],
           [ 2.,  0.,  3.,  1.,  4.,  2.],
           [ 3., -5.,  4., -6.,  5., -7.],
           [ 5.,  3.,  6.,  4.,  7.,  5.]])
    >>> view_as_complex_mat(B) == A
    True

    Notes
    -----
    Real-valued matrices are returned unchanged. Complex-valued matrices :math:`A\in\mathbb{C}^{M\times N}` are mapped into a
    real-valued matrix :math:`\hat{A}\in\mathbb{R}^{2M\times 2N}` defined, for :math:`1\leq n \leq N`, :math:`1\leq m\leq M` as

    .. math::

        \hat{A}_{2m-1,2n-1}=\mathcal{R}(A_{m,n}),& \quad \hat{A}_{2m-1,2n}=-\mathcal{I}(A_{m,n}),\\
        \hat{A}_{2m,2n-1}=\mathcal{I}(A_{m,n}), & \quad \hat{A}_{2m,2n}=\mathcal{R}(A_{m,n}).

    If ``real_input=True`` or ``real_output=True``, even rows/columns (or both) are furthermore dropped.
    We have ``view_as_real(A @ x)=view_as_real_mat(A) @ view_as_real(x)``.

    See Also
    --------
    view_as_real, view_as_complex, view_as_complex_mat

    """
    if _is_real(cmat):
        return cmat

    try:
        c_dtype = cmat.dtype
        c_width = pycrt._CWidth(c_dtype)
        r_dtype = c_width.real.value
    except:
        raise ValueError(f"Unsupported dtype {c_dtype}.")

    xp = pyca.get_array_module(cmat)
    rmat = xp.zeros((2 * cmat.shape[0], 2 * cmat.shape[1]), dtype=r_dtype)
    rsh = rmat.shape
    rmat = rmat.ravel()
    ri, rj, rrij, crij, ccij, rcij = _rc_masks(cmat, xp, c_dtype, r_dtype)
    rmat[rrij.ravel()] = cmat.real.ravel()
    rmat[crij.ravel()] = cmat.imag.ravel()
    rmat[ccij.ravel()] = cmat.real.ravel()
    rmat[rcij.ravel()] = -cmat.imag.ravel()
    rmat = rmat.reshape(rsh)
    if real_input:
        rmat = rmat[:, rj]
    if real_output:
        rmat = rmat[ri, :]

    return rmat


def _rc_masks(cmat, xp, c_dtype, r_dtype) -> typ.Tuple:
    ri, ci = xp.ones(cmat.shape[0], dtype=c_dtype).view(r_dtype).astype(bool), (
        1j * xp.ones(cmat.shape[0], dtype=c_dtype)
    ).view(r_dtype).astype(bool)
    rj, cj = xp.ones(cmat.shape[1], dtype=c_dtype).view(r_dtype).astype(bool), (
        1j * xp.ones(cmat.shape[1], dtype=c_dtype)
    ).view(r_dtype).astype(bool)
    rrij = (ri[:, None] * rj[None, :]).astype(bool)
    crij = (ci[:, None] * rj[None, :]).astype(bool)
    ccij = (ci[:, None] * cj[None, :]).astype(bool)
    rcij = (ri[:, None] * cj[None, :]).astype(bool)
    return pyca.compute(ri, rj, rrij, crij, ccij, rcij)


def view_as_complex_mat(rmat: pyct.NDArray, real_input: bool = False, real_output: bool = False):
    r"""
    View real-valued matrix as its complex-valued equivalent (inverse of :py:func:`~pycsou.util.complex.view_as_real_mat`).

    Useful to transform real-valued matrix/vector products in their complex counterparts.

    Parameters
    ----------
    rmat: NDArray
        * (2M, 2N) real-valued matrix if ``real_input=real_output=False``,
        * (M, 2N) real-valued matrix if ``real_output=True`` and ``real_input=False``,
        * (2M, N) real-valued matrix if ``real_output=False`` and ``real_input=True``,
        * (M, N) real-valued matrix if ``real_input=real_output=True``.

    Returns
    -------
    cmat: NDArray
        (M, N) complex-valued matrix.

    Examples
    --------
    >>> from pycsou.util import view_as_real_mat, view_as_complex_mat
    >>> A = (np.arange(6) + 1j * (np.arange(6)+2)).reshape(2,3)
    array([[0.+2.j, 1.+3.j, 2.+4.j],
           [3.+5.j, 4.+6.j, 5.+7.j]])
    >>> B = view_as_real_mat(A)
    array([[ 0., -2.,  1., -3.,  2., -4.],
           [ 2.,  0.,  3.,  1.,  4.,  2.],
           [ 3., -5.,  4., -6.,  5., -7.],
           [ 5.,  3.,  6.,  4.,  7.,  5.]])
    >>> C = view_as_complex_mat(B)
    array([[0.+2.j, 1.+3.j, 2.+4.j],
           [3.+5.j, 4.+6.j, 5.+7.j]])

    Notes
    -----
    Complex-valued matrices are returned unchanged. Real-valued matrices are mapped into a
    complex-valued matrix :math:`{A}\in\mathbb{C}^{M\times N}` as follows:

        * :math:`\hat{A}\in\mathbb{R}^{2M\times 2N}`:  :math:`A_{m,n} = \hat{A}_{2m-1,2n-1} + j \hat{A}_{2m,2n-1}`,
        * :math:`\hat{A}\in\mathbb{R}^{M\times 2N}`:  :math:`A_{m,n} = \hat{A}_{m,2n-1} - j \hat{A}_{m,2n}`,
        * :math:`\hat{A}\in\mathbb{R}^{2M\times N}`:  :math:`A_{m,n} = \hat{A}_{2m-1,n} + j \hat{A}_{2m,n}`,
        * :math:`\hat{A}\in\mathbb{R}^{M\times N}`:  :math:`A_{m,n} = \hat{A}_{m,n} + 0j`,

    for :math:`1\leq n \leq N`, :math:`1\leq m\leq M`.

    See Also
    --------
    view_as_real, view_as_complex, view_as_real_mat

    """
    if _is_complex(rmat):
        return rmat

    try:
        r_dtype = rmat.dtype
        r_width = pycrt.Width(r_dtype)
        c_dtype = r_width.complex.value
    except:
        raise ValueError(f"Unsupported dtype {r_dtype}.")

    xp = pyca.get_array_module(rmat)
    if real_input and real_output:
        return (rmat + 0j).astype(c_dtype)
    else:
        if real_input and not real_output:
            assert rmat.shape[0] % 2 == 0, "First array dimension should be even-valued."
            cmat = xp.zeros((rmat.shape[0] // 2, rmat.shape[1]), dtype=c_dtype)
            _, rj, rrij, crij, _, _ = _rc_masks(cmat, xp, c_dtype, r_dtype)
            rrij, crij = rrij[:, rj], crij[:, rj]
            rpart = rmat[rrij]
            ipart = rmat[crij]
        elif not real_input and real_output:
            assert rmat.shape[1] % 2 == 0, "Last array dimension should be even-valued."
            cmat = xp.zeros((rmat.shape[0], rmat.shape[1] // 2), dtype=c_dtype)
            ri, _, rrij, _, _, rcij = _rc_masks(cmat, xp, c_dtype, r_dtype)
            rrij, rcij = rrij[ri, :], rcij[ri, :]
            rpart = rmat[rrij]
            ipart = -rmat[rcij]
        else:
            assert rmat.shape[0] % 2 == 0 and rmat.shape[1] % 2 == 0, "Both array dimensions should be even-valued."
            cmat = xp.zeros((rmat.shape[0] // 2, rmat.shape[1] // 2), dtype=c_dtype)
            _, _, rrij, crij, _, _ = _rc_masks(cmat, xp, c_dtype, r_dtype)
            rpart = rmat[rrij]
            ipart = rmat[crij]
        if isinstance(rpart, da.Array):
            rpart.compute_chunk_sizes()
            ipart.compute_chunk_sizes()
        cmat = (rpart + 1j * ipart).reshape(cmat.shape).astype(c_dtype)
        return cmat
