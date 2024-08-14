import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt

__all__ = [
    "backtracking_linesearch",
]


LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 1e-4
newaxis = None  # Since we don't import NumPy


def backtracking_linesearch(
    f: pxa.DiffFunc,
    x: pxt.NDArray,
    direction: pxt.NDArray,
    gradient: pxt.NDArray = None,
    a0: pxt.Real = None,
    r: pxt.Real = LINESEARCH_DEFAULT_R,
    c: pxt.Real = LINESEARCH_DEFAULT_C,
) -> pxt.NDArray:
    r"""
    Backtracking line search algorithm based on the `Armijo-Goldstein condition
    <https://www.wikiwand.com/en/Backtracking_line_search>`_.

    Parameters
    ----------
    f: ~pyxu.abc.operator.DiffFunc
        Differentiable functional.
    x: NDArray
        (..., M1,...,MD) initial search point(s).
    direction: NDArray
        (..., M1,...,MD) search direction(s) corresponding to initial point(s).
    gradient: NDArray
        (..., M1,...,MD) gradient of `f` at initial search point(s).

        Specifying `gradient` when known is an optimization: it will be autocomputed via
        :py:meth:`~pyxu.abc.DiffFunc.grad` if unspecified.
    a0: Real
        Initial step size.

        If unspecified and :math:`\nabla f` is :math:`\beta`-Lipschitz continuous, then `a0` is auto-chosen as
        :math:`\frac{1}{\beta}`.
    r: Real
        Step reduction factor.
    c: Real
        Bound reduction factor.

    Returns
    -------
    a: NDArray
        (..., 1) step sizes.

    Notes
    -----
    * Performing a line-search with DASK inputs is inefficient due to iterative nature of algorithm.
    """
    ndi = pxd.NDArrayInfo.from_obj(x)
    xp = ndi.module()

    assert 0 < r < 1
    assert 0 < c < 1
    if a0 is None:
        a0 = 1.0 / f.diff_lipschitz
        assert a0 > 0, "a0: cannot auto-set step size."
    else:
        assert a0 > 0

    if gradient is None:
        gradient = f.grad(x)

    f_x = f.apply(x)  # (..., 1)
    d_f = (  # \delta f  (..., 1)
        c
        * xp.sum(
            gradient * direction,
            axis=tuple(range(-f.dim_rank, 0)),
        )[..., newaxis]
    )

    def refine(a: pxt.NDArray) -> pxt.NDArray:
        # Do one iteration of the algorithm.
        #
        # Parameters
        # ----------
        # a : NDArray
        #     (..., 1) current step size(s).
        #
        # Returns
        # -------
        # mask : NDArray[bool]
        #     (..., 1) refinement points
        a_D = a[..., *((newaxis,) * (f.dim_rank - 1))]  # (..., 1,...,1)
        lhs = f.apply(x + a_D * direction)  # (..., 1)
        rhs = f_x + a * d_f  # (..., 1)
        return lhs > rhs  # mask

    a = xp.full_like(d_f, fill_value=a0, dtype=x.dtype)
    while xp.any(mask := refine(a)):
        a[mask] *= r

        if ndi == pxd.NDArrayInfo.DASK:
            a.compute_chunk_sizes()
    return a
