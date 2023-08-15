import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "backtracking_linesearch",
]


LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 1e-4


@pxrt.enforce_precision(
    i=("x", "direction", "gradient", "a0", "r", "c"),
    allow_None=True,
)
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
    Backtracking line search algorithm based on the
    `Armijo-Goldstein condition <https://www.wikiwand.com/en/Backtracking_line_search>`_.

    Parameters
    ----------
    f: pxa.DiffFunc
        Differentiable functional.
    x: pxt.NDArray
        (..., N) initial search point(s).
    direction: pxt.NDArray
        (..., N) search direction(s) corresponding to initial point(s).
    gradient: pxt.NDArray
        (..., N) gradient of `f` at initial search point(s).

        Specifying `gradient` when known is an optimization:
        it will be autocomputed via :py:meth:`~pyxu.abc.DiffFunc.grad` if unspecified.
    a0: pxt.Real
        Initial step size.

        If unspecified and :math:`\nabla f` is :math:`\beta`-Lipschitz continuous, then `a0` is
        auto-chosen as :math:`\frac{1}{\beta}`.
    r: pxt.Real
        Step reduction factor.
    c: pxt.Real
        Bound reduction factor.

    Returns
    -------
    a: pxt.NDArray
        (..., 1) step sizes.
    """
    assert 0 < r < 1
    assert 0 < c < 1
    if a0 is None:
        a0 = pxrt.coerce(1 / f.diff_lipschitz)
        assert a0 > 0, "a0: cannot auto-set step size."
    else:
        assert a0 > 0

    if gradient is None:
        gradient = f.grad(x)

    f_x = f.apply(x)
    d_f = c * (gradient * direction).sum(axis=-1, keepdims=True)  # \delta f

    def refine(a: pxt.NDArray) -> pxt.NDArray:
        # Do one iteration of the algorithm.
        #
        # Parameters
        # ----------
        # a : pxt.NDArray
        #     (..., 1) current step size(s)
        #
        # Returns
        # -------
        # mask : pxt.NDArray[bool]
        #     (..., 1) refinement points
        lhs = f.apply(x + a * direction)
        rhs = f_x + a * d_f
        return lhs > rhs  # mask

    xp = pxu.get_array_module(x)
    a = xp.full(shape=(*x.shape[:-1], 1), fill_value=a0, dtype=x.dtype)
    while (mask := refine(a)).any():
        a[mask] *= r
    return a
