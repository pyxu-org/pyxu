import pycsou.abc as pyca
import pycsou.info.ptype as pyct
import pycsou.runtime as pycrt
import pycsou.util as pycu

__all__ = [
    "backtracking_linesearch",
]


LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 1e-4


@pycrt.enforce_precision(
    i=("x", "direction", "gradient", "a0", "r", "c"),
    allow_None=True,
)
def backtracking_linesearch(
    f: pyca.DiffFunc,
    x: pyct.NDArray,
    direction: pyct.NDArray,
    gradient: pyct.NDArray = None,
    a0: pyct.Real = None,
    r: pyct.Real = LINESEARCH_DEFAULT_R,
    c: pyct.Real = LINESEARCH_DEFAULT_C,
) -> pyct.NDArray:
    r"""
    Backtracking line search algorithm based on the
    `Armijo-Goldstein condition <https://www.wikiwand.com/en/Backtracking_line_search>`_.

    Parameters
    ----------
    f: pyca.DiffFunc
        Differentiable functional.
    x: pyct.NDArray
        (..., N) initial search point(s).
    direction: pyct.NDArray
        (..., N) search direction(s) corresponding to initial point(s).
    gradient: pyct.NDArray
        (..., N) gradient of `f` at initial search point(s).

        Specifying `gradient` when known is an optimization:
        it will be autocomputed via ``f.grad(x)`` if unspecified.
    a0: pyct.Real
        Initial step size.

        If unspecified and :math:`\nabla f` is :math:`\beta`-Lipschitz continuous, then `a0` is
        auto-chosen as :math:`\frac{1}{\beta}`.
    r: pyct.Real
        Step reduction factor.
    c: pyct:Real
        Bound reduction factor.

    Returns
    -------
    a: pyct.NDArray
        (..., 1) step sizes.
    """
    assert 0 < r < 1
    assert 0 < c < 1
    if a0 is None:
        try:
            a0 = pycrt.coerce(1 / f.diff_lipschitz())
            assert a0 > 0, "a0: cannot auto-set step size."
        except ZeroDivisionError:
            # f is linear -> line-search unbounded
            raise ValueError("Line-search does not converge for linear functionals.")
    else:
        assert a0 > 0

    if gradient is None:
        gradient = f.grad(x)

    f_x = f.apply(x)
    d_f = c * (gradient * direction).sum(axis=-1, keepdims=True)  # \delta f

    def refine(a: pyct.NDArray) -> pyct.NDArray:
        # Do one iteration of the algorithm.
        #
        # Parameters
        # ----------
        # a : pyct.NDArray
        #     (..., 1) current step size(s)
        #
        # Returns
        # -------
        # mask : pyct.NDArray[bool]
        #     (..., 1) refinement points
        lhs = f.apply(x + a * direction)
        rhs = f_x + a * d_f
        return lhs > rhs  # mask

    xp = pycu.get_array_module(x)
    a = xp.full(shape=(*x.shape[:-1], 1), fill_value=a0, dtype=x.dtype)
    while (mask := refine(a)).any():
        a[mask] *= r
    return a
