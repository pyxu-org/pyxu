import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "backtracking_linesearch",
]


LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 10e-4


def backtracking_linesearch(
    f: pyca.DiffFunc,
    x: pyct.NDArray,
    direction: pyct.NDArray,
    gradient: pyct.NDArray = None,
    a_bar: pyct.Real = None,
    r: pyct.Real = LINESEARCH_DEFAULT_R,
    c: pyct.Real = LINESEARCH_DEFAULT_C,
) -> pyct.NDArray:
    r"""
    Backtracking line search algorithm based on the
    `Armijo-Goldstein condition <https://www.wikiwand.com/en/Backtracking_line_search>`_.

    Parameters
    ----------
    f: pyca.DiffFunc
        Differentiable functional
    x: pyct.NDArray
        (..., N) initial search point(s)
    direction: pyct.NDArray
        (..., N) search direction(s) corresponding to initial point(s)
    gradient: pyct.NDArray
        (..., N) gradient of `f` at initial search point(s)

        Specifying `gradient` when known is an optimization:
        it will be autocomputed via ``f.grad(x)`` if unspecified.
    a_bar: pyct.Real
        Initial step size.

        If unspecified and :math:`\nabla f` is :math:`\beta`-Lipschitz continuous, then `a_bar` is
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

    xp = pycu.get_array_module(x)

    def coeff_rows_multip(coeffs, rows):
        return xp.transpose(xp.transpose(rows) * coeffs)

    def sanitize(v, default_v):
        return v if v not in [default_v, None] else default_v

    def correct_shape(v):
        return xp.full((*x.shape[:-1], 1), v, dtype=x.dtype)

    def dot_prod_last_axis(v1, v2):
        return (v1 * v2).sum(axis=-1)

    if a_bar is None:
        try:
            a_bar = pycrt.coerce(1 / f.diff_lipschitz())
        except ZeroDivisionError as exc:
            # f is linear -> line-search unbounded
            raise ValueError("Line-search does not converge for linear functionals.")

    if gradient is None:
        gradient = f.grad(x)

    a = correct_shape(a_bar)
    r = correct_shape(sanitize(r, LINESEARCH_DEFAULT_R))
    c = correct_shape(sanitize(c, LINESEARCH_DEFAULT_C))

    f_x = f.apply(x)
    scalar_prod = c * dot_prod_last_axis(gradient, direction)
    f_x_ap = f.apply(x + a_bar * direction)
    a_prod = coeff_rows_multip(a, scalar_prod)
    cond = f_x_ap > f_x + a_prod

    while xp.any(cond):
        a = xp.where(cond, r * a, a)
        f_x_ap = f.apply(x + a * direction)
        a_prod = coeff_rows_multip(a, scalar_prod)
        cond = f_x_ap > f_x + a_prod

    return a
