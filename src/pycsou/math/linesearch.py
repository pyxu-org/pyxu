import numpy as np

import pycsou.abc as pyca
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

    Performs a line search along the given direction(s).

    Parameters
    ----------
    f: pyca.DiffFunc
        Differentiable functional
    x: pyct.NDArray
        (..., N) initial search point(s)
    direction: pyct.NDArray
        (..., N) search direction(s) corresponding to initial point(s)
    gradient: pyct.NDArray
        (..., N) gradient of `f` at initial search point(s). Can be provided as
        well as left as None.
    a_bar: pyct.Real
        Initial step size. If left None, will use :math:`\frac{1}{L}` where
        L is a Lipschitz constant for :math:'\nabla f`
    r: pyct.Real
        Step reduction factor.
    c: pyct:Real
        Bound reduction factor.

    Returns
    -------
    a: pyct.NDArray
        (N,) step sizes.
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
        d_l = f.diff_lipschitz()
        if d_l is np.inf or d_l == 0:
            raise ValueError(
                "Either f gradient's lipschitz constant should be implemented through the diff_lipschitz"
                "method or a maximal step size a_bar should be given as an argument to this line search."
            )
        else:
            a_bar = 1.0 / f.diff_lipschitz()

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
