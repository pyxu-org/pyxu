import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct

LINESEARCH_DEFAULT_A_BAR = 1.0
LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 10e-4


def backtracking_linesearch(
    f: pyca.DiffFunc,
    x: pyct.NDArray,
    g: pyct.NDArray,
    p: pyct.NDArray,
    a_bar: pyct.Real = LINESEARCH_DEFAULT_A_BAR,
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
    g: pyct.NDArray
        (..., N) gradient of `f` at initial search point(s)
    p: pyct.NDArray
        (..., N) search direction(s) corresponding to initial point(s)
    a_bar: pyct.Real
        Initial step size.
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

    def correct_shape(v, default_v):
        return v if v not in [default_v, None] else xp.full(x.shape[:-1], default_v, dtype=x.dtype)

    def dot_prod_last_axis(v1, v2):
        return (v1 * v2).sum(axis=-1)

    a = correct_shape(a_bar, LINESEARCH_DEFAULT_A_BAR)
    r = correct_shape(r, LINESEARCH_DEFAULT_R)
    c = correct_shape(c, LINESEARCH_DEFAULT_C)

    f_x = f.apply(x)
    scalar_prod = c * dot_prod_last_axis(g, p)
    f_x_ap = f.apply(x + a_bar * p)
    a_prod = coeff_rows_multip(a, scalar_prod)
    cond = f_x_ap > f_x + a_prod

    while xp.any(cond):
        a = xp.where(cond, r * a, a)
        f_x_ap = f.apply(x + a * p)
        a_prod = coeff_rows_multip(a, scalar_prod)
        cond = f_x_ap > f_x + a_prod

    return a
