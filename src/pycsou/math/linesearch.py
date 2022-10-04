import pycsou.abc as pyca
import pycsou.util.array_module as pyarr
import pycsou.util.ptype as pyct

LINESEARCH_DEFAULT_A_BAR = 1.0
LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 10e-4


def backtracking_linesearch(
    f: pyca.DiffFunc,
    x: pyct.NDArray,
    g: pyct.NDArray,
    p: pyct.NDArray,
    a_bar=LINESEARCH_DEFAULT_A_BAR,
    r=LINESEARCH_DEFAULT_R,
    c=LINESEARCH_DEFAULT_C,
):
    r"""
    Backtracking line search algorithm based on the Armijo-Goldstein condition.

    Follow `this link <https://www.wikiwand.com/en/Backtracking_line_search>`_ for reference of the algorithm and
    default values.

    **Parameterization**

    f: pyca.DiffFunc
        Differentiable functional
    x: pyct.NDArray
        (..., N) initial search position(s)
    g: pyct.NDArray
        (..., N) gradient of `f` at initial search position(s)
    p: pyct.NDArray
        (..., N) search direction(s) corresponding to the initial search position(s)
    a_bar: pyct.Real
        Initial step size, defaults to 1
    r: pyct.Real
        Step reduction factor, defaults to 0.5
    c: pyct:Real
        Bound reduction factor, defaults to 10e-4
    """

    arrmod = pyarr.get_array_module(x)

    def coeff_rows_multip(coeffs, rows):
        return arrmod.transpose(arrmod.transpose(rows) * coeffs)

    def correct_shape(v, default_v):
        return v if v not in [default_v, None] else arrmod.full(x.shape[:-1], default_v, dtype=x.dtype)

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

    while arrmod.any(cond):
        a = arrmod.where(cond, r * a, a)
        f_x_ap = f.apply(x + a * p)
        a_prod = coeff_rows_multip(a, scalar_prod)
        cond = f_x_ap > f_x + a_prod

    return a
