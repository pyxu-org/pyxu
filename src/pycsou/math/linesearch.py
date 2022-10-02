import pycsou.abc as pyca
import pycsou.util.ptype as pyct

LINESEARCH_DEFAULT_A_BAR = 1.0
LINESEARCH_DEFAULT_R = 0.5
LINESEARCH_DEFAULT_C = 10e-4


def backtracking_linesearch(
    f: pyca.DiffFunc,
    x: pyct.NDArray,
    g_f_x: pyct.NDArray,
    p: pyct.NDArray,
    a_bar: pyct.Real = LINESEARCH_DEFAULT_A_BAR,
    r: pyct.Real = LINESEARCH_DEFAULT_R,
    c: pyct.Real = LINESEARCH_DEFAULT_C,
) -> pyct.Real:
    r"""
    Backtracking line search algorithm based on the Armijo-Goldstein condition.

    Follow `this link <https://www.wikiwand.com/en/Backtracking_line_search>`_ for reference of the algorithm and
    default values.

    **Parameterization**

    f: pyca.DiffFunc
        Differentiable functional
    x: pyct.NDArray
        (N,) initial search position
    g_f_x: pyct.NDArray
        (N,) gradient of `f` at initial search position(s)
    p: pyct.NDArray
        (N,) search direction(s)
    a_bar: pyct.Real
        Initial step size, defaults to 1
    r: pyct.Real
        Step reduction factor, defaults to 0.5
    c: pyct:Real
        Bound reduction factor, defaults to 10e-4
    """

    def default_if_none(v, default_v):
        return v if v is not None else default_v

    a = default_if_none(a_bar, LINESEARCH_DEFAULT_A_BAR)
    r = default_if_none(r, LINESEARCH_DEFAULT_R)
    c = default_if_none(c, LINESEARCH_DEFAULT_C)

    f_x = f.apply(x)
    scalar_prod = c * (p @ g_f_x)

    while f.apply(x + a * p) > f_x + a * scalar_prod:
        a = r * a

    return a
