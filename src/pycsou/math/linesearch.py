import pycsou.abc as pyca
import pycsou.util.ptype as pyct

LINESEARCH_DEFAULT_A_BAR = 1
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

    # document where default values come from (continuous opti course)

    def default_if_none(v, default_v):
        return v if v is not None else default_v

    a = default_if_none(a_bar, LINESEARCH_DEFAULT_A_BAR)
    r = default_if_none(r, LINESEARCH_DEFAULT_R)
    c = default_if_none(c, LINESEARCH_DEFAULT_C)

    f_x = f.apply(x)
    scalar_prod = c * (g_f_x @ p)

    while f.apply(x + a * p) > f_x + a * scalar_prod:
        a = r * a

    return a
