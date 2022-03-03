import warnings

import pycsou.runtime as pycrt
import pycsou.util as pycu


@pycrt.enforce_precision(o=True)
def hutchpp(linop, m: int = 4002, gpu: bool = False):
    r"""
    Computes an stocastic estimate of the trace for linear operators based on the Hutch++ algorithm (specifically,
    algorithm 3 of the paper https://arxiv.org/abs/2010.09649.

    Parameters
    ----------
    linop: :py:class:`~pycsou.abc.operator.LinOp`
        Linear operator object compliant with Pycsou's interface with square shape.
    m: int
        The number of queries desired to estimate the trace of the linear operator. ``m`` is set to 4002 by default,
        based on the analysis of variance described in theorem 10 of the Hutch++ paper. This default number of queries
        corresponds to having an estimation error smaller than 0.01 with a probability of 0.9.

    Returns
    -------
    float
        Hutch++ stochastic estimate of the trace.

    Notes
    -----
    This function calls Scipy’s function: :py:func:`scipy.sparse.linalg.qr`. See the documentation of this function
    for more information on its behaviour and the underlying LAPACK routines it relies on.
    """
    if (
        pycu.deps.CUPY_ENABLED and gpu
    ):  # Scipy casts any input to the LinOp as a Numpy array so the cupyx version is needed.
        import cupy as xp
        from cupy.linalg import qr

        mode = "reduced"
    else:
        import numpy as xp
        from scipy.linalg import qr

        mode = "economic"
    d = linop.shape[1]
    if m >= d:
        warnings.warn(
            "Full trace computation performed. Stochastic trace estimation not performed because the number "
            "of queries is larger or equal to the dimension of the linear operator.",
            UserWarning,
        )
        return xp.trace(linop.apply(xp.eye(d)))

    s = xp.random.randn(d, (m + 2) // 4)
    g = xp.random.randn(d, (m - 2) // 2)
    q, _ = qr(linop.apply(s.T).T, mode=mode)
    proj = g - q @ (q.T @ g)
    return xp.trace(q.T @ linop.apply(q.T).T) + (2.0 / (m - 2)) * xp.trace(proj.T @ linop.apply(proj.T).T)
