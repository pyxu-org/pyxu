import warnings

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.linop.base as pycb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


def hutchpp(
    op: pyco.SquareOp,
    m: int = 4002,
    xp: pyct.ArrayModule = np,
    seed: float = 0,
    enable_warnings: bool = True,
) -> float:
    r"""
    Stochastic estimate of the trace of a linear operator based on the Hutch++ algorithm.
    (Specifically algorithm 3 of the paper https://arxiv.org/abs/2010.09649)

    Parameters
    ----------
    op: :py:class:`~pycsou.abc.operator.SquareOp`
    m: int
        Number of queries used to estimate the trace of the linear operator.

        ``m`` is set to 4002 by default based on the analysis of the variance described in theorem
        10. This default corresponds to having an estimation error smaller than 0.01 with a
        probability of 0.9.
    xp: pycsou.util.ptype.ArrayModule
        Array module used for internal computations.
    seed: int
        Seed for the random number generator.

    Returns
    -------
    tr: float
        Stochastic estimate of tr(op).
    """

    if linop.shape[0] != linop.shape[1]:
        raise NotImplementedError

    import dask.array as da

    xlin = xp.linalg
    if xp == da:
        kwargs = {}
    else:
        kwargs = {"mode": "reduced"}

    d = linop.shape[1]
    if m >= d:
        warnings.warn(
            "Full trace computation performed. Stochastic trace estimation not performed because the number "
            "of queries is larger or equal to the dimension of the linear operator.",
            UserWarning,
        )
        return xp.sum(xp.array([linop.apply(e)[i] for i, e in enumerate(xp.eye(d))]))

    if isinstance(linop, pycb.ExplicitLinOp):
        if xp != pycu.get_array_module(linop.mat):
            warnings.warn(
                f"The array module of the :py:class:`~pycsou.linop.base.ExplicitLinOp` "
                f"({pycu.get_array_module(linop.mat)}) and the requested array module "
                f"({xp}) are different.",
                UserWarning,
            )

    rng = np.random.default_rng(seed=seed)
    s = xp.asarray(rng.standard_normal(size=(d, (m + 2) // 4)))
    g = xp.asarray(rng.binomial(n=1, p=0.5, size=(d, (m - 2) // 2)) * 2 - 1)
    if xp == da:
        q, _ = xlin.qr(linop.apply(s.T).T.rechunk({0: "auto", 1: -1}), **kwargs)
    else:
        q, _ = xlin.qr(linop.apply(s.T).T, **kwargs)

    proj = g - q @ (q.T @ g)
    return xp.trace(q.T @ linop.apply(q.T).T) + (2.0 / (m - 2)) * xp.trace(proj.T @ linop.apply(proj.T).T)
