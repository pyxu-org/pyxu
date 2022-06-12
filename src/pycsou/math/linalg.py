import warnings

import dask.array as da
import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.linop.base as pycb
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
    if isinstance(op, pycb.ExplicitLinOp):
        op_xp = pycu.get_array_module(op.mat)
        if enable_warnings and (op_xp != xp):
            warnings.warn(
                f"Desired array module ({xp}) differs from array module of {op} ({op_xp}).",
                UserWarning,
            )

    if m >= op.dim:
        if enable_warnings:
            warnings.warn(
                "Number of queries >= dim(op): fallback to deterministic trace eval.",
                UserWarning,
            )
        tr = op.asarray(xp=xp).trace()
    else:
        rng = np.random.default_rng(seed=seed)
        s = xp.asarray(rng.standard_normal(size=(op.dim, (m + 2) // 4)))
        g = xp.asarray(rng.binomial(n=1, p=0.5, size=(op.dim, (m - 2) // 2)) * 2 - 1)

        data = op.apply(s.T).T
        kwargs = dict(mode="reduced")
        if xp == da:
            data = data.rechunk({0: "auto", 1: -1})
            kwargs.pop("mode")

        q, _ = xp.linalg.qr(data, **kwargs)
        proj = g - q @ (q.T @ g)

        tr = xp.trace(op.apply(q.T) @ q)
        tr += (2.0 / (m - 2)) * xp.trace(op.apply(proj.T) @ proj)
    return tr.item()
