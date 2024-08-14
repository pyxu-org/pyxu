import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt

__all__ = [
    "hutchpp",
    "trace",
]


def trace(
    op: pxa.SquareOp,
    xp: pxt.ArrayModule = None,
    dtype: pxt.DType = None,
) -> pxt.Real:
    r"""
    Exact trace of a linear operator based on multiple evaluation of the forward operator.

    Parameters
    ----------
    op: ~pyxu.abc.operator.SquareOp
    xp: ArrayModule
        Array module used for internal computations. (Default: NumPy.)
    dtype: DType
        Precision to use for internal computations. (Default: current runtime precision.)

    Returns
    -------
    tr: Real
        Exact value of tr(op).
    """
    if xp is None:
        xp = pxd.NDArrayInfo.default().module()

    if dtype is None:
        dtype = pxrt.Width.DOUBLE.value

    tr = 0
    for i in range(op.dim_size):
        idx_in = np.unravel_index(i, op.dim_shape)
        idx_out = np.unravel_index(i, op.codim_shape)

        e = xp.zeros(op.dim_shape, dtype=dtype)
        e[idx_in] = 1
        tr += op.apply(e)[idx_out]
    return float(tr)


def hutchpp(
    op: pxa.SquareOp,
    m: pxt.Integer = 4002,
    xp: pxt.ArrayModule = None,
    dtype: pxt.DType = None,
    seed: pxt.Integer = None,
) -> pxt.Real:
    r"""
    Stochastic trace estimation of a linear operator based on the Hutch++ algorithm.  (Specifically `algorithm 3 from
    this paper <https://arxiv.org/abs/2010.09649>`_.)

    Parameters
    ----------
    op: ~pyxu.abc.operator.SquareOp
    m: Integer
        Number of queries used to estimate the trace of the linear operator.

        `m` is set to 4002 by default based on the analysis of the variance described in theorem 10.  This default
        corresponds to having an estimation error smaller than 0.01 with probability 0.9.
    xp: ArrayModule
        Array module used for internal computations. (Default: NumPy.)
    dtype: DType
        Precision to use for internal computations. (Default: current runtime precision.)
    seed: Integer
        Seed for the random number generator.

    Returns
    -------
    tr: Real
        Stochastic estimate of tr(op).
    """
    from pyxu.operator import ReshapeAxes

    if xp is None:
        xp = pxd.NDArrayInfo.default().module()
    if using_dask := (xp == pxd.NDArrayInfo.DASK.module()):
        msg = "\n".join(
            [
                "DASK.linalg.qr() has limitations.",
                "[More info] https://docs.dask.org/en/stable/_modules/dask/array/linalg.html#qr",
            ]
        )
        pxw.warn_dask_perf(msg)

    if dtype is None:
        dtype = pxrt.Width.DOUBLE.value

    # To avoid constant reshaping below, we use the 2D-equivalent operator.
    lhs = ReshapeAxes(dim_shape=op.codim_shape, codim_shape=op.codim_size)
    rhs = ReshapeAxes(dim_shape=op.dim_size, codim_shape=op.dim_shape)
    op = lhs * op * rhs

    rng = xp.random.default_rng(seed=seed)
    s = rng.standard_normal(size=(op.dim_size, (m + 2) // 4), dtype=dtype)
    g = rng.integers(0, 2, size=(op.dim_size, (m - 2) // 2)) * 2 - 1

    data = op.apply(s.T).T  # (dim, (m+2)//4)

    kwargs = dict(mode="reduced")
    if using_dask:
        data = data.rechunk({0: "auto", 1: -1})
        kwargs.pop("mode", None)

    q, _ = xp.linalg.qr(data, **kwargs)
    proj = g - q @ (q.T @ g)

    tr = (op.apply(q.T) @ q).trace()
    tr += (2 / (m - 2)) * (op.apply(proj.T) @ proj).trace()
    return float(tr)
