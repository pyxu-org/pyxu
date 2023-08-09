import warnings

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "hutchpp",
    "norm",
]


def norm(x: pyct.NDArray, **kwargs):
    """
    Matrix or vector norm.

    This function is identical to :py:func:`numpy.linalg.norm`.
    It exists to correct bugs in Dask's implementation.

    Parameters
    ----------
    x: pyct.NDArray
        Input array.
    kwargs: dict
        Any kwarg accepted by :py:func:`numpy.linalg.norm`.

    Returns
    -------
    nrm: pyct.Real | pyct.NDArray
        Norm of the matrix or vector(s).
    """
    xp = pycu.get_array_module(x)
    nrm = xp.linalg.norm(x, **kwargs)
    nrm = nrm.astype(x.dtype, copy=False)  # dask bug: linalg.norm() always returns float64
    return nrm


def hutchpp(
    op: pyca.SquareOp,
    m: pyct.Integer = 4002,
    xp: pyct.ArrayModule = pycd.NDArrayInfo.NUMPY.module(),
    dtype: pyct.DType = None,
    seed: pyct.Integer = 0,
    enable_warnings: bool = True,
    **kwargs,
) -> pyct.Real:
    r"""
    Stochastic trace estimation of a linear operator based on the Hutch++ algorithm.
    (Specifically algorithm 3 from `this paper <https://arxiv.org/abs/2010.09649>`_.)

    Parameters
    ----------
    op: pyca.SquareOp
    m: pyct.Integer
        Number of queries used to estimate the trace of the linear operator.

        ``m`` is set to 4002 by default based on the analysis of the variance described in theorem 10.
        This default corresponds to having an estimation error smaller than 0.01 with probability 0.9.
    xp: pyct.ArrayModule
        Array module used for internal computations.
    dtype: pyct.DType
        Precision to use for internal computations.
        (Default: infer from runtime.)
    seed: pyct.Integer
        Seed for the random number generator.
    kwargs: dict
        Extra keyword arguments for (future) extensibility.

    Returns
    -------
    tr: pyct.Real
        Stochastic estimate of tr(op).
    """
    import pycsou.runtime as pycrt

    if dtype is None:
        dtype = pycrt.getPrecision().value
    width = pycrt.Width(dtype)

    with pycrt.Precision(width):
        if m >= op.dim:
            if enable_warnings:
                msg = "Number of queries >= dim(op): fallback to deterministic trace eval."
                warnings.warn(msg, pycuw.DenseWarning)
            tr = 0
            for i in range(op.dim):
                e = xp.zeros(op.dim, dtype=dtype)
                e[i] = 1
                tr += op.apply(e)[i]
        else:
            rng = xp.random.default_rng(seed=seed)
            s = rng.standard_normal(size=(op.dim, (m + 2) // 4), dtype=dtype)
            g = rng.integers(0, 2, size=(op.dim, (m - 2) // 2)) * 2 - 1

            data = op.apply(s.T).T
            kwargs = dict(mode="reduced")
            if xp == pycd.NDArrayInfo.DASK.module():
                data = data.rechunk({0: "auto", 1: -1})
                kwargs.pop("mode")

            q, _ = xp.linalg.qr(data, **kwargs)
            proj = g - q @ (q.T @ g)

            tr = (op.apply(q.T) @ q).trace()
            tr += (2 / (m - 2)) * (op.apply(proj.T) @ proj).trace()
    return float(tr)
