import pycsou.abc as pyca
import pycsou.info.deps as pycd
import pycsou.info.ptype as pyct
import pycsou.runtime as pycrt
import pycsou.util as pycu

__all__ = [
    "hutchpp",
    "norm",
    "trace",
]


def norm(x: pyct.NDArray, **kwargs):
    # This function is identical to :py:func:`numpy.linalg.norm`.
    # It exists to correct bugs in Dask's implementation.
    xp = pycu.get_array_module(x)
    nrm = xp.linalg.norm(x, **kwargs)
    nrm = nrm.astype(x.dtype, copy=False)  # dask bug: linalg.norm() always returns float64
    return nrm


def trace(
    op: pyca.SquareOp,
    xp: pyct.ArrayModule = None,
    dtype: pyct.DType = None,
) -> pyct.Real:
    r"""
    Exact trace of a linear operator based on multiple evaluation of the forward operator.

    Parameters
    ----------
    op: pyca.SquareOp
    xp: pyct.ArrayModule
        Array module used for internal computations. (Default: NumPy.)
    dtype: pyct.DType
        Precision to use for internal computations. (Default: current runtime precision.)

    Returns
    -------
    tr: pyct.Real
        Exact value of tr(op).
    """
    if xp is None:
        xp = pycd.NDArrayInfo.default().module()

    if dtype is None:
        dtype = pycrt.getPrecision().value
    width = pycrt.Width(dtype)

    tr = 0
    for i in range(op.dim):
        e = xp.zeros(op.dim, dtype=dtype)
        e[i] = 1
        with pycrt.Precision(width):
            tr += op.apply(e)[i]
    return float(tr)


def hutchpp(
    op: pyca.SquareOp,
    m: pyct.Integer = 4002,
    xp: pyct.ArrayModule = None,
    dtype: pyct.DType = None,
    seed: pyct.Integer = None,
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
        Array module used for internal computations. (Default: NumPy.)
    dtype: pyct.DType
        Precision to use for internal computations. (Default: current runtime precision.)
    seed: pyct.Integer
        Seed for the random number generator.

    Returns
    -------
    tr: pyct.Real
        Stochastic estimate of tr(op).
    """
    if xp is None:
        xp = pycd.NDArrayInfo.default().module()

    if dtype is None:
        dtype = pycrt.getPrecision().value
    width = pycrt.Width(dtype)

    rng = xp.random.default_rng(seed=seed)
    s = rng.standard_normal(size=(op.dim, (m + 2) // 4), dtype=dtype)
    g = rng.integers(0, 2, size=(op.dim, (m - 2) // 2)) * 2 - 1

    with pycrt.Precision(width):
        data = op.apply(s.T).T

    kwargs = dict(mode="reduced")
    if xp == pycd.NDArrayInfo.DASK.module():
        data = data.rechunk({0: "auto", 1: -1})
        kwargs.pop("mode", None)

    q, _ = xp.linalg.qr(data, **kwargs)
    proj = g - q @ (q.T @ g)

    tr = (op.apply(q.T) @ q).trace()
    tr += (2 / (m - 2)) * (op.apply(proj.T) @ proj).trace()
    return float(tr)
