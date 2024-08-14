import warnings

import scipy.sparse.linalg as spsl

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "from_sciop",
    "to_sciop",
]


def from_sciop(cls: pxt.OpC, sp_op: spsl.LinearOperator) -> pxt.OpT:
    r"""
    Wrap a :py:class:`~scipy.sparse.linalg.LinearOperator` as a 2D :py:class:`~pyxu.abc.LinOp` (or sub-class thereof).

    Parameters
    ----------
    sp_op: ~scipy.sparse.linalg.LinearOperator
        (N, M) Linear CPU/GPU operator compliant with SciPy's interface.

    Returns
    -------
    op: OpT
        Pyxu-compliant linear operator with:

        * dim_shape: (M,)
        * codim_shape: (N,)
    """
    assert cls.has(pxa.Property.LINEAR)

    if sp_op.dtype not in [_.value for _ in pxrt.Width]:
        warnings.warn(
            "Computation may not be performed at the requested precision.",
            pxw.PrecisionWarning,
        )

    # [r]matmat only accepts 2D inputs -> reshape apply|adjoint inputs as needed.

    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.dim_size)
        out = _._sp_op.matmat(arr.T).T
        out = out.reshape(*sh, _.codim_size)
        return out

    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.codim_size)
        out = _._sp_op.rmatmat(arr.T).T
        out = out.reshape(*sh, _.dim_size)
        return out

    def op_asarray(_, **kwargs) -> pxt.NDArray:
        # Determine XP-module accepted by sci_op, then compute array-representation.
        for ndi in [
            pxd.NDArrayInfo.NUMPY,
            pxd.NDArrayInfo.CUPY,
        ]:
            try:
                cls = _.__class__
                _A = cls.asarray(_, xp=ndi.module(), dtype=_._sp_op.dtype)
                break
            except Exception:
                pass

        # Cast to user specs.
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        A = xp.array(pxu.to_NUMPY(_A), dtype=dtype)
        return A

    def op_expr(_) -> tuple:
        return ("from_sciop", _._sp_op)

    op = px_src.from_source(
        cls=cls,
        dim_shape=sp_op.shape[1],
        codim_shape=sp_op.shape[0],
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        _expr=op_expr,
    )
    op._sp_op = sp_op

    return op


def to_sciop(
    op: pxt.OpT,
    dtype: pxt.DType = None,
    gpu: bool = False,
) -> spsl.LinearOperator:
    r"""
    Cast a :py:class:`~pyxu.abc.LinOp` to a CPU/GPU :py:class:`~scipy.sparse.linalg.LinearOperator`, compatible with
    the matrix-free linear algebra routines of :py:mod:`scipy.sparse.linalg`.

    Parameters
    ----------
    dtype: DType
        Working precision of the linear operator.
    gpu: bool
        Operate on CuPy inputs (True) vs. NumPy inputs (False).

    Returns
    -------
    op: ~scipy.sparse.linalg.LinearOperator
        Linear operator object compliant with SciPy's interface.
    """
    if not (op.dim_rank == op.codim_rank == 1):
        msg = "SciPy LinOps are limited to 1D -> 1D maps."
        raise ValueError(msg)

    def matmat(arr):
        return op.apply(arr.T).T

    def rmatmat(arr):
        return op.adjoint(arr.T).T

    if dtype is None:
        dtype = pxrt.Width.DOUBLE.value

    if gpu:
        assert pxd.CUPY_ENABLED
        spx = pxu.import_module("cupyx.scipy.sparse.linalg")
    else:
        spx = spsl
    return spx.LinearOperator(
        shape=(op.codim_size, op.dim_size),
        matvec=matmat,
        rmatvec=rmatmat,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=dtype,
    )
