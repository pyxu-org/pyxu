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
]


def from_sciop(cls: pxt.OpC, sp_op: spsl.LinearOperator) -> pxt.OpT:
    r"""
    Wrap a :py:class:`~scipy.sparse.linalg.LinearOperator` as a :py:class:`~pyxu.abc.LinOp` (or sub-class thereof).

    Parameters
    ----------
    sp_op: ~scipy.sparse.linalg.LinearOperator
        (N, M) Linear CPU/GPU operator compliant with SciPy's interface.

    Returns
    -------
    op: OpT
        (N, M) Pyxu-compliant linear operator.
    """
    assert cls.has(pxa.Property.LINEAR)

    if sp_op.dtype not in [_.value for _ in pxrt.Width]:
        warnings.warn(
            "Computation may not be performed at the requested precision.",
            pxw.PrecisionWarning,
        )

    # [r]matmat only accepts 2D inputs -> reshape apply|adjoint inputs as needed.

    @pxrt.enforce_precision(i="arr")
    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.dim)
        out = _._sp_op.matmat(arr.T).T
        out = out.reshape(*sh, _.codim)
        return out

    @pxrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.codim)
        out = _._sp_op.rmatmat(arr.T).T
        out = out.reshape(*sh, _.dim)
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
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        A = xp.array(pxu.to_NUMPY(_A), dtype=dtype)
        return A

    def op_expr(_) -> tuple:
        return ("from_sciop", _._sp_op)

    op = px_src.from_source(
        cls=cls,
        shape=sp_op.shape,
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        _expr=op_expr,
    )
    op._sp_op = sp_op

    return op
