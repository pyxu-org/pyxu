import warnings

import scipy.sparse.linalg as spsl

import pycsou.abc.operator as pyco
import pycsou.info.deps as pycd
import pycsou.info.ptype as pyct
import pycsou.info.warning as pycw
import pycsou.operator.interop.source as pycsrc
import pycsou.runtime as pycrt
import pycsou.util as pycu

__all__ = [
    "from_sciop",
]


def from_sciop(cls: pyct.OpC, sp_op: spsl.LinearOperator) -> pyct.OpT:
    r"""
    Wrap a :py:class:`scipy.sparse.linalg.LinearOperator` as a
    :py:class:`~pycsou.abc.operator.LinOp` (or sub-classes thereof).

    Parameters
    ----------
    sp_op: [scipy|cupyx].sparse.linalg.LinearOperator
        (N, M) Linear operator compliant with SciPy's interface.

    Returns
    -------
    op: pyct.OpT
        (N, M) Pycsou-compliant linear operator.
    """
    assert cls.has(pyco.Property.LINEAR)

    if sp_op.dtype not in [_.value for _ in pycrt.Width]:
        warnings.warn(
            "Computation may not be performed at the requested precision.",
            pycw.PrecisionWarning,
        )

    # [r]matmat only accepts 2D inputs -> reshape apply|adjoint inputs as needed.

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.dim)
        out = _._sp_op.matmat(arr.T).T
        out = out.reshape(*sh, _.codim)
        return out

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, _.codim)
        out = _._sp_op.rmatmat(arr.T).T
        out = out.reshape(*sh, _.dim)
        return out

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        # Determine XP-module accepted by sci_op, then compute array-representation.
        for ndi in [
            pycd.NDArrayInfo.NUMPY,
            pycd.NDArrayInfo.CUPY,
        ]:
            try:
                cls = _.__class__
                _A = cls.asarray(_, xp=ndi.module(), dtype=_._sp_op.dtype)
                break
            except Exception:
                pass

        # Cast to user specs.
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        A = xp.array(pycu.to_NUMPY(_A), dtype=dtype)
        return A

    def op_expr(_) -> tuple:
        return ("from_sciop", _._sp_op)

    op = pycsrc.from_source(
        cls=cls,
        shape=sp_op.shape,
        apply=op_apply,
        adjoint=op_adjoint,
        asarray=op_asarray,
        _expr=op_expr,
    )
    op._sp_op = sp_op

    return op
