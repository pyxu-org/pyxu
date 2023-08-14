import numpy as np

import pycsou.abc as pyca
import pycsou.info.ptype as pyct
import pycsou.operator.interop.source as pycsrc
import pycsou.runtime as pycrt
import pycsou.util as pycu

__all__ = [
    "ConstantValued",
]


def ConstantValued(
    shape: pyct.OpShape,
    cst: pyct.Real,
) -> pyct.OpT:
    r"""
    Constant-valued operator :math:`C: \mathbb{R}^{M} \to \mathbb{R}^{N}`.
    """
    cst = float(cst)
    if np.isclose(cst, 0):
        from pycsou.operator.linop import NullOp

        op = NullOp(shape=shape)
    else:

        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            x = xp.full((1,), fill_value=_._cst, dtype=arr.dtype)
            out = xp.broadcast_to(x, (*arr.shape[:-1], _.codim))
            return out

        def op_jacobian(_, arr: pyct.NDArray) -> pyct.OpT:
            from pycsou.operator.linop import NullOp

            return NullOp(shape=_.shape).squeeze()

        @pycrt.enforce_precision(i="arr")
        def op_grad(_, arr: pyct.NDArray) -> pyct.NDArray:
            xp = pycu.get_array_module(arr)
            x = xp.zeros((1,), dtype=arr.dtype)
            out = xp.broadcast_to(x, arr.shape)
            return out

        @pycrt.enforce_precision(i=("arr", "tau"))
        def op_prox(_, arr: pyct.NDArray, tau: pyct.NDArray) -> pyct.NDArray:
            return pycu.read_only(arr)

        op = pycsrc.from_source(
            cls=pyca.ProxDiffFunc if (shape[0] == 1) else pyca.DiffMap,
            shape=shape,
            embed=dict(
                _name="ConstantValued",
                _cst=cst,
            ),
            apply=op_apply,
            jacobian=op_jacobian,
            grad=op_grad,
            prox=op_prox,
        )
        op.lipschitz = 0
        op.diff_lipschitz = 0
    return op.squeeze()
