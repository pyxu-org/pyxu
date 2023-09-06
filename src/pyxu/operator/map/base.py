import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "ConstantValued",
]


def ConstantValued(
    shape: pxt.OpShape,
    cst: pxt.Real,
) -> pxt.OpT:
    r"""
    Constant-valued operator :math:`C: \mathbb{R}^{M} \to \mathbb{R}^{N}`.
    """
    cst = float(cst)
    if np.isclose(cst, 0):
        from pyxu.operator import NullOp

        op = NullOp(shape=shape)
    else:

        @pxrt.enforce_precision(i="arr")
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            xp = pxu.get_array_module(arr)
            x = xp.full((1,), fill_value=_._cst, dtype=arr.dtype)
            out = xp.broadcast_to(x, (*arr.shape[:-1], _.codim))
            return out

        def op_jacobian(_, arr: pxt.NDArray) -> pxt.OpT:
            from pyxu.operator import NullOp

            return NullOp(shape=_.shape).squeeze()

        @pxrt.enforce_precision(i="arr")
        def op_grad(_, arr: pxt.NDArray) -> pxt.NDArray:
            xp = pxu.get_array_module(arr)
            x = xp.zeros((1,), dtype=arr.dtype)
            out = xp.broadcast_to(x, arr.shape)
            return out

        @pxrt.enforce_precision(i=("arr", "tau"))
        def op_prox(_, arr: pxt.NDArray, tau: pxt.NDArray) -> pxt.NDArray:
            return pxu.read_only(arr)

        op = px_src.from_source(
            cls=pxa.ProxDiffFunc if (shape[0] == 1) else pxa.DiffMap,
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
