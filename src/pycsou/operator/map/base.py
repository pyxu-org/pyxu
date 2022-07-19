import types

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    "ConstantValued",
]


def ConstantValued(
    shape: pyct.Shape,
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
            return np.full(
                (*arr.shape[:-1], _.codim),
                fill_value=_._cst,
                dtype=arr.dtype,
                like=arr,
            )

        def op_jacobian(_, arr: pyct.NDArray) -> pyct.OpT:
            from pycsou.operator.linop import NullOp

            return NullOp(shape=_.shape)._squeeze()

        @pycrt.enforce_precision(i="arr")
        def op_grad(_, arr: pyct.NDArray) -> pyct.NDArray:
            return np.zeros_like(arr)

        @pycrt.enforce_precision(i=("arr", "tau"))
        def op_prox(_, arr: pyct.NDArray, tau: pyct.NDArray) -> pyct.NDArray:
            return arr

        klass = pyco.ProxDiffFunc if (shape[0] == 1) else pyco.DiffMap
        op = klass(shape=shape)
        op._cst = cst
        op._lipschitz = 0
        op._diff_lipschitz = 0

        op.apply = types.MethodType(op_apply, op)
        op.jacobian = types.MethodType(op_jacobian, op)
        op.grad = types.MethodType(op_grad, op)
        op.prox = types.MethodType(op_prox, op)

    return op._squeeze()
