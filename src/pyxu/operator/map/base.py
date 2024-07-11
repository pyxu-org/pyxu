import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.util as pxu

__all__ = [
    "ConstantValued",
]


def ConstantValued(
    dim_shape: pxt.NDArrayShape,
    codim_shape: pxt.NDArrayShape,
    cst: pxt.Real,
) -> pxt.OpT:
    r"""
    Constant-valued operator :math:`C: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{N_{1} \times\cdots\times N_{K}}`.
    """
    codim_shape = pxu.as_canonical_shape(codim_shape)

    cst = float(cst)
    if np.isclose(cst, 0):
        if codim_shape == (1,):
            from pyxu.operator import NullFunc

            op = NullFunc(dim_shape=dim_shape)
        else:
            from pyxu.operator import NullOp

            op = NullOp(
                dim_shape=dim_shape,
                codim_shape=codim_shape,
            )
    else:

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            ndi = pxd.NDArrayInfo.from_obj(arr)
            kwargs = dict()
            if ndi == pxd.NDArrayInfo.DASK:
                stack_chunks = arr.chunks[: -_.dim_rank]
                core_chunks = ("auto",) * _.codim_rank
                kwargs.update(chunks=stack_chunks + core_chunks)

            xp = ndi.module()
            sh = arr.shape[: -_.dim_rank]
            out = xp.broadcast_to(
                xp.array(_._cst, arr.dtype),
                (*sh, *_.codim_shape),
                **kwargs,
            )
            return out

        def op_jacobian(_, arr: pxt.NDArray) -> pxt.OpT:
            from pyxu.operator import NullOp

            return NullOp(
                dim_shape=_.dim_shape,
                codim_shape=_.codim_shape,
            )

        def op_grad(_, arr: pxt.NDArray) -> pxt.NDArray:
            ndi = pxd.NDArrayInfo.from_obj(arr)
            kwargs = dict()
            if ndi == pxd.NDArrayInfo.DASK:
                kwargs.update(chunks=arr.chunks)

            xp = ndi.module()
            out = xp.broadcast_to(
                xp.array(0, arr.dtype),
                arr.shape,
                **kwargs,
            )
            return out

        def op_prox(_, arr: pxt.NDArray, tau: pxt.NDArray) -> pxt.NDArray:
            return pxu.read_only(arr)

        if codim_shape == (1,):
            klass = pxa.ProxDiffFunc
        else:
            klass = pxa.DiffMap
        op = px_src.from_source(
            cls=klass,
            dim_shape=dim_shape,
            codim_shape=codim_shape,
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
    return op
