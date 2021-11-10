import numbers as nb
import types
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
from pycsou import runtime as pycrt
from pycsou import util as pycu
from pycsou.abc import Property, ProxFunc, SelfAdjointOp

NDArray = pycu.NDArray


class ExplicitLinFunc(pyco.LinFunc):
    @pycrt.enforce_precision(i="vec")
    def __init__(self, vec: NDArray):
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        xp = pycu.get_array_module(vec)
        self._vec = vec.copy().reshape(-1)
        self._lipschitz = xp.linalg.norm(vec)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: NDArray) -> NDArray:
        return (self._vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: NDArray) -> NDArray:
        return arr * self._vec

    @pycrt.enforce_precision(i="arr")
    def gradient(self, arr: NDArray) -> NDArray:
        xp = pycu.get_array_module(arr)
        arr = xp.asarray(arr)
        return xp.broadcast_to(self._vec, arr.shape)


class IdentityOperator(pyco.PosDefOp, pyco.SelfAdjointOp, pyco.UnitOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        pyco.PosDefOp.__init__(self, shape)
        pyco.SelfAdjointOp.__init__(self, shape)
        pyco.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: NDArray) -> NDArray:
        return arr


class HomothetyOp(SelfAdjointOp):
    def __init__(self, cst: nb.Real, dim: int):
        if not isinstance(cst, nb.Real):
            raise ValueError("Argument [cst] must be a real number.")
        super(HomothetyOp, self).__init__(shape=(dim, dim))
        self._cst = cst
        self._lipschitz = np.abs(cst)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: NDArray) -> NDArray:
        xp = pycu.get_array_module(arr)
        cst = xp.array(self._cst, dtype=arr.dtype)
        return cst * arr

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: NDArray) -> NDArray:
        xp = pycu.get_array_module(arr)
        cst = xp.array(self._cst, dtype=arr.dtype)
        return cst * arr

    def __mul__(self, other):
        out_op = Property.__mul__(self, other)
        if isinstance(other, ProxFunc):
            out_op.specialize(cast_to=ProxFunc)
            post_composition_prox = lambda obj, arr, tau: other.prox(arr, self._cst * tau)
            out_op.prox = types.MethodType(post_composition_prox, out_op)
            return out_op
