import typing as typ

import numpy as np

import pycsou.abc
import pycsou.abc.operator as pycop
import pycsou.runtime as pycrt

NDArray = pycsou.abc.NDArray


class ExplicitLinFunc(pycop.LinFunc):
    @pycrt.enforce_precision(i="vec")
    def __init__(self, vec: NDArray):
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        self._vec = vec.copy().reshape(-1)
        self._lipschitz = np.linalg.norm(vec)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: NDArray) -> NDArray:
        return (self._vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: NDArray) -> NDArray:
        return arr * self._vec

    @pycrt.enforce_precision(i="arr")
    def gradient(self, arr: NDArray) -> NDArray:
        arr = np.asarray(arr)
        return np.broadcast_to(self._vec, arr.shape)


class IdentityOperator(pycop.PosDefOp, pycop.SelfAdjointOp, pycop.UnitOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        pycop.PosDefOp.__init__(self, shape)
        pycop.SelfAdjointOp.__init__(self, shape)
        pycop.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: NDArray) -> NDArray:
        return arr
