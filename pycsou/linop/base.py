import pycsou.abc.operator as pycop
import pycsou.abc
import numpy as np
import typing as typ

NDArray = pycsou.abc.NDArray


class ExplicitLinFunc(pycop.LinFunc):
    def __init__(self, vec: NDArray):
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        self._vec = vec.copy().reshape(-1)
        self._lipschitz = np.linalg.norm(vec)
        self._diff_lipschitz = 0

    def apply(self, arr: NDArray) -> NDArray:
        return (self._vec[None, :] * arr).sum(axis=-1)

    def adjoint(self, arr: NDArray) -> NDArray:
        return arr * self._vec[None, :]

    def gradient(self, arr: NDArray) -> NDArray:
        return np.broadcast_to(self._vec, arr.shape)


class IdentityOperator(pycop.PosDefOp, pycop.SelfAdjointOp, pycop.UnitOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        pycop.PosDefOp.__init__(self, shape)
        pycop.SelfAdjointOp.__init__(self, shape)
        pycop.UnitOp.__init__(self, shape)

    def apply(self, arr: NDArray) -> NDArray:
        return arr
