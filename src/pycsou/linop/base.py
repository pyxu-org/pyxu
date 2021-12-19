import typing as typ

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu

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
