import types
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class ExplicitLinFunc(pyco.LinFunc):
    @pycrt.enforce_precision(i="vec")
    def __init__(self, vec: pyct.NDArray):
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        xp = pycu.get_array_module(vec)
        self._vec = vec.copy().reshape(-1)
        self._lipschitz = xp.linalg.norm(vec)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return (self._vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr * self._vec

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(self._vec, arr.shape)

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr - tau * self._vec


class IdentityOp(pyco.PosDefOp, pyco.SelfAdjointOp, pyco.UnitOp):
    def __init__(self, shape: pyct.Shape):
        pyco.PosDefOp.__init__(self, shape)
        pyco.SelfAdjointOp.__init__(self, shape)
        pyco.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr


class HomothetyOp(pyco.SelfAdjointOp):
    def __init__(self, cst: pyct.Real, dim: int):
        if not isinstance(cst, pyct.Real):
            raise ValueError(f"cst: expected real number, got {cst}.")
        super(HomothetyOp, self).__init__(shape=(dim, dim))
        self._cst = cst
        self._lipschitz = abs(cst)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def __mul__(self, other):
        op = pyco.Property.__mul__(self, other)
        if isinstance(other, pyco.ProxFunc):
            op.specialize(cast_to=pyco.ProxFunc)
            post_composition_prox = lambda obj, arr, tau: other.prox(arr, self._cst * tau)
            op.prox = types.MethodType(post_composition_prox, op)
            return op


class NullOp(pyco.LinOp):
    def __init__(self, shape: typ.Tuple[int, int]):
        super(NullOp, self).__init__(shape)
        self._lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (arr.shape[:-1], self.dim),
        )


class NullFunc(NullOp, pyco.LinFunc):
    def __init__(self, dim: typ.Optional[int] = None):
        pyco.LinFunc.__init__(self, shape=(1, dim))
        NullOp.__init__(self, shape=self.shape)

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr
