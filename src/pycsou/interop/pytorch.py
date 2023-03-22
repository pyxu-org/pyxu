import types
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycw


class _DLWrapper:
    @pycrt.enforce_precision(i=["lipschitz", "diff_lipschitz"])
    def __init__(self, dlop, shape: pyct.OpShape, base: pyct.OpT = pyco.Map, lipschitz=np.inf, diff_lipschitz=np.inf):
        self._lipschitz = lipschitz
        self._diff_lipschitz = diff_lipschitz
        self._dlop = dlop
        self._base = base
        self._shape = shape

    def op(self) -> pyct.OpT:
        if self._base.has((pyco.Property.PROXIMABLE, pyco.Property.QUADRATIC)):
            raise pycw.AutoInferenceWarning(
                f"Automatic construction of a {self._base} object from a DL operator is ambiguous."
            )
        op = self._base(shape=self._shape)
        op._dlop = self._dlop  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_attributes():
                attr = getattr(self, name)
                setattr(op, name, attr)
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(op, name, types.MethodType(func, op))
        return op


class _TorchWrapper:
    def astensor(self, arr: pyct.NDArray) -> pyct.NDArray:
        pass

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._torch.apply(arr)
