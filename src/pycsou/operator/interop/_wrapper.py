import types
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycw


class _Wrapper:
    @pycrt.enforce_precision(i=["lipschitz", "diff_lipschitz"])
    def __init__(
        self,
        op,
        shape: pyct.OpShape,
        arg_shape: pyct.NDArrayShape = None,
        cls: pyct.OpT = pyco.Map,
        lipschitz=np.inf,
        diff_lipschitz=np.inf,
        name: str = "PyDataOp",
        meta: typ.Any = None,
    ):
        self._lipschitz = lipschitz
        self._diff_lipschitz = diff_lipschitz
        self._op = op
        self._cls = cls
        self._shape = shape
        self._arg_shape = arg_shape if arg_shape is not None else pycu.as_canonical_shape(shape[1])
        self._name = name
        self._meta = op if meta is None else meta

    def op(self) -> pyct.OpT:
        if self._cls.has((pyco.Property.PROXIMABLE, pyco.Property.QUADRATIC)):
            raise pycw.AutoInferenceWarning(
                f"Automatic construction of a {self._cls} object from a DL operator is ambiguous."
            )
        op = self._cls(shape=self._shape)
        op._op = self._op  # embed for introspection
        op._arg_shape = self._arg_shape
        op._name = self._name
        op._meta = self._meta
        for p in op.properties():
            for name in p.arithmetic_attributes():
                attr = getattr(self, name)
                setattr(op, name, attr)
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name, None)
                if func is not None:
                    setattr(op, name, types.MethodType(func, op))  # comment on warning
        return op

    def _expr(self):
        return ("from_pydata", self._meta)
