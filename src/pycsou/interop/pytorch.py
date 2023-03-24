import types
import warnings

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycw


class _PyDataWrapper:
    @pycrt.enforce_precision(i=["lipschitz", "diff_lipschitz"])
    def __init__(
        self,
        pydata_op,
        shape: pyct.OpShape,
        arg_shape: pyct.NDArrayShape = None,
        base: pyct.OpT = pyco.Map,
        lipschitz=np.inf,
        diff_lipschitz=np.inf,
        name: str = "PyDataOp",
    ):
        self._lipschitz = lipschitz
        self._diff_lipschitz = diff_lipschitz
        self._pydata_op = pydata_op
        self._base = base
        self._shape = shape
        self._arg_shape = arg_shape if arg_shape is not None else (shape[1],)  # pycu.canonical_form
        self._name = name

    def op(self) -> pyct.OpT:
        if self._base.has((pyco.Property.PROXIMABLE, pyco.Property.QUADRATIC)):
            raise pycw.AutoInferenceWarning(
                f"Automatic construction of a {self._base} object from a DL operator is ambiguous."
            )
        op = self._base(shape=self._shape)
        op._pydata_op = self._pydata_op  # embed for introspection
        op._arg_shape = self._arg_shape
        op._name = self._name
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
        return ("import", self._pydata_op)


class _TorchWrapper(_PyDataWrapper):
    torch = pycu.import_module("torch")

    def __init__(
        self,
        torchop,
        shape: pyct.OpShape,
        arg_shape: pyct.NDArrayShape = None,
        base: pyct.OpT = pyco.Map,
        lipschitz=np.inf,
        diff_lipschitz=np.inf,
        name: str = "TorchOp",
    ):
        super().__init__(
            torchop,
            shape=shape,
            arg_shape=arg_shape,
            base=base,
            lipschitz=lipschitz,
            diff_lipschitz=diff_lipschitz,
            name=name,
        )

    @staticmethod
    def _astensor(arr: pyct.NDArray, requires_grad: bool = False) -> "torch.Tensor":
        if pycd.NDArrayInfo.from_obj(arr) == pycd.NDArrayInfo.CUPY:
            return torch.as_tensor(arr, device="cuda").requires_grad_(requires_grad)
        else:
            return torch.from_numpy(arr).requires_grad_(requires_grad)

    @staticmethod
    def _asarray(tensor: "torch.Tensor") -> pyct.NDArray:
        if tensor.get_device() == -1:
            return tensor.detach().numpy(force=False)
        else:
            cp = pycd.NDArrayInfo.CUPY.module()
            return cp.asarray(tensor.detach())

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        @pycu._delayed_if_dask(oshape=arr.shape[:-1] + (self._shape[0],))
        def _apply(arr: pyct.NDArray) -> pyct.NDArray:
            tensor = _TorchWrapper._astensor(arr.reshape((-1,) + self._arg_shape))
            with torch.inference_mode():
                return _TorchWrapper._asarray(self._pydata_op(tensor)).reshape(arr.shape[:-1] + (-1,))

        return _apply(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        pass


def from_pytorch(
    torchop,
    shape: pyct.OpShape,
    arg_shape: pyct.NDArrayShape = None,
    base: pyct.OpT = pyco.Map,
    lipschitz=np.inf,
    diff_lipschitz=np.inf,
    name: str = "TorchOp",
):
    return _TorchWrapper(
        torchop,
        shape=shape,
        arg_shape=arg_shape,
        base=base,
        lipschitz=lipschitz,
        diff_lipschitz=diff_lipschitz,
        name=name,
    ).op()


if __name__ == "__main__":

    import pdb
    import time as t

    import cupy as cp
    import dask.array as da
    import numpy as np
    import torch

    import pycsou.runtime as pycrt

    xp = cp
    size = 4000
    m = torch.nn.Linear(size, size)
    device = {cp: "cuda", np: "cpu"}
    if xp == cp:
        m = m.cuda()
    op = from_pytorch(m, shape=(size, size), name="TorchOp", base=pyco.LinOp)
    arr = da.from_array(xp.ones(size, dtype=np.float32))
    arr_t = torch.ones(size, dtype=torch.float32, device=device[xp])
    with pycrt.Precision(pycrt.Width.SINGLE):
        t1 = t.time()
        y1 = op(arr).compute()
        print(f"{t.time()-t1} seconds ellapsed (Pycsou wrapper)")
    t1 = t.time()
    with torch.inference_mode():
        y2 = m(arr_t)
    print(f"{t.time() - t1} seconds ellapsed (Pytorch)")
    assert xp.allclose(y1, _TorchWrapper._asarray(y2))
