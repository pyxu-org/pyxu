import types
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.interop._wrapper as pyciw
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "from_pytorch",
]


class _TorchWrapper(pyciw._Wrapper):
    @pycrt.enforce_precision(i=["lipschitz", "diff_lipschitz"], o=False)
    def __init__(
        self,
        func,
        shape: pyct.OpShape,
        arg_shape: pyct.NDArrayShape = None,
        cls: pyct.OpT = pyco.Map,
        lipschitz=np.inf,
        diff_lipschitz=np.inf,
        name: str = "TorchOp",
        meta: typ.Any = None,
    ):
        super().__init__(
            func,
            shape=shape,
            arg_shape=arg_shape,
            cls=cls,
            lipschitz=lipschitz,
            diff_lipschitz=diff_lipschitz,
            name=name,
            meta=meta,
        )

    @staticmethod
    def torch():
        return pycu.import_module("torch")

    @staticmethod
    def functorch():
        return pycu.import_module("torch.func")

    @staticmethod
    def _astensor(arr: pyct.NDArray, requires_grad: bool = False) -> "torch.Tensor":
        if pycd.NDArrayInfo.from_obj(arr) == pycd.NDArrayInfo.CUPY:
            return _TorchWrapper.torch().as_tensor(arr, device="cuda").requires_grad_(requires_grad)
        else:
            return _TorchWrapper.torch().from_numpy(arr).requires_grad_(requires_grad)

    @staticmethod
    def _asarray(tensor: "torch.Tensor") -> pyct.NDArray:
        if tensor.get_device() == -1:
            return tensor.detach().numpy(force=False)
        else:
            cp = pycd.NDArrayInfo.CUPY.module()
            return cp.asarray(tensor.detach())

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        @pycu._delayed_if_dask(i="x", oshape=arr.shape[:-1] + (self._shape[0],))
        def _apply(x: pyct.NDArray) -> pyct.NDArray:
            tensor = _TorchWrapper._astensor(x.reshape((-1,) + self._arg_shape))
            with _TorchWrapper.torch().inference_mode():
                return _TorchWrapper._asarray(self._op(tensor)).reshape(x.shape[:-1] + (-1,))

        return _apply(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        arr_tensor = _TorchWrapper._astensor(arr.reshape(self._arg_shape))

        @pycrt.enforce_precision(i="tan")
        def op_apply(_, tan: pyct.NDArray) -> pyct.NDArray:
            @pycu._delayed_if_dask(i="x", oshape=tan.shape[:-1] + (self._shape[0],))
            def _apply(x: pyct.NDArray) -> pyct.NDArray:
                tan_tensor = _TorchWrapper._astensor(x.reshape((-1,) + self._arg_shape))
                jvp = lambda v: _TorchWrapper.functorch().jvp(self._op, primals=(arr_tensor,), tangents=(v,))[1]
                batched_jvp = _TorchWrapper.functorch().vmap(jvp)
                return _TorchWrapper._asarray(batched_jvp(tan_tensor)).reshape(x.shape[:-1] + (-1,))

            return _apply(tan)

        @pycrt.enforce_precision(i="cotan")
        def op_adjoint(_, cotan: pyct.NDArray) -> pyct.NDArray:
            @pycu._delayed_if_dask(i="y", oshape=cotan.shape[:-1] + (self._shape[1],))
            def _adjoint(y: pyct.NDArray) -> pyct.NDArray:
                cotan_tensor = _TorchWrapper._astensor(y.reshape((-1, self._shape[0])))
                call_ravel = lambda t: _TorchWrapper.torch().reshape(self._op(t), (self._shape[0],))
                _, vjp = _TorchWrapper.functorch().vjp(call_ravel, arr_tensor)
                batched_vjp = lambda v: _TorchWrapper.functorch().vmap(vjp)(v)[0]
                return _TorchWrapper._asarray(batched_vjp(cotan_tensor)).reshape(y.shape[:-1] + (-1,))

            return _adjoint(cotan)

        def op_expr(_) -> tuple:
            return ("autojac_from_pytorch", self)

        op = pyco.LinOp(shape=self._shape)
        op._name = "AutoJacOp"
        for (name, meth) in [
            ("apply", op_apply),
            ("adjoint", op_adjoint),
            ("_expr", op_expr),
        ]:
            setattr(op, name, types.MethodType(meth, op))
        return op

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        @pycu._delayed_if_dask(i="x", oshape=arr.shape[:-1] + (self._shape[0],))
        def _grad(x: pyct.NDArray) -> pyct.NDArray:
            tensor = _TorchWrapper._astensor(x.reshape((-1,) + self._arg_shape))
            grad = _TorchWrapper.functorch().vmap(_TorchWrapper.functorch().grad(self._op))
            return _TorchWrapper._asarray(grad(tensor)).reshape(x.shape[:-1] + (-1,))

        return _grad(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        jac = self.jacobian(arr)
        return jac.adjoint(arr)

    def _expr(self):
        return ("from_pytorch", self._meta)


def from_pytorch(
    func: typ.Callable,
    shape: pyct.OpShape,
    arg_shape: pyct.NDArrayShape = None,
    cls: pyct.OpT = pyco.Map,
    lipschitz=np.inf,
    diff_lipschitz=np.inf,
    name: str = "TorchOp",
    meta: typ.Any = None,
) -> pyct.OpT:
    r"""
    Wrap a Python function as a
    :py:class:`~pycsou.abc.operator.Map` (or sub-classes thereof).

    Parameters
    ----------
    func: Callable
        A Python function with single-element Tensor input/output. Defines the :py:meth:`~pycsou.abc.operator.Map.apply`
        method of the operator: ``func(x)==op.apply(x)``.
    shape: tuple
        (N,M) shape of the operator, where N and M are the sizes of the output and input Tensors respectively.
    arg_shape: tuple
        Optional shape of the input Tensor for N-D inputs.
    cls: pyct.OpT
        Pycsou abstract base class to subclass from.
    lipschitz: float
        Lipschitz constant of the operator (if known).
    diff_lipschitz: float
        Diff-Lipschitz constant of the operator (if known).
    name: str
        Name of the operator.
    meta: Any
        Meta information to be provided as tail to :py:class:`~pycsou.abc.operator.Operator._expr`.
    vectorize: bool
        If ``True``, ``func`` is vectorized via `torch.func.vmap <https://pytorch.org/docs/stable/generated/torch.func.vmap.html#torch-func-vmap>`_
        to be able to process batched inputs (not implemented yet).

    Returns
    -------
    op: pyct.OpT
        (N, M) Pycsou-compliant operator.

    Notes
    -----
    For :py:class:`~pycsou.abc.operator.DiffMap` (or subclasses thereof) the methods :py:meth:`~pycsou.abc.operator.DiffMap.jacobian`,
    :py:meth:`~pycsou.abc.operator.DiffFunc.grad` and :py:meth:`~pycsou.abc.operator.DiffFunc.adjoint` [*]_ are defined implicitly
    using the auto-differentiation transforms from the module `torch.func <https://pytorch.org/docs/stable/func.html>`_
    of PyTorch. As detailed `on this page <https://pytorch.org/docs/stable/func.ux_limitations.html>`_, such transforms works
    well on pure functions (that is, functions where the output is completely determined by the input and that do not
    involve side effects like mutation), but may fail on more complex functions. Moreover, the ``torch.func`` module does not
    yet have full coverage over PyTorch operations. For functions that calls a ``torch.nn.Module``
    `see here for some utilities <https://pytorch.org/docs/stable/func.api.html#utilities-for-working-with-torch-nn-modules>`_.

    .. [*] For a linear operator ``L`` we have ``L.jacobian(arr)==L`` for any input ``arr``. Given its apply method, the
           adjoint of ``L`` can hence be computed via automatic-differentiation as ``L.adjoint(arr) = L.jacobian(arr).adjoint(arr)``.

    Warnings
    --------
    Operators defined from external librairies only have weak-compatibility with Dask, that is we delay the execution of the function
    but the Dask array is fed to the latter in a single chunk.
    """
    return _TorchWrapper(
        func,
        shape=shape,
        arg_shape=arg_shape,
        cls=cls,
        lipschitz=lipschitz,
        diff_lipschitz=diff_lipschitz,
        name=name,
        meta=meta,
    ).op()


if __name__ == "__main__":

    import time as t

    import cupy as cp
    import numpy as np
    import torch

    import pycsou.runtime as pycrt

    xp = cp
    batch_size, in_size, out_size = 200, 4000, 3000
    m = torch.nn.Linear(in_size, out_size)
    device = {cp: "cuda", np: "cpu"}
    if xp == cp:
        m = m.cuda()
    op = from_pytorch(lambda x: m(x), shape=(out_size, in_size), name="TorchOp", cls=pyco.DiffMap, meta=m)
    arr = xp.ones((batch_size, in_size), dtype=np.float32)
    arr_t = torch.ones((batch_size, in_size), dtype=torch.float32, device=device[xp], requires_grad=True)
    with pycrt.Precision(pycrt.Width.SINGLE):
        t1 = t.time()
        y1 = op(arr)
        print(f"{t.time()-t1} seconds ellapsed (Pycsou wrapper)")
    t1 = t.time()
    with torch.inference_mode():
        y2 = m(arr_t)
    print(f"{t.time() - t1} seconds ellapsed (Pytorch)")
    assert xp.allclose(y1, _TorchWrapper._asarray(y2), atol=1e-4)

    with pycrt.Precision(pycrt.Width.SINGLE):
        jac = op.jacobian(arr[0])

    tangents = xp.eye(in_size, dtype=np.float32)
    cotangents = xp.eye(out_size, dtype=np.float32)

    with pycrt.Precision(pycrt.Width.SINGLE):
        jac_mat = jac.asarray(xp=cp, dtype=cp.float32)
        jac_batch = jac.apply(arr)
    assert xp.allclose(jac_mat, _TorchWrapper._asarray(m.weight), atol=1e-4)

    with pycrt.Precision(pycrt.Width.SINGLE):
        jac_matT = jac.adjoint(cotangents)
    assert xp.allclose(jac_matT, _TorchWrapper._asarray(m.weight), atol=1e-4)
