r"""
.. Warning::

   This package requires the optional dependency `PyTorch <https://pytorch.org/>`_. See installation instructions.
"""

import typing as typ
import warnings
from functools import wraps

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.interop.source as pycsrc
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw
from pycsou.util.inspect import import_module

torch = import_module("torch", fail_on_error=False)
if torch is not None:
    import torch._dynamo as dynamo
    import torch.func as functorch

    TorchTensor = torch.Tensor
else:
    TorchTensor = typ.TypeVar("torch.Tensor")

__all__ = [
    "from_torch",
    "astensor",
    "asarray",
]


def _traceable(f):
    # Needed to compile functorch transforms. See this issue: https://github.com/pytorch/pytorch/issues/98822
    f = dynamo.allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def astensor(arr: pyct.NDArray, requires_grad: bool = False) -> TorchTensor:
    r"""
    Convert a Numpy-like array into a PyTorch tensor, sharing data, dtype and device.

    Parameters
    ----------
    arr: pyct.NDArray
        Input array.
    requires_grad: bool
        If autograd should record operations on the returned tensor.

    Returns
    -------
    tensor: pyct.NDArray
        Output tensor.

    Notes
    -----
    The returned tensor and input array share the same memory.
    Modifications to the tensor will be reflected in the ndarray and vice versa.
    The returned tensor is not resizable.
    """
    if pycd.NDArrayInfo.from_obj(arr) == pycd.NDArrayInfo.CUPY:
        with torch.device("cuda", arr.device.id):
            return torch.as_tensor(arr).requires_grad_(requires_grad)
    else:
        return torch.from_numpy(arr).requires_grad_(requires_grad)


def asarray(tensor: TorchTensor) -> pyct.NDArray:
    r"""
    Convert a PyTorch tensor into a Numpy-like array, sharing data, dtype and device.

    Parameters
    ----------
    tensor: TorchTensor
        Input tensor.

    Returns
    -------
    arr: pyct.NDArray
        Output array.

    Notes
    -----
    The returned array and input tensor share the same memory.
    Modifications to the tensor will be reflected in the ndarray and vice versa.
    """
    if tensor.get_device() == -1:
        return tensor.detach().numpy(force=False)
    else:
        cp = pycd.NDArrayInfo.CUPY.module()
        with cp.cuda.Device(tensor.get_device()):
            return cp.asarray(tensor.detach())


class _FromTorch(pycsrc._FromSource):
    # supported attributes/methods in __init__(**kwargs)
    _attr = frozenset({"_lipschitz", "_diff_lipschitz"})
    _meth = frozenset({"apply", "grad", "prox", "pinv", "adjoint"})

    @pycrt.enforce_precision(i=["lipschitz", "diff_lipschitz"], o=False)
    def __init__(  # See from_torch() for a detailed description.
        self,
        apply: typ.Callable,
        shape: pyct.OpShape,
        cls: pyct.OpC = pyco.Map,
        arg_shape: pyct.NDArrayShape = None,
        out_shape: pyct.NDArrayShape = None,
        lipschitz: pyct.Real = np.inf,
        diff_lipschitz: pyct.Real = np.inf,
        vectorize: frozenset[str] = frozenset(),
        batch_size: typ.Optional[int] = None,
        jit: bool = False,  # Unused for now
        dtype: typ.Optional[pyct.DType] = None,
        enable_warnings: bool = True,
        name: str = "TorchOp",
        meta: typ.Any = None,
        **kwargs,
    ):
        if arg_shape is not None:
            assert shape[1] == np.prod(arg_shape)
        else:
            arg_shape = pycu.as_canonical_shape(shape[1])
        if out_shape is not None:
            assert shape[0] == np.prod(out_shape)
        else:
            out_shape = pycu.as_canonical_shape(shape[0])
        super().__init__(
            cls=cls,
            shape=shape,
            embed=dict(),
            vectorize=vectorize,
            vmethod=None,  # pycsou.util.vectorize() not used for torch-funcs
            apply=apply,
            _lipschitz=lipschitz,
            _diff_lipschitz=diff_lipschitz,
            enforce_precision=frozenset(),  # will be applied manually
            **kwargs,
        )
        self._arg_shape = arg_shape
        self._out_shape = out_shape
        self._batch_size = batch_size
        self._dtype = dtype
        self._jit = False  # JIT-compilation is currently deactivated until torch.func goes out of beta.
        self._enable_warnings = enable_warnings
        self._name = name
        self._meta = meta

        # Only a subset of arithmetic attributes/methods allowed from from_torch().
        if not (set(self._kwargs) <= self._attr | self._meth):
            msg_head = "Unsupported arithmetic attributes/methods:"
            unsupported = set(self._kwargs) - (self._attr | self._meth)
            msg_tail = ", ".join([f"{name}()" for name in unsupported])
            raise ValueError(f"{msg_head} {msg_tail}")

    def op(self) -> pyct.OpT:
        # Idea: modify `**kwargs` from constructor to [when required]:
        #   1. auto-define omitted methods.            [_infer_missing()]
        #   2. auto-vectorize via vmap().              [_auto_vectorize()]
        #   3. JIT-compile via compile().              [_compile()]
        #   4. TORCH<>NUMPY/CUPY conversions.          [_interface()]
        #   Note: JIT-compilation is currently deactivated due to the undocumented interaction of torch.func transforms and
        #   torch.compile. Will be reactivated once torch.func goes out of beta.

        self._infer_missing()
        self._compile()
        self._auto_vectorize()
        t_state, kwargs = self._interface()

        _op = pycsrc.from_source(
            cls=self._op.__class__,
            shape=self._op.shape,
            embed=dict(
                _arg_shape=self._arg_shape,
                _out_shape=self._out_shape,
                _batch_size=self._batch_size,
                _dtype=self._dtype,
                _jit=self._jit,
                _enable_warnings=self._enable_warnings,
                _name=self._name,
                _meta=self._meta,
                _coerce=self._coerce,
                _torch=t_state,
            ),
            # vectorize=None,  # see top-level comment.
            # vmethod=None,    #
            **kwargs,
        )
        return _op

    def _infer_missing(self):
        # The following methods must be auto-inferred if missing from `kwargs`:
        #
        #     grad(), adjoint()
        #
        # Missing methods are auto-inferred via auto-diff rules and added to `_kwargs`.
        # At the end of _infer_missing(), all torch-funcs required for _interface() have been added to
        # `_kwargs`.
        #
        # Notes
        # -----
        # This method does NOT produce vectorized implementations: _auto_vectorize() is responsible
        # for this.
        self._vectorize = set(self._vectorize)  # to allow updates below

        nl_difffunc = all(  # non-linear diff-func
            [
                self._op.has(pyco.Property.DIFFERENTIABLE_FUNCTION),
                not self._op.has(pyco.Property.LINEAR),
            ]
        )
        if nl_difffunc and ("grad" not in self._kwargs):

            def f_grad(tensor: TorchTensor) -> TorchTensor:
                grad = functorch.grad(self._kwargs["apply"])
                return grad(tensor)

            self._vectorize.add("grad")
            self._kwargs["grad"] = f_grad

        non_selfadj = all(  # linear, but not self-adjoint
            [
                self._op.has(pyco.Property.LINEAR),
                not self._op.has(pyco.Property.LINEAR_SELF_ADJOINT),
            ]
        )
        if non_selfadj and ("adjoint" not in self._kwargs):

            def f_adjoint(tensor: TorchTensor) -> TorchTensor:
                # out_shape -> arg_shape
                f = self._kwargs["apply"]
                primal = torch.zeros(size=self._arg_shape, dtype=tensor.dtype, device=tensor.device)
                _, f_vjp = functorch.vjp(f, primal)
                out = f_vjp(tensor)[0]  # f_vjp returns a tuple
                return out  # shape arg_shape

            self._vectorize.add("adjoint")
            self._kwargs["adjoint"] = f_adjoint

        self._vectorize = frozenset(self._vectorize)

    def _compile(self):
        # JIT-compile user-specified [or _infer_missing()-added] arithmetic methods via torch.compile().
        #
        # Modifies `_kwargs` to hold compiled torch-funcs.
        # Note: Currently deactivated until torch.func goes out of beta.

        if self._jit:
            for name in self._kwargs:
                if name in self._meth:
                    func = self._kwargs[name]  # necessarily torch_func
                    self._kwargs[name] = torch.compile(func)
        else:
            pass

    def _auto_vectorize(self):
        # Vectorize user-specified [or _infer_missing()-added] arithmetic methods via torch.vmap().
        #
        # Modifies `_kwargs` to hold vectorized torch-funcs.

        for name in self._kwargs:
            if name in self._vectorize:
                func = self._kwargs[name]  # necessarily torch_func
                if name in [
                    "prox",
                    "pinv",
                ]:  # These methods have two arguments, but vectorization should be for the first argument only.
                    self._kwargs[name] = _traceable(torch.vmap(func, in_dims=(0, None), chunk_size=self._batch_size))
                else:
                    self._kwargs[name] = _traceable(torch.vmap(func, chunk_size=self._batch_size))

    def _interface(self):
        # Arithmetic methods supplied in `kwargs`:
        #
        #     * take `torch.Tensor` inputs
        #     * do not have the `self` parameter. (Reason: to be JIT-compatible.)
        #
        # This method creates modified arithmetic functions to match Pycsou's API, and `state`
        # required for them to work.
        #
        # Returns
        # -------
        # t_state: dict
        #     Torch functions referenced by wrapper arithmetic methods. (See below.)
        # kwargs: dict
        #     Pycsou-compatible functions which can be submitted to interop.from_source().

        t_state = dict()
        for name, obj in self._kwargs.items():
            if name in self._meth:
                t_state[name] = obj  # necessarily torch_func (potentially auto_vec/jitted at this stage)

        kwargs = dict()
        for name, obj in self._kwargs.items():
            if name in self._attr:
                kwargs[name] = obj
            else:
                func = getattr(self.__class__, name)
                kwargs[name] = func  # necessarily a pycsou_func

        # Special cases.
        # (Reason: not in `_kwargs` [c.f. from_torch() docstring], but need to be inferred.)
        for name in (
            "jacobian",
            "_quad_spec",
            "_expr",
        ):  # "asarray"):
            kwargs[name] = getattr(self.__class__, name)

        return t_state, kwargs

    def _coerce(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Coerce inputs (and raise a warning) in case of precision mis-matches.
        if self._dtype is not None and (arr.dtype != self._dtype):
            if self._enable_warnings:
                msg = f"Precision mis-match! Input array was coerced to {self._dtype} precision automatically."
                warnings.warn(msg, pycuw.PrecisionWarning)
            arr = arr.astype(self._dtype, copy=False)
        return arr

    # Wrapper arithmetic methods ----------------------------------------------
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = self._coerce(arr)
        tensor = astensor(arr.reshape((-1,) + self._arg_shape))
        func = self._torch["apply"]
        out = asarray(func(tensor)).reshape(arr.shape[:-1] + (-1,))
        return out

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = self._coerce(arr)
        tensor = astensor(arr.reshape((-1,) + self._arg_shape))
        func = self._torch["grad"]
        return asarray(func(tensor)).reshape(arr.shape[:-1] + (-1,))

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = self._coerce(arr)
        tensor = astensor(arr.reshape((-1,) + self._out_shape))
        func = self._torch["adjoint"]
        return asarray(func(tensor)).reshape(arr.shape[:-1] + (-1,))

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        arr = self._coerce(arr)
        tensor = astensor(arr.reshape((-1,) + self._arg_shape))
        func = self._torch["prox"]
        return asarray(func(tensor, tau)).reshape(arr.shape[:-1] + (-1,))

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        arr = self._coerce(arr)
        tensor = astensor(arr.reshape((-1,) + self._out_shape))
        func = self._torch["pinv"]
        damp = kwargs.get("damp", 0)
        out = func(tensor, damp)  # positional args only if auto-vectorized.
        return asarray(out).reshape(arr.shape[:-1] + (-1,))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        try:
            # Use the class' method if available ...
            klass = self.__class__
            op = klass.jacobian(self, arr)
        except NotImplementedError:
            # ... and fallback to auto-inference if undefined.
            arr = self._coerce(arr)
            primal = astensor(arr.reshape(self._arg_shape))
            f = self._torch["apply"]

            def jf_apply(tan: TorchTensor) -> TorchTensor:
                return functorch.jvp(f, primals=(primal,), tangents=(tan,))[1]

            def jf_adjoint(cotan: TorchTensor) -> TorchTensor:
                _, f_vjp = functorch.vjp(f, primal)
                out = f_vjp(cotan)[0]  # f_vjp returns a tuple
                return out

            klass = pyco.LinFunc if (self.codim == 1) else pyco.LinOp
            op = from_torch(
                apply=jf_apply,
                shape=self.shape,
                cls=klass,
                arg_shape=self._arg_shape,
                out_shape=self._out_shape,
                lipschitz=self._diff_lipschitz,
                vectorize=("apply", "adjoint"),
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
                name="TorchJacobian",
                meta=self,
                adjoint=jf_adjoint,
            )
        return op

    def _quad_spec(self):
        if self.has(pyco.Property.QUADRATIC):
            # auto-infer (Q, c, t)
            _grad = self._torch["grad"]

            def Q_apply(tensor: TorchTensor) -> TorchTensor:
                # \grad_{f}(x) = Q x + c = Q x + \grad_{f}(0)
                # ==> Q x = \grad_{f}(x) - \grad_{f}(0)
                z = torch.zeros_like(tensor)
                out = _grad(tensor) - _grad(z)
                return out

            def c_apply(tensor: TorchTensor) -> TorchTensor:
                z = torch.zeros_like(tensor)
                c = _grad(z)
                out = torch.sum(c * tensor, dtype=tensor.dtype, dim=-1, keepdim=True)
                return out

            Q = from_torch(
                apply=Q_apply,
                shape=(self.dim, self.dim),
                cls=pyco.PosDefOp,
                arg_shape=self._arg_shape,
                lipschitz=self._diff_lipschitz,
                vectorize="apply",
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
                name="TorchQuadSpec_Q",
                meta=self,
            )

            c = from_torch(
                apply=c_apply,
                shape=(1, self.dim),
                cls=pyco.LinFunc,
                arg_shape=self._arg_shape,
                vectorize="apply",
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
                name="TorchQuadSpec_c",
                meta=self,
            )

            # We cannot know a-priori which backend the supplied torch-apply() function works with.
            # Consequence: to compute `t`, we must try different backends until one works.
            f = self._torch["apply"]
            try:  # test a CPU implementation ...
                with torch.device("cpu"):
                    t = float(f(torch.zeros(self.dim)))
            except Exception:  # ... and use GPU(s) if it doesn't work.
                for id in range(torch.cuda.device_count()):
                    try:
                        with torch.device("gpu", id):
                            t = float(f(torch.zeros(self.dim)))
                    except Exception:
                        continue

            return (Q, c, t)
        else:
            raise NotImplementedError

    # def asarray(self, **kwargs) -> pyct.NDArray:
    #     if self.has(pyco.Property.LINEAR):
    #         # If the operator is backend-specific (i.e. only works with CUPY), then we have no
    #         # way to determine which `xp` value use in the generic LinOp.asarray()
    #         # implementation without a potential fail.
    #         #
    #         # Consequence:
    #         # We must try different `xp` values until one works.
    #         N = pycd.NDArrayInfo  # shorthand
    #         dtype_orig = kwargs.get("dtype", pycrt.getPrecision().value)
    #         xp_orig = kwargs.get("xp", N.NUMPY.module())
    #
    #         klass = self.__class__
    #         try:
    #             A = klass.asarray(self, dtype=dtype_orig, xp=N.NUMPY.module())
    #         except:
    #             A = klass.asarray(self, dtype=dtype_orig, xp=N.CUPY.module())
    #
    #         # Not the most efficient method, but fail-proof
    #         A = xp_orig.array(pycu.to_NUMPY(A), dtype=dtype_orig)
    #         return A
    #     else:
    #         raise NotImplementedError

    def _expr(self):
        torch_funcs = list(self._torch.keys())
        if self._meta is not None:
            torch_funcs.append(self._meta)
        return ("from_torch", *torch_funcs)


def from_torch(
    apply: typ.Callable,
    shape: pyct.OpShape,
    cls: pyct.OpC = pyco.Map,
    arg_shape: pyct.NDArrayShape = None,
    out_shape: pyct.NDArrayShape = None,
    lipschitz: pyct.Real = np.inf,
    diff_lipschitz: pyct.Real = np.inf,
    vectorize: pyct.VarName = frozenset(),
    batch_size: typ.Optional[int] = None,
    jit: bool = False,
    dtype: typ.Optional[pyct.DType] = None,
    enable_warnings: bool = True,
    name: str = "TorchOp",
    meta: typ.Any = None,
    **kwargs,
) -> pyct.OpT:
    r"""
    Define an :py:class:`~pycsou.abc.operator.Operator` from Torch-based callables.

    Parameters
    ----------
    apply: Callable
        A Python function with single-element Tensor input/output. Defines the :py:meth:`~pycsou.abc.operator.Map.apply`
        method of the operator: ``apply(x)==op.apply(x)``.
    shape: pyct.OpShape
        (N,M) shape of the operator, where N and M are the sizes of the output and input Tensors of ``apply`` respectively.
    arg_shape: pyct.NDArrayShape | None
        Optional shape of the input Tensor for N-D inputs.
    out_shape: pyct.NDArrayShape | None
        Optional shape of the output Tensor for N-D inputs.
    cls: pyct.OpT
        Pycsou abstract base class to instantiate from.
    lipschitz: float
        Lipschitz constant of the operator (if known).
    diff_lipschitz: float
        Diff-Lipschitz constant of the operator (if known).
    vectorize: pyct.VarName
        Arithmetic methods to vectorize.

        `vectorize` is useful if an arithmetic method provided to `kwargs` does not support stacking
        dimensions.
    batch_size: int | None
        If None (default), vectorized methods are applied as a single map over all inputs.
        If not None, then compute the vectorized methods `batch_size` samples at a time.
        Note that ``batch_size=1`` is equivalent to computing the vectorization with a for-loop.
        If you run into memory issues applying your vectorized methods over stacked inputs, try a non-None `batch_size`.
    jit: bool
        Currently has no effect (for future-compatibility only). In the future, if ``True``, then Torch-backed arithmetic methods
        will be JIT-compiled for better performance.
    dtype: pyct.DType | None
        Assumed `dtype` of the Torch-defined arithmetic methods. If `None` the arithmetic methods are assumed precision-agnostic.
    enable_warnings: bool
        If ``True``, emit warnings in case of precision/zero-copy issues.
    name: str
        Name of the operator.
    meta: Any
        Meta information to be provided as tail to :py:class:`~pycsou.abc.operator.Operator._expr`.
    kwargs: dict
        Optional   (k[str], v[value | callable]) pairs to use as arithmetic methods.

        Keys are restricted to the following arithmetic methods:

        .. code-block:: python3

          grad(), prox(), pinv(), adjoint()  # methods

        Omitted arithmetic attributes/methods default to those provided by `cls`, or are
        auto-inferred via auto-diff rules.

    Returns
    -------
    op: pyct.OpT
        (N, M) Pycsou-compliant operator.

    Notes
    -----
    * If provided, arithmetic methods must abide exactly to the interface below:

      .. code-block:: python3

         def apply(arr: torch.Tensor) -> torch.Tensor
         def grad(arr: torch.Tensor) -> torch.Tensor
         def adjoint(arr: torch.Tensor) -> torch.Tensor
         def prox(arr: torch.Tensor, tau: pyct.Real) -> torch.Tensor
         def pinv(arr: torch.Tensor, damp: pyct.Real) -> torch.Tensor

      Moreover, the methods above MUST accept an optional stacking/batching dimension as first dimension for ``arr``.
      If this does not hold, consider populating `vectorize`. If `arg_shape` and/or `out_shape` are not `None` then
      the arithmetic methods `apply, grad, prox` are supposed to have N-dimensional inputs with shape `arg_shape` and/or
      N-dimensional outputs with shape `out_shape` on their core (i.e., non-stacked) dimensions. For `adjoint, pinv` the roles
      of `arg_shape` and `out_shape` are switched since the latter are backward maps
      (i.e., `out_shape` specifies the shape of the input and `arg_shape` the shape of the output).

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`torch.vmap`. See the `PyTorch documentation <https://pytorch.org/docs/stable/func.ux_limitations.html#vmap-limitations>`_
      for known limitations.

    * All arithmetic methods provided in `kwargs` are decorated using
      :py:func:`~pycsou.runtime.enforce_precision` to abide by Pycsou's FP-runtime semantics.
      Note however that Torch does not allow mixed-precision computation, so this wrapper will coerce the input if its precision
      does not comply with the specified `dtype`. This triggers a warning by default, which can be silenced via the `enable_warnings`
      argument.

    * Arithmetic methods are **not currently JIT-ed** and that even if `jit` is set to ``True``.
      This is because of the undocumented and currently poor interaction between :py:mod:`torch.func` transforms
      and :py:func:`torch.compile`. See `this issue <https://github.com/pytorch/pytorch/issues/98822>`_ for additional details.

    * For :py:class:`~pycsou.abc.operator.DiffMap` (or subclasses thereof) the methods :py:meth:`~pycsou.abc.operator.DiffMap.jacobian`,
      :py:meth:`~pycsou.abc.operator.DiffFunc.grad` and :py:meth:`~pycsou.abc.operator.DiffFunc.adjoint` [*]_ are defined implicitly
      if not provided using the auto-differentiation transforms from the module `torch.func <https://pytorch.org/docs/stable/func.html>`_
      of PyTorch. As detailed `on this page <https://pytorch.org/docs/stable/func.ux_limitations.html>`_, such transforms work
      well on pure functions (that is, functions where the output is completely determined by the input and that do not
      involve side effects like mutation), but may fail on more complex functions. Moreover, the ``torch.func`` module does not
      yet have full coverage over PyTorch operations. For functions that calls a ``torch.nn.Module``
      `see here for some utilities <https://pytorch.org/docs/stable/func.api.html#utilities-for-working-with-torch-nn-modules>`_.

    .. [*] For a linear operator ``L`` we have ``L.jacobian(arr)==L`` for any input ``arr``. Given its apply method, the
           adjoint of ``L`` can hence be computed via automatic-differentiation as ``L.adjoint(arr) = L.jacobian(arr).adjoint(arr)``.

    .. Warning::

        Operators created with this wrapper do not support Dask inputs for now.
    """
    if isinstance(vectorize, str):
        vectorize = (vectorize,)
    vectorize = frozenset(vectorize)

    src = _FromTorch(
        apply=apply,
        shape=shape,
        cls=cls,
        arg_shape=arg_shape,
        out_shape=out_shape,
        lipschitz=lipschitz,
        diff_lipschitz=diff_lipschitz,
        vectorize=vectorize,
        batch_size=batch_size,
        jit=jit,
        dtype=dtype,
        enable_warnings=enable_warnings,
        name=name,
        meta=meta,
        **kwargs,
    )
    op = src.op()
    return op


# if __name__ == "__main__":
#
#     import time as t
#
#     import cupy as cp
#     import numpy as np
#     import torch
#
#     import pycsou.abc.operator as pyco
#     import pycsou.runtime as pycrt
#     from pycsou.operator.interop.torch import *
#
#     xp = cp
#     batch_size, in_size, out_size = 200, 400, 300
#     m = torch.nn.Linear(in_size, out_size)
#     device = {cp: "cuda", np: "cpu"}
#     if xp == cp:
#         m = m.cuda()
#
#     op = from_torch(
#         apply=lambda x: m(x),
#         shape=(out_size, in_size),
#         cls=pyco.DiffMap,
#         jit=False,
#         dtype=pycrt.Width.SINGLE.value,
#         # vectorize="apply",
#         batch_size=1,
#         name="TorchOp",
#         meta=m,
#     )
#     arr = xp.ones((batch_size, in_size), dtype=np.float32)
#     arr_t = torch.ones((batch_size, in_size), dtype=torch.float32, device=device[xp])
#     with pycrt.Precision(pycrt.Width.SINGLE):
#         t1 = t.time()
#         y1 = op(arr)
#         print(f"{t.time()-t1} seconds ellapsed (Pycsou wrapper)")
#     t1 = t.time()
#     y2 = m(arr_t)
#     print(f"{t.time() - t1} seconds ellapsed (Pytorch)")
#     assert xp.allclose(y1, asarray(y2), atol=1e-4)
#
#     with pycrt.Precision(pycrt.Width.SINGLE):
#         jac = op.jacobian(arr[0])
#
#     tangents = xp.eye(in_size, dtype=np.float32)
#     cotangents = xp.eye(out_size, dtype=np.float32)
#
#     with pycrt.Precision(pycrt.Width.SINGLE):
#         jac_mat = jac.asarray(xp=xp, dtype=np.float32)
#         jac_batch = jac.apply(arr)
#     assert xp.allclose(jac_mat, asarray(m.weight), atol=1e-4)
#
#     with pycrt.Precision(pycrt.Width.SINGLE):
#         jac_matT = jac.adjoint(cotangents)
#     assert xp.allclose(jac_matT, asarray(m.weight), atol=1e-4)
