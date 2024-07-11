import typing as typ
import warnings
from functools import wraps

import numpy as np
import packaging.version as pkgv

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
from pyxu.util import import_module

torch = import_module("torch", fail_on_error=False)
if torch is not None:
    version = pkgv.Version(torch.__version__)
    supported = pxd.PYTORCH_SUPPORT
    assert supported["min"] <= version < supported["max"]

    import torch._dynamo as dynamo
    import torch.func as functorch

    TorchTensor = torch.Tensor
else:
    TorchTensor = typ.TypeVar("torch.Tensor")

__all__ = [
    "from_torch",
]


def _traceable(f):
    # Needed to compile functorch transforms. See this issue: https://github.com/pytorch/pytorch/issues/98822
    f = dynamo.allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def _to_torch(arr: pxt.NDArray, requires_grad: bool = False) -> TorchTensor:
    r"""
    NumPy/CuPy -> PyTorch conversion.

    Convert a NumPy-like array into a PyTorch tensor, sharing data, dtype and device.

    Parameters
    ----------
    arr: NDArray
        Input array.
    requires_grad: bool
        If autograd should record operations on the returned tensor.

    Returns
    -------
    tensor: torch.Tensor
        Output tensor.

    Notes
    -----
    The returned tensor and input array share the same memory.  Modifications to the tensor will be reflected in the
    ndarray and vice versa.  The returned tensor is not resizable.
    """
    if pxd.NDArrayInfo.from_obj(arr) == pxd.NDArrayInfo.CUPY:
        with torch.device("cuda", arr.device.id):
            return torch.as_tensor(arr).requires_grad_(requires_grad)
    else:
        return torch.from_numpy(arr).requires_grad_(requires_grad)


def _from_torch(tensor: TorchTensor) -> pxt.NDArray:
    r"""
    PyTorch -> NumPy/CuPy conversion.

    Convert a PyTorch tensor into a NumPy-like array, sharing data, dtype and device.

    Parameters
    ----------
    tensor: torch.Tensor
        Input tensor.

    Returns
    -------
    arr: NDArray
        Output array.

    Notes
    -----
    The returned array and input tensor share the same memory.  Modifications to the tensor will be reflected in the
    ndarray and vice versa.
    """
    if tensor.get_device() == -1:
        return tensor.detach().numpy(force=False)
    else:
        cp = pxd.NDArrayInfo.CUPY.module()
        with cp.cuda.Device(tensor.get_device()):
            return cp.asarray(tensor.detach())


def from_torch(
    cls: pxt.OpC,
    dim_shape: pxt.NDArrayShape,
    codim_shape: pxt.NDArrayShape,
    vectorize: pxt.VarName = frozenset(),
    jit: bool = False,
    enable_warnings: bool = True,
    **kwargs,
) -> pxt.OpT:
    r"""
    Define an :py:class:`~pyxu.abc.Operator` from PyTorch functions.

    Parameters
    ----------
    cls: OpC
        Operator sub-class to instantiate.
    dim_shape: NDArrayShape
        Operator domain shape (M1,...,MD).
    codim_shape: NDArrayShape
        Operator co-domain shape (N1,...,NK).
    kwargs: dict
        (k[str], v[callable]) pairs to use as arithmetic methods.

        Keys are restricted to the following arithmetic methods:

        .. code-block:: python3

           apply(), grad(), prox(), pinv(), adjoint()

        Omitted arithmetic methods default to those provided by `cls`, or are auto-inferred via auto-diff rules.
    vectorize: VarName
        Arithmetic methods to vectorize.

        `vectorize` is useful if an arithmetic method provided to `kwargs` does not support stacking dimensions.
    jit: bool
        Currently has no effect (for future-compatibility only). In the future, if ``True``, then Torch-backed
        arithmetic methods will be JIT-compiled for better performance.
    enable_warnings: bool
        If ``True``, emit warnings in case of precision/zero-copy issues.

    Returns
    -------
    op: OpT
        Pyxu-compliant operator :math:`A: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1}
        \times\cdots\times N_{K}}`.

     Notes
    -----
    * If provided, arithmetic methods must abide exactly to the interface below:

      .. code-block:: python3

         def apply(arr: torch.Tensor) -> torch.Tensor                  # (..., M1,...,MD) -> (..., N1,...,NK)
         def grad(arr: torch.Tensor) -> torch.Tensor                   # (..., M1,...,MD) -> (..., M1,...,MD)
         def adjoint(arr: torch.Tensor) -> torch.Tensor                # (..., N1,...,NK) -> (..., M1,...,MD)
         def prox(arr: torch.Tensor, tau: pxt.Real) -> torch.Tensor    # (..., M1,...,MD) -> (..., M1,...,MD)
         def pinv(arr: torch.Tensor, damp: pxt.Real) -> torch.Tensor   # (..., N1,...,NK) -> (..., M1,...,MD)

      Moreover, the methods above **must** accept stacking dimensions in ``arr``.  If this does not hold, consider
      populating `vectorize`.

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with :py:func:`torch.vmap`.  See
      the `PyTorch documentation <https://pytorch.org/docs/stable/func.ux_limitations.html#vmap-limitations>`_ for known
      limitations.

    * Arithmetic methods are **not currently JIT-ed** even if `jit` is set to ``True``.  This is because of the
      undocumented and currently poor interaction between :py:mod:`torch.func` transforms and :py:func:`torch.compile`.
      See `this issue <https://github.com/pytorch/pytorch/issues/98822>`_ for additional details.

    * For :py:class:`~pyxu.abc.DiffMap` (or subclasses thereof), the methods :py:meth:`~pyxu.abc.DiffMap.jacobian`,
      :py:meth:`~pyxu.abc.DiffFunc.grad` and :py:meth:`~pyxu.abc.LinOp.adjoint` are defined implicitly if not provided
      using the auto-differentiation transforms from :py:mod:`torch.func`.  As detailed `on this page
      <https://pytorch.org/docs/stable/func.ux_limitations.html>`_, such transforms work well on pure functions (that
      is, functions where the output is completely determined by the input and that do not involve side effects like
      mutation), but may fail on more complex functions.  Moreover, :py:mod:`torch.func` does not yet have full coverage
      over PyTorch operations.  For functions that call a :py:class:`torch.nn.Module`, see `here
      <https://pytorch.org/docs/stable/func.api.html#utilities-for-working-with-torch-nn-modules>`_ for some utilities.

    .. Warning::

       Operators created with this wrapper do not support Dask inputs for now.
    """
    if isinstance(vectorize, str):
        vectorize = (vectorize,)
    vectorize = frozenset(vectorize)

    src = _FromTorch(
        cls=cls,
        dim_shape=dim_shape,
        codim_shape=codim_shape,
        vectorize=vectorize,
        jit=bool(jit),
        enable_warnings=bool(enable_warnings),
        **kwargs,
    )
    op = src.op()
    return op


class _FromTorch(px_src._FromSource):
    # supported methods in __init__(**kwargs)
    _meth = frozenset({"apply", "grad", "prox", "pinv", "adjoint"})

    def __init__(  # See from_torch() for a detailed description.
        self,
        cls: pxt.OpC,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
        vectorize: frozenset[str],
        jit: bool,  # Unused for now
        enable_warnings: bool,
        **kwargs,
    ):
        self._batch_size = kwargs.pop("batch_size", None)
        self._dtype = kwargs.pop("dtype", None)
        super().__init__(
            cls=cls,
            dim_shape=dim_shape,
            codim_shape=codim_shape,
            embed=dict(),
            vectorize=vectorize,
            **kwargs,
        )

        self._jit = False  # JIT-compilation is currently deactivated until torch.func goes out of beta.
        self._enable_warnings = enable_warnings

        # Only a subset of arithmetic methods allowed from from_torch().
        if not (set(self._kwargs) <= self._meth):
            msg_head = "Unsupported arithmetic methods:"
            unsupported = set(self._kwargs) - self._meth
            msg_tail = ", ".join([f"{name}()" for name in unsupported])
            raise ValueError(f"{msg_head} {msg_tail}")

    def op(self) -> pxt.OpT:
        # Idea: modify `**kwargs` from constructor to [when required]:
        #   1. auto-define omitted methods.            [_infer_missing()]
        #   2. auto-vectorize via vmap().              [_auto_vectorize()]
        #   3. JIT-compile via compile().              [_compile()]
        #   4. TORCH<>NumPy/CuPy conversions.          [_interface()]
        #   Note: JIT-compilation is currently deactivated due to the undocumented interaction of torch.func transforms
        #   and torch.compile. Will be reactivated once torch.func goes out of beta.

        self._infer_missing()
        self._compile()
        self._auto_vectorize()
        t_state, kwargs = self._interface()

        _op = px_src.from_source(
            cls=self._op.__class__,
            dim_shape=self._op.dim_shape,
            codim_shape=self._op.codim_shape,
            embed=dict(
                _batch_size=self._batch_size,
                _dtype=self._dtype,
                _jit=self._jit,
                _enable_warnings=self._enable_warnings,
                _coerce=self._coerce,
                _torch=t_state,
            ),
            # vectorize=None,  # see top-level comment.
            **kwargs,
        )
        return _op

    def _infer_missing(self):
        # The following methods must be auto-inferred if missing from `kwargs`:
        #
        #     grad(), adjoint()
        #
        # Missing methods are auto-inferred via auto-diff rules and added to `_kwargs`.
        # At the end of _infer_missing(), all torch-funcs required for _interface() have been added to `_kwargs`.
        #
        # Notes
        # -----
        # This method does NOT produce vectorized implementations: _auto_vectorize() is responsible for this.
        self._vectorize = set(self._vectorize)  # to allow updates below

        nl_difffunc = all(  # non-linear diff-func
            [
                self._op.has(pxa.Property.DIFFERENTIABLE_FUNCTION),
                not self._op.has(pxa.Property.LINEAR),
            ]
        )
        if nl_difffunc and ("grad" not in self._kwargs):
            apply_no_vec = self._copy_function(self._kwargs["apply"])

            def f_grad(tensor: TorchTensor) -> TorchTensor:
                apply = lambda tnsr: torch.squeeze(apply_no_vec(tnsr))
                grad = functorch.grad(apply)
                return grad(tensor)

            self._vectorize.add("grad")
            self._kwargs["grad"] = f_grad

        non_selfadj = all(  # linear, but not self-adjoint
            [
                self._op.has(pxa.Property.LINEAR),
                not self._op.has(pxa.Property.LINEAR_SELF_ADJOINT),
            ]
        )
        if non_selfadj and ("adjoint" not in self._kwargs):
            apply_no_vec = self._copy_function(self._kwargs["apply"])

            def f_adjoint(tensor: TorchTensor) -> TorchTensor:
                primal = torch.zeros(self._op.dim_shape, dtype=tensor.dtype)
                _, f_vjp = functorch.vjp(apply_no_vec, primal)
                return f_vjp(tensor)[0]  # f_vjp returns a tuple

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
        # This method creates modified arithmetic functions to match Pyxu's API, and `state` required for them to work.
        #
        # Returns
        # -------
        # t_state: dict
        #     Torch functions referenced by wrapper arithmetic methods. (See below.)
        # kwargs: dict
        #     Pyxu-compatible functions which can be submitted to interop.from_source().

        # torch_func's (potentially auto_vec/jitted at this stage)
        t_state = {name: obj for name, obj in self._kwargs.items() if name in self._meth}

        # Pyxu-compatible functions
        kwargs = {name: getattr(self.__class__, name) for name in self._kwargs}

        # Special cases.
        # (Reason: not in `_kwargs` [c.f. from_torch() docstring], but need to be inferred.)
        for name in ("jacobian", "_quad_spec", "_expr", "asarray"):
            kwargs[name] = getattr(self.__class__, name)

        return t_state, kwargs

    def _coerce(self, arr: pxt.NDArray) -> pxt.NDArray:
        # Coerce inputs (and raise a warning) in case of precision mis-matches.
        if self._dtype is not None and (arr.dtype != self._dtype):
            if self._enable_warnings:
                msg = f"Precision mis-match! Input array was coerced to {self._dtype} precision automatically."
                warnings.warn(msg, pxw.PrecisionWarning)
            arr = arr.astype(self._dtype, copy=False)
        return arr

    # Wrapper arithmetic methods ----------------------------------------------
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        bsh = arr.shape[: -self.dim_rank]
        bsize = (int(np.prod(bsh)),)
        arr = arr.reshape(bsize + self.dim_shape)
        arr = self._coerce(arr)
        tensor = _to_torch(arr)
        func = self._torch["apply"]
        out = _from_torch(func(tensor))
        return out.reshape(bsh + self.codim_shape)

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        bsh = arr.shape[: -self.dim_rank]
        bsize = (int(np.prod(bsh)),)
        arr = arr.reshape(bsize + self.dim_shape)
        arr = self._coerce(arr)
        tensor = _to_torch(arr)
        func = self._torch["grad"]
        out = _from_torch(func(tensor))
        return out.reshape(bsh + self.dim_shape)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        bsh = arr.shape[: -self.codim_rank]
        bsize = (int(np.prod(bsh)),)
        arr = arr.reshape(bsize + self.codim_shape)
        arr = self._coerce(arr)
        tensor = _to_torch(arr)
        func = self._torch["adjoint"]
        out = _from_torch(func(tensor))
        return out.reshape(bsh + self.dim_shape)

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        bsh = arr.shape[: -self.dim_rank]
        bsize = (int(np.prod(bsh)),)
        arr = arr.reshape(bsize + self.dim_shape)
        arr = self._coerce(arr)
        tensor = _to_torch(arr)
        func = self._torch["prox"]
        out = _from_torch(func(tensor, tau))
        return out.reshape(bsh + self.dim_shape)

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        bsh = arr.shape[: -self.codim_rank]
        bsize = (int(np.prod(bsh)),)
        arr = arr.reshape(bsize + self.codim_shape)
        arr = self._coerce(arr)
        tensor = _to_torch(arr)
        func = self._torch["pinv"]
        out = func(tensor, damp)  # positional args only if auto-vectorized.
        out = _from_torch(out)
        return out.reshape(bsh + self.dim_shape)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        try:
            # Use the class' method if available ...
            klass = self.__class__
            op = klass.jacobian(self, arr)
        except NotImplementedError:
            # ... and fallback to auto-inference if undefined.
            # bsh = arr.shape[:-self.codim_rank]
            # bsize = (int(np.prod(bsh)), )
            # arr = arr.reshape(bsize + self.codim_shape)
            arr = self._coerce(arr)
            primal = _to_torch(arr)
            f = self._torch["apply"]

            def jf_apply(tan: TorchTensor) -> TorchTensor:
                return functorch.jvp(f, primals=(primal,), tangents=(tan,))[1]

            def jf_adjoint(cotan: TorchTensor) -> TorchTensor:
                _, f_vjp = functorch.vjp(f, primal)
                return f_vjp(cotan)[0]  # f_vjp returns a tuple

            klass = pxa.LinFunc if (self.codim_shape == (1,)) else pxa.LinOp
            op = from_torch(
                cls=klass,
                dim_shape=self.dim_shape,
                codim_shape=self.codim_shape,
                vectorize=("apply", "adjoint"),
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
                apply=jf_apply,
                adjoint=jf_adjoint,
            )
        return op

    def _quad_spec(self):
        if self.has(pxa.Property.QUADRATIC):
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
                out = torch.sum(c * tensor, dtype=tensor.dtype).unsqueeze(-1)
                return out

            Q = from_torch(
                apply=Q_apply,
                dim_shape=self.dim_shape,
                codim_shape=self.dim_shape,
                cls=pxa.PosDefOp,
                vectorize="apply",
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
            )

            c = from_torch(
                apply=c_apply,
                dim_shape=self.dim_shape,
                codim_shape=1,
                cls=pxa.LinFunc,
                vectorize="apply",
                batch_size=self._batch_size,
                jit=self._jit,
                dtype=self._dtype,
                enable_warnings=self._enable_warnings,
            )

            # We cannot know a-priori which backend the supplied torch-apply() function works with.
            # Consequence: to compute `t`, we must try different backends until one works.
            f = self._torch["apply"]
            t = self._compute_t(f)

            return (Q, c, t)
        else:
            raise NotImplementedError

    def _compute_t(self, f):
        try:
            with torch.device("cpu"):
                return float(f(torch.zeros(self.dim_shape)))
        except Exception:
            for id in range(torch.cuda.device_count()):
                try:
                    with torch.device("gpu", id):
                        return float(f(torch.zeros(self.dim_shape)))
                except Exception:
                    continue
        raise RuntimeError("Failed to compute t")

    def asarray(self, **kwargs) -> pxt.NDArray:
        if self.has(pxa.Property.LINEAR):
            # Torch operators don't accept DASK inputs: cannot call Lin[Op,Func].asarray() with user-specified `xp` value.
            # -> We arbitrarily perform computations using the NUMPY backend, then cast as needed.
            N = pxd.NDArrayInfo  # shorthand
            dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
            xp = kwargs.get("xp", N.default().module())

            klass = self.__class__
            A = klass.asarray(self, dtype=dtype, xp=N.NUMPY.module())

            # Not the most efficient method, but fail-proof
            return xp.array(A, dtype=dtype)
        else:
            raise NotImplementedError

    def _expr(self):
        torch_funcs = list(self._torch.keys())
        return ("from_torch", *torch_funcs)

    def _copy_function(self, fn):
        # Create a copy of a function's code, defaults, and closure.
        # This method is necessary to create a separate instance of a function, preserving its
        # original properties, while allowing modifications to the copied instance without affecting
        # the original. This is particularly useful for functions that need to be wrapped or altered
        # dynamically within the class.

        import types

        new_fn = types.FunctionType(
            fn.__code__, fn.__globals__, name=fn.__name__, argdefs=fn.__defaults__, closure=fn.__closure__
        )
        new_fn.__doc__ = fn.__doc__
        new_fn.__annotations__ = fn.__annotations__
        new_fn.__dict__.update(fn.__dict__)
        return new_fn
