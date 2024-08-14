import functools
import types
import warnings

import packaging.version as pkgv

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu

jax = pxu.import_module("jax", fail_on_error=False)
if jax is None:
    import typing as typ

    JaxArray = typ.TypeVar("JaxArray", bound="jax.Array")
else:
    version = pkgv.Version(jax.__version__)
    supported = pxd.JAX_SUPPORT
    assert supported["min"] <= version < supported["max"]

    JaxArray = jax.Array
    import jax.numpy as jnp


__all__ = [
    "from_jax",
]


def _from_jax(
    x: JaxArray,
    xp: pxt.ArrayModule = None,
) -> pxt.NDArray:
    """
    JAX -> NumPy/CuPy conversion.

    The transform is always zero-copy, but it is not easy to check this condition for all array types (contiguous,
    views, etc.) and backends (NUMPY, CUPY).

    [More info] https://github.com/google/jax/issues/1961#issuecomment-875773326
    """
    N = pxd.NDArrayInfo  # shorthand

    if xp is None:
        xp = N.default().module()

    if xp not in (N.NUMPY.module(), N.CUPY.module()):
        raise pxw.BackendWarning("Only NumPy/CuPy inputs are supported.")

    y = xp.asarray(x)
    return y


def _to_jax(x: pxt.NDArray, enable_warnings: bool = True) -> JaxArray:
    """
    NumPy/CuPy -> JAX conversion.

    Conversion is zero-copy when possible, i.e. 16-byte alignment, on the right device, etc.

    [More info] https://github.com/google/jax/issues/4486#issuecomment-735842976
    """
    N = pxd.NDArrayInfo  # shorthand
    W, cW = pxrt.Width, pxrt.CWidth  # shorthand

    ndi = N.from_obj(x)
    if ndi == N.DASK:
        raise pxw.BackendWarning("DASK inputs are unsupported.")

    supported_dtype = set(w.value for w in W) | set(w.value for w in cW)
    if x.dtype not in supported_dtype:
        msg = "For safety reasons, _to_jax() only accepts pyxu.runtime.[C]Width-supported dtypes."
        raise pxw.PrecisionWarning(msg)

    xp = ndi.module()
    if ndi == N.NUMPY:
        dev_type = "cpu"
        f_wrap = jnp.asarray
    elif ndi == N.CUPY:
        dev_type = "gpu"
        x = xp.require(x, requirements="C")  # JAX-DLPACK only supports contiguous arrays [2023.04.05]
        f_wrap = jnp.from_dlpack
    else:
        raise ValueError("Unknown NDArray category.")
    dev = jax.devices(dev_type)[0]
    with jax.default_device(dev):
        y = f_wrap(x)

    same_dtype = x.dtype == y.dtype
    same_mem = xp.byte_bounds(x)[0] == y.addressable_data(0).unsafe_buffer_pointer()
    if not (same_dtype and same_mem) and enable_warnings:
        msg = "\n".join(
            [
                "_to_jax(): a zero-copy conversion did not take place.",
                "[More info] https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision",
                "[More info] https://github.com/google/jax/issues/4486#issuecomment-735842976",
            ]
        )
        warnings.warn(msg, pxw.PrecisionWarning)
    return y


def from_jax(
    cls: pxt.OpC,
    dim_shape: pxt.NDArrayShape,
    codim_shape: pxt.NDArrayShape,
    vectorize: pxt.VarName = frozenset(),
    jit: bool = False,
    enable_warnings: bool = True,
    **kwargs,
) -> pxt.OpT:
    r"""
    Define an :py:class:`~pyxu.abc.Operator` from JAX functions.

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
        If ``True``, JIT-compile JAX-backed arithmetic methods for better performance.
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

         def apply(arr: jax.Array) -> jax.Array                  # (..., M1,...,MD) -> (..., N1,...,NK)
         def grad(arr: jax.Array) -> jax.Array                   # (..., M1,...,MD) -> (..., M1,...,MD)
         def adjoint(arr: jax.Array) -> jax.Array                # (..., N1,...,NK) -> (..., M1,...,MD)
         def prox(arr: jax.Array, tau: pxt.Real) -> jax.Array    # (..., M1,...,MD) -> (..., M1,...,MD)
         def pinv(arr: jax.Array, damp: pxt.Real) -> jax.Array   # (..., N1,...,NK) -> (..., M1,...,MD)

      Moreover, the methods above **must** accept stacking dimensions in ``arr``.  If this does not hold, consider
      populating `vectorize`.

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`jax.numpy.vectorize`.

    * Note that JAX enforces `32-bit arithmetic
      <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`_ by default,
      and this constraint cannot be changed at runtime.  As such, to allow zero-copy transfers between JAX and
      NumPy/CuPy arrays, it is advised to perform computations in single-precision mode.

    * Inferred arithmetic methods are not JIT-ed by default since the operation is error-prone depending on how
      :py:meth:`~pyxu.abc.Map.apply` is defined.  If :py:meth:`~pyxu.abc.Map.apply` supplied to
      :py:func:`~pyxu.operator.interop.from_jax` is JIT-friendly, then consider enabling `jit`.

    Examples
    --------
    Create the custom differential map :math:`f: \mathbb{R}^{2} \to \mathbb{R}^{3}`:

    .. math::

       f(x, y) =
       \left[
           \sin(x) + \cos(y),
           \cos(x) - \sin(y),
           \sin(x) + \cos(x)
       \right]

    .. code-block:: python3

       import pyxu.abc as pxa
       import pyxu.runtime as pxrt
       import pyxu.operator.interop as pxi
       import jax, jax.numpy as jnp
       import numpy as np

       @jax.jit
       def j_apply(arr: jax.Array) -> jax.Array:
           x, y = arr[0], arr[1]
           o1 = jnp.sin(x) + jnp.cos(y)
           o2 = jnp.cos(x) - jnp.sin(y)
           o3 = jnp.sin(x) + jnp.cos(x)
           out = jnp.r_[o1, o2, o3]
           return out

       op = pxi.from_jax(
           cls=pxa.DiffMap,
           dim_shape=2,
           codim_shape=3,
           vectorize="apply",  # j_apply() does not work on stacked inputs
                               # --> let JAX figure it out automatically.
           apply=j_apply,
       )

       rng = np.random.default_rng(0)
       x = rng.normal(size=(5,3,4,2))
       y1 = op.apply(x)  # (5,3,4,3)

       x = rng.normal(size=(2,))
       opJ = op.jacobian(x)  # JAX auto-infers the Jacobian for you.

       v = rng.normal(size=(5,2))
       w = rng.normal(size=(4,3))
       y2 = opJ.apply(v)  # (5,3)
       y3 = opJ.adjoint(w)  # (4,2)
    """
    if isinstance(vectorize, str):
        vectorize = (vectorize,)
    vectorize = frozenset(vectorize)

    src = _FromJax(
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


class _FromJax(px_src._FromSource):
    # supported methods in __init__(**kwargs)
    _meth = frozenset({"apply", "grad", "prox", "pinv", "adjoint"})

    def __init__(  # See from_jax() for a detailed description.
        self,
        cls: pxt.OpC,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
        vectorize: frozenset[str],
        jit: bool,
        enable_warnings: bool,
        **kwargs,
    ):
        super().__init__(
            cls=cls,
            dim_shape=dim_shape,
            codim_shape=codim_shape,
            embed=dict(),  # jax-funcs are state-free.
            vectorize=vectorize,
            **kwargs,
        )

        self._jit = jit
        self._enable_warnings = enable_warnings

        # Only a subset of arithmetic methods allowed from from_jax().
        if not (set(self._kwargs) <= self._meth):
            msg_head = "Unsupported arithmetic methods:"
            unsupported = set(self._kwargs) - self._meth
            msg_tail = ", ".join([f"{name}()" for name in unsupported])
            raise ValueError(f"{msg_head} {msg_tail}")

    def op(self) -> pxt.OpT:
        # Idea: modify `**kwargs` from constructor to [when required]:
        #   1. auto-define omitted methods.            [_infer_missing()]
        #   2. auto-vectorize via vmap().              [_auto_vectorize()]
        #   3. JIT & JAX<>NumPy/CuPy conversions.      [_interface()]
        self._infer_missing()
        self._auto_vectorize()
        j_state, kwargs = self._interface()

        _op = px_src.from_source(
            cls=self._op.__class__,
            dim_shape=self._op.dim_shape,
            codim_shape=self._op.codim_shape,
            embed=dict(
                _jax=j_state,
                _enable_warnings=self._enable_warnings,
                _jit=self._jit,
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
        # At the end of _infer_missing(), all jax-funcs required for _interface() have been added to `_kwargs`.
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

            def f_grad(arr: JaxArray) -> JaxArray:
                f = self._kwargs["apply"]
                y, f_vjp = jax.vjp(f, arr)
                v = jnp.ones_like(y)
                out = f_vjp(v)[0]  # f_vjp() returns a tuple
                return out

            self._vectorize.add("grad")
            self._kwargs["grad"] = f_grad

        non_selfadj = all(  # linear, but not self-adjoint
            [
                self._op.has(pxa.Property.LINEAR),
                not self._op.has(pxa.Property.LINEAR_SELF_ADJOINT),
            ]
        )
        if non_selfadj and ("adjoint" not in self._kwargs):

            def f_adjoint(arr: JaxArray) -> JaxArray:
                f = self._kwargs["apply"]
                x = jnp.zeros_like(arr, shape=self._op.dim_shape)
                _, f_vjp = jax.vjp(f, x)
                out = f_vjp(arr)[0]  # f_vjp() returns a tuple
                return out

            self._vectorize.add("adjoint")
            self._kwargs["adjoint"] = f_adjoint

        self._vectorize = frozenset(self._vectorize)

    def _auto_vectorize(self):
        # Vectorize user-specified [or _infer_missing()-added] arithmetic methods via jax.vmap().
        #
        # Modifies `_kwargs` to hold vectorized jax-funcs.
        d_sh = ",".join([f"m{i}" for i in range(self._op.dim_rank)])  # dim_shape
        cd_sh = ",".join([f"n{i}" for i in range(self._op.codim_rank)])  # codim_shape
        vkwargs = dict(  # kwargs to jax.numpy.vectorize()
            apply=dict(signature=f"({d_sh})->({cd_sh})"),
            adjoint=dict(signature=f"({cd_sh})->({d_sh})"),
            grad=dict(signature=f"({d_sh})->({d_sh})"),
            prox=dict(signature=f"({d_sh})->({d_sh})", excluded={1}),
            pinv=dict(signature=f"({cd_sh})->({d_sh})", excluded={1}),
        )

        for name in self._kwargs:
            if name in self._vectorize:
                func = self._kwargs[name]  # necessarily jax_func
                vectorize = functools.partial(
                    jax.numpy.vectorize,
                    **vkwargs[name],
                )
                self._kwargs[name] = vectorize(func)

    def _interface(self):
        # Arithmetic methods supplied in `kwargs`:
        #
        #     * take `jax.Array` inputs
        #     * do not have the `self` parameter. (Reason: to be JIT-compatible.)
        #
        # This method creates modified arithmetic functions to match Pyxu's API, and `state` required for them to work.
        #
        # Returns
        # -------
        # j_state: dict
        #     Jax functions referenced by wrapper arithmetic methods. (See below.)
        #     These functions are JIT-compiled if specified.
        # kwargs: dict
        #     Pyxu-compatible functions which can be submitted to interop.from_source().
        j_state = dict()
        for name, obj in self._kwargs.items():
            if name in self._meth:
                if self._jit:
                    obj = jax.jit(obj)
                j_state[name] = obj  # necessarily jax_func

        kwargs = dict()
        for name, obj in self._kwargs.items():
            func = getattr(self.__class__, name)
            kwargs[name] = func  # necessarily a pyxu_func

        # Special cases.
        # (Reason: not in `_kwargs` [c.f. from_jax() docstring], but need to be inferred.)
        for name in ("jacobian", "_quad_spec", "asarray", "_expr"):
            kwargs[name] = getattr(self.__class__, name)

        return j_state, kwargs

    # Wrapper arithmetic methods ----------------------------------------------
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)
        func = self._jax["apply"]
        dev = j_arr.devices().pop()
        with jax.default_device(dev):
            j_out = func(j_arr)
        out = _from_jax(j_out, xp=pxu.get_array_module(arr))
        return out

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)
        func = self._jax["grad"]
        dev = j_arr.devices().pop()
        with jax.default_device(dev):
            j_out = func(j_arr)
        out = _from_jax(j_out, xp=pxu.get_array_module(arr))
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)
        func = self._jax["adjoint"]
        dev = j_arr.devices().pop()
        with jax.default_device(dev):
            j_out = func(j_arr)
        out = _from_jax(j_out, xp=pxu.get_array_module(arr))
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)
        func = self._jax["prox"]
        dev = j_arr.devices().pop()
        with jax.default_device(dev):
            j_out = func(j_arr, tau)  # positional args only if auto-vectorized.
        out = _from_jax(j_out, xp=pxu.get_array_module(arr))
        return out

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)
        func = self._jax["pinv"]
        dev = j_arr.devices().pop()
        with jax.default_device(dev):
            j_out = func(j_arr, damp)  # positional args only if auto-vectorized.
        out = _from_jax(j_out, xp=pxu.get_array_module(arr))
        return out

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        try:
            # Use the class' method if available ...
            klass = self.__class__
            op = klass.jacobian(self, arr)
        except NotImplementedError:
            # ... and fallback to auto-inference if undefined.
            f = self._jax["apply"]
            j_arr = _to_jax(arr, enable_warnings=self._enable_warnings)

            # define forward: [1] explains why jvp() is a better fit than linearize().
            # [1] https://jax.readthedocs.io/en/latest/_autosummary/jax.linearize.html
            _fwd = functools.partial(jax.jvp, f, (j_arr,))
            f_fwd = lambda arr: _fwd((arr,))[1]  # jax returns a tuple

            # define adjoint: [2] explains benefits of linear_transpose() over vjp().
            # [2] https://jax.readthedocs.io/en/latest/_autosummary/jax.linear_transpose.html
            hint = types.SimpleNamespace(shape=self.dim_shape, dtype=arr.dtype)
            _adj = jax.linear_transpose(f_fwd, hint)
            f_adj = lambda arr: _adj(arr)[0]  # jax returns a tuple

            klass = pxa.LinFunc if (self.codim_shape == (1,)) else pxa.LinOp
            op = from_jax(
                cls=klass,
                dim_shape=self.dim_shape,
                codim_shape=self.codim_shape,
                vectorize=("apply", "adjoint"),
                jit=self._jit,
                enable_warnings=self._enable_warnings,
                apply=f_fwd,
                adjoint=f_adj,
            )
        return op

    def _quad_spec(self):
        if self.has(pxa.Property.QUADRATIC):
            # auto-infer (Q, c, t)
            def _grad(arr: JaxArray) -> JaxArray:
                # Just like jax.grad(f)(arr), but works with (1,)-valued functions.
                # [jax.grad(f) expects scalar outputs.]
                f = self._jax["apply"]
                y, f_vjp = jax.vjp(f, arr)
                v = jnp.ones_like(y)
                out = f_vjp(v)[0]  # f_vjp() returns a tuple
                return out

            # vectorize & JIT internal function
            d_sh = ",".join([f"m{i}" for i in range(self._op.dim_rank)])  # dim_shape
            _grad = jnp.vectorize(_grad, signature=f"({d_sh})->({d_sh})")
            if self._jit:
                _grad = jax.jit(_grad)

            def Q_apply(arr: JaxArray) -> JaxArray:
                # \grad_{f}(x) = Q x + c = Q x + \grad_{f}(0)
                # ==> Q x = \grad_{f}(x) - \grad_{f}(0)
                z = jnp.zeros_like(arr)
                out = _grad(arr) - _grad(z)
                return out

            def c_apply(arr: JaxArray) -> JaxArray:
                z = jnp.zeros_like(arr)
                c = _grad(z)
                out = jnp.sum(c * arr)[jnp.newaxis]
                return out

            Q = from_jax(
                cls=pxa.PosDefOp,
                dim_shape=self.dim_shape,
                codim_shape=self.dim_shape,
                vectorize="apply",
                jit=self._jit,
                enable_warnings=self._enable_warnings,
                apply=Q_apply,
            )
            c = from_jax(
                cls=pxa.LinFunc,
                dim_shape=self.dim_shape,
                codim_shape=1,
                vectorize="apply",
                jit=self._jit,
                enable_warnings=self._enable_warnings,
                apply=c_apply,
            )

            # `t` can be computed using any backend, so we choose NUMPY.
            f = self._jax["apply"]
            with jax.default_device(jax.devices("cpu")[0]):
                t = float(f(jnp.zeros(self.dim_shape)))

            return (Q, c, t)
        else:
            raise NotImplementedError

    def asarray(self, **kwargs) -> pxt.NDArray:
        if self.has(pxa.Property.LINEAR):
            # JAX operators don't accept DASK inputs: cannot call Lin[Op,Func].asarray() with user-specified `xp` value.
            # -> We arbitrarily perform computations using the NUMPY backend, then cast as needed.
            N = pxd.NDArrayInfo  # shorthand
            dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
            xp = kwargs.get("xp", N.default().module())

            klass = self.__class__
            A = klass.asarray(self, dtype=dtype, xp=N.NUMPY.module())

            # Not the most efficient method, but fail-proof
            B = xp.array(A, dtype=dtype)
            return B
        else:
            raise NotImplementedError

    def _expr(self) -> tuple:
        # show which arithmetic methods are backed by jax-funcs
        jax_funcs = tuple(self._jax.keys())
        return ("from_jax", *jax_funcs)
