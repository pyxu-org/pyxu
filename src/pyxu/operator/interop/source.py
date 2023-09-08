import collections.abc as cabc
import types
import typing as typ

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "from_source",
]


def from_source(
    cls: pxt.OpC,
    shape: pxt.OpShape,
    embed: dict = None,
    vectorize: pxt.VarName = frozenset(),
    vmethod: typ.Union[str, dict] = None,
    enforce_precision: pxt.VarName = frozenset(),
    **kwargs,
) -> pxt.OpT:
    r"""
    Define an :py:class:`~pyxu.abc.Operator` from low-level constructs.

    Parameters
    ----------
    cls: OpC
        Operator sub-class to instantiate.
    shape: OpShape
        (N, M) operator shape.
    embed: dict
        (k[str], v[value]) pairs to embed into the created operator.

        `embed` is useful to attach extra information to synthesized :py:class:`~pyxu.abc.Operator` used by arithmetic
        methods.
    kwargs: dict
        (k[str], v[callable]) pairs to use as arithmetic methods.

        Keys must be entries from :py:meth:`~pyxu.abc.Property.arithmetic_methods`.

        Omitted arithmetic attributes/methods default to those provided by `cls`.
    vectorize: VarName
        Arithmetic methods to vectorize.

        `vectorize` is useful if an arithmetic method provided to `kwargs` (ex: :py:meth:`~pyxu.abc.Map.apply`):

        * does not support stacking dimensions; OR
        * does not support DASK inputs.
    vmethod: str, dict
        Vectorization strategy. (See :py:class:`~pyxu.util.vectorize`.)

        Different strategies can be applied per arithmetic method via a dictionary.
    enforce_precision: VarName
        Arithmetic methods to make compliant with Pyxu's runtime FP-precision.

        `enforce_precision` is useful if an arithmetic method provided to `kwargs` (ex: :py:meth:`~pyxu.abc.Map.apply`)
        is unaware of Pyxu's runtime FP context.

    Returns
    -------
    op: OpT
        (N, M) Pyxu-compliant operator.

    Notes
    -----
    * If provided, arithmetic methods must abide exactly to the Pyxu interface.  In particular, the following arithmetic
      methods, if supplied, **must** have the following interface:

      .. code-block:: python3

         def apply(self, arr: pxt.NDArray) -> pxt.NDArray
         def grad(self, arr: pxt.NDArray) -> pxt.NDArray
         def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray
         def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray
         def pinv(self, arr: pxt.NDArray, damp: pxt.Real) -> pxt.NDArray

      Moreover, the methods above **must** accept ``(..., M)``-shaped inputs for ``arr``.  If this does not hold,
      consider populating `vectorize`.

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`~pyxu.util.vectorize`.  Auto-vectorization may be less efficient than explicitly providing a vectorized
      implementation.

    * Enforcing precision consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`~pyxu.runtime.enforce_precision`.  Not all arithmetic methods can be made runtime FP-precision
      compliant.  It is thus recommended to make arithmetic methods precision-compliant manually.

    Examples
    --------
    Creation of the custom element-wise differential operator :math:`f(\mathbf{x}) = \mathbf{x}^{2}`.

    .. code-block:: python3

       N = 5
       f = from_source(
           cls=pyxu.abc.DiffMap,
           shape=(N, N),
           apply=lambda _, arr: arr**2,
       )
       x = np.arange(N)
       y = f(x)  # [0, 1, 4, 9, 16]
       dL = f.diff_lipschitz  # inf (default value provided by DiffMap class.)

    In practice we know that :math:`f` has a finite-valued diff-Lipschitz constant.  It is thus recommended to set it
    too when instantiating via ``from_source``:

    .. code-block:: python3

       N = 5
       f = from_source(
           cls=pyxu.abc.DiffMap,
           shape=(N, N),
           embed=dict(  # special form to set (diff-)Lipschitz attributes via from_source()
               _diff_lipschitz=2,
           ),
           apply=lambda _, arr: arr**2,
       )
       x = np.arange(N)
       y = f(x)  # [0, 1, 4, 9, 16]
       dL = f.diff_lipschitz  # 2  <- instead of inf
    """
    if embed is None:
        embed = dict()

    if isinstance(vectorize, str):
        vectorize = (vectorize,)
    vectorize = frozenset(vectorize)

    if isinstance(enforce_precision, str):
        enforce_precision = (enforce_precision,)
    enforce_precision = frozenset(enforce_precision)

    src = _FromSource(
        cls=cls,
        shape=shape,
        embed=embed,
        vectorize=vectorize,
        vmethod=vmethod,
        enforce_precision=enforce_precision,
        **kwargs,
    )
    op = src.op()
    return op


class _FromSource:  # See from_source() for a detailed description.
    def __init__(
        self,
        cls: pxt.OpC,
        shape: pxt.OpShape,
        embed: dict,
        vectorize: frozenset[str],
        vmethod: typ.Union[str, dict],
        enforce_precision: frozenset[str],
        **kwargs,
    ):
        from pyxu.abc.operator import _core_operators

        assert cls in _core_operators(), f"Unknown Operator type: {cls}."
        self._op = cls(shape)  # ensure shape well-formed

        # Arithmetic methods to attach to `_op`.
        meth = frozenset.union(*[p.arithmetic_methods() for p in pxa.Property])
        if not (set(kwargs) <= meth):
            msg_head = "Unknown arithmetic methods:"
            unknown = set(kwargs) - meth
            msg_tail = ", ".join([f"{name}()" for name in unknown])
            raise ValueError(f"{msg_head} {msg_tail}")
        self._kwargs = kwargs

        # Extra attributes to attach to `_op`.
        assert isinstance(embed, cabc.Mapping)
        self._embed = embed

        # Add-on functionality to enable.
        self._vkwargs = self._parse_vectorize(vectorize, vmethod)
        self._vectorize = vectorize

        self._ekwargs = self._parse_precision(enforce_precision)
        self._enforce_fp = enforce_precision

    def op(self) -> pxt.OpT:
        _op = self._op  # shorthand
        for p in _op.properties():
            for name in p.arithmetic_methods():
                if func := self._kwargs.get(name, False):
                    # vectorize() & enforce_precision() do NOT kick in for default-provided methods.
                    # (We assume they are Pyxu-compliant from the start.)

                    if name in self._vectorize:
                        decorate = pxu.vectorize(**self._vkwargs[name])
                        func = decorate(func)

                    if name in self._enforce_fp:
                        decorate = pxrt.enforce_precision(**self._ekwargs[name])
                        func = decorate(func)

                    setattr(_op, name, types.MethodType(func, _op))

        # Embed extra attributes
        for name, attr in self._embed.items():
            setattr(_op, name, attr)

        return _op

    def _parse_vectorize(
        self,
        vectorize: frozenset[str],
        vmethod: typ.Union[str, dict],
    ):
        vsize = dict(  # `codim` hint for vectorize()
            apply=self._op.codim,
            prox=self._op.dim,
            grad=self._op.dim,
            adjoint=self._op.dim,
            pinv=self._op.dim,
        )

        if not (vectorize <= set(vsize)):  # un-recognized arithmetic method
            msg_head = "Can only vectorize arithmetic methods"
            msg_tail = ", ".join([f"{name}()" for name in vsize])
            raise ValueError(f"{msg_head} {msg_tail}")

        vmethod_default = pxu.parse_params(
            pxu.vectorize,
            i="bogus",  # doesn't matter
        )["method"]
        if vmethod is None:
            vmethod = vmethod_default
        if isinstance(vmethod, str):
            vmethod = {name: vmethod for name in vsize}

        vkwargs = {
            name: dict(
                i="arr",  # Pyxu arithmetic methods broadcast along parameter `arr`.
                method=vmethod.get(name, vmethod_default),
                codim=vsize[name],
            )
            for name in vsize
        }
        return vkwargs

    def _parse_precision(self, enforce_precision: frozenset[str]):
        ekwargs = dict(
            # Pyxu arithmetic methods enforce FP-precision along these parameters.
            apply=dict(i="arr"),
            prox=dict(i=("arr", "tau")),
            grad=dict(i="arr"),
            adjoint=dict(i="arr"),
            pinv=dict(i=("arr", "damp")),
            svdvals=dict(),
            trace=dict(),
        )

        if not (enforce_precision <= set(ekwargs)):
            msg_head = "Can only enforce precision on arithmetic methods"
            msg_tail = ", ".join([f"{name}()" for name in ekwargs])
            raise ValueError(f"{msg_head} {msg_tail}")
        return ekwargs
