import collections.abc as cabc
import types
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "from_source",
]


def from_source(
    cls: pyct.OpC,
    shape: pyct.OpShape,
    embed: dict = None,
    vectorize: pyct.VarName = frozenset(),
    vmethod: typ.Union[str, dict] = None,
    enforce_precision: pyct.VarName = frozenset(),
    **kwargs,
) -> pyct.OpT:
    r"""
    Define an :py:class:`~pycsou.abc.operator.Operator` from low-level constructs.

    Parameters
    ----------
    cls: pyct.OpC
        Operator sub-class to instantiate.
    shape: pyct.OpShape
        (N, M) operator shape.
    embed: dict
        (k[str], v[value]) pairs to embed into the created operator.

        `embed` is useful to attach extra information to `op` to be used by arithmetic methods.
    kwargs: dict
        ``(k[str], v[value | callable])`` pairs to use as attributes and methods.

        Keys must be entries from ``cls.Property.arithmetic_[attributes,methods]()``.

        If provided, then ``op.<k> = v``.

        Omitted arithmetic attributes/methods default to those provided by ``cls(shape)``.
    vectorize: pyct.VarName
        Arithmetic methods to vectorize.

        `vectorize` is useful if an arithmetic method provided to `kwargs` (ex:
        :py:meth:`~pycsou.abc.operator.Map.apply`):

        * does not support stacking dimensions; OR
        * does not support DASK inputs.
    vmethod: str | dict
        Vectorization strategy. (See :py:class:`~pycsou.util.operator.vectorize`.)

        Different strategies can be applied per arithmetic method via a dictionary.
    enforce_precision: pyct.VarName
        Arithmetic methods to make compliant with Pycsou's runtime FP-precision.

        `enforce_precision` is useful if an arithmetic method provided to `kwargs` (ex:
        :py:meth:`~pycsou.abc.Map.apply`) does not comply with Pycsou's runtime FP-precision.

    Returns
    -------
    op: pyct.OpT
        (N, M) Pycsou-compliant operator.

    Notes
    -----
    * If provided, arithmetic methods must abide exactly to the Pycsou interface, i.e.:
      :py:meth:`~pycsou.abc.operator.Map.apply`,
      :py:meth:`~pycsou.abc.operator.ProxFunc.prox`,
      :py:meth:`~pycsou.abc.operator.DiffFunc.grad`,
      :py:meth:`~pycsou.abc.operator.LinOp.adjoint`, and
      :py:meth:`~pycsou.abc.operator.LinOp.pinv`
      must accept ``(..., M)``-shaped inputs for ``arr``.
      If this does not hold, consider populating `vectorize`.

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`~pycsou.util.operator.vectorize`.
      Auto-vectorization may be less efficient than explicitly providing a vectorized arithmetic
      method.

    * Enforcing precision consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`~pycsou.runtime.enforce_precision`.
      Not all arithmetic methods can be made runtime FP-precision compliant.
      It is thus recommended to make arithmetic methods precision-compliant manually.

    Examples
    --------
    Creation of the custom element-wise differential operator :math:`f(\mathbf{x}) = \mathbf{x}^{2}`.

    .. code-block:: python3

       N = 5
       f = from_source(
           cls=pycsou.abc.DiffMap,
           shape=(N, N),
           apply=lambda _, arr: arr**2,
       )
       x = np.arange(N)
       y = f(x)  # [0, 1, 4, 9, 16]
       L = f.diff_lipschitz()  # inf (default value provided by DiffMap class.)

    In practice we know that :math:`f` has a finite-valued diff-Lipschitz constant.
    It is thus recommended to set it too when instantiating via ``from_source``:

    .. code-block:: python3

       N = 5
       f = from_source(
           cls=pycsou.abc.DiffMap,
           shape=(N, N),
           apply=lambda _, arr: arr**2,
           diff_lipschitz=lambda _, **kwargs: 2,
       )
       x = np.arange(N)
       y = f(x)  # [0, 1, 4, 9, 16]
       L = f.diff_lipschitz()  # 2  <- instead of inf
    """
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
        cls: pyct.OpC,
        shape: pyct.OpShape,
        embed: dict = None,
        vectorize: pyct.VarName = frozenset(),
        vmethod: typ.Union[str, dict] = None,
        enforce_precision: pyct.VarName = frozenset(),
        **kwargs,
    ):
        assert cls in pyco._core_operators(), f"Unknown Operator type: {cls}."
        self._op = cls(shape)

        # Arithmetic attributes/methods to attach to `_op` in op()
        attr = frozenset.union(*[p.arithmetic_attributes() for p in pyco.Property])
        meth = frozenset.union(*[p.arithmetic_methods() for p in pyco.Property])
        assert set(kwargs) <= attr | meth, "Unknown arithmetic attributes/methods provided."
        self._kwargs = kwargs

        # Extra attributes to attach to `_op` in op()
        if embed is None:
            self._embed = dict()
        else:
            assert isinstance(embed, cabc.Mapping)
            self._embed = embed

        # Add-on functionality to enable in op()
        self._vectorize, self._vkwargs = self._parse_vectorize(vectorize, vmethod)
        self._enforce_fp, self._ekwargs = self._parse_precision(enforce_precision)

    def op(self) -> pyct.OpT:
        _op = self._op  # shorthand
        for p in _op.properties():
            for name in p.arithmetic_attributes():
                attr = self._kwargs.get(name, getattr(_op, name))
                setattr(_op, name, attr)
            for name in p.arithmetic_methods():
                if func := self._kwargs.get(name, False):
                    # vectorize() & enforce_precision() do NOT kick in for default-provided methods.
                    # (We assume they are Pycsou-compliant from the start.)

                    if name in self._vectorize:
                        decorate = pycu.vectorize(**self._vkwargs[name])
                        func = decorate(func)

                    if name in self._enforce_fp:
                        decorate = pycrt.enforce_precision(**self._ekwargs[name])
                        func = decorate(func)

                    setattr(_op, name, types.MethodType(func, _op))

        # Embed extra attributes
        for (name, attr) in self._embed.items():
            setattr(_op, name, attr)

        return _op

    def _parse_vectorize(
        self,
        vectorize: pyct.VarName,
        vmethod: typ.Union[str, dict],
    ):
        vsize = dict(  # `codim` hint for vectorize()
            apply=self._op.codim,
            prox=self._op.dim,
            grad=self._op.dim,
            adjoint=self._op.dim,
            pinv=self._op.dim,
        )

        if isinstance(vectorize, str):
            vectorize = (vectorize,)
        if not (set(vectorize) <= set(vsize)):  # un-recognized arithmetic method
            msg_head = "Can only vectorize arithmetic methods"
            msg_tail = ", ".join([f"{name}()" for name in vsize])
            raise ValueError(f"{msg_head} {msg_tail}")
        vectorize = tuple(vectorize)

        vmethod_default = pycu.parse_params(
            pycu.vectorize,
            i="bogus",  # doesn't matter
        )["method"]
        if vmethod is None:
            vmethod = vmethod_default
        if isinstance(vmethod, str):
            vmethod = {name: vmethod for name in vsize}

        vkwargs = {
            name: dict(
                i="arr",  # Pycsou arithmetic methods broadcast along parameter `arr`.
                method=vmethod.get(name, vmethod_default),
                codim=vsize[name],
            )
            for name in vsize
        }
        return vectorize, vkwargs

    def _parse_precision(self, enforce_precision: pyct.VarName):
        ekwargs = dict(
            # Pycsou arithmetic methods enforce FP-precision along these parameters.
            apply=dict(i="arr"),
            prox=dict(i=("arr", "tau")),
            grad=dict(i="arr"),
            adjoint=dict(i="arr"),
            pinv=dict(i=("arr", "damp")),
        )

        if isinstance(enforce_precision, str):
            enforce_precision = (enforce_precision,)
        if not (set(enforce_precision) <= set(ekwargs)):
            msg_head = "Can only enforce precision on arithmetic methods"
            msg_tail = ", ".join([f"{name}()" for name in vsize])
            raise ValueError(f"{msg_head} {msg_tail}")
        enforce_precision = tuple(enforce_precision)

        return enforce_precision, ekwargs
