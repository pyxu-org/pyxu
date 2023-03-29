import types
import typing as typ

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
    # START arg-parsing =======================================================
    if embed is None:
        embed = dict()

    # compute vectorize() kwargs per arithmetic method ------------------------
    codim, dim = shape
    vsize = dict(  # `codim` hint for vectorize()
        apply=codim,
        prox=dim,
        grad=dim,
        adjoint=dim,
        pinv=dim,
    )
    vmethod_default = pycu.parse_params(
        pycu.vectorize,
        i="bogus",  # doesn't matter
    )["method"]

    if isinstance(vectorize, str):
        vectorize = (vectorize,)
    if not (set(vectorize) <= set(vsize)):  # un-recognized arithmetic method
        msg_head = "Can only vectorize arithmetic methods"
        msg_tail = ", ".join([f"{name}()" for name in vsize])
        raise ValueError(f"{msg_head} {msg_tail}")
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
    # -------------------------------------------------------------------------

    # compute enforce_precision() kwargs per arithmetic method ----------------
    if isinstance(enforce_precision, str):
        enforce_precision = (enforce_precision,)
    if not (set(enforce_precision) <= set(vsize)):
        msg_head = "Can only enforce precision on arithmetic methods"
        msg_tail = ", ".join([f"{name}()" for name in vsize])
        raise ValueError(f"{msg_head} {msg_tail}")
    ekwargs = dict(  # Pycsou arithmetic methods enforce FP-precision along these parameters.
        apply=dict(i="arr"),
        prox=dict(i=("arr", "tau")),
        grad=dict(i="arr"),
        adjoint=dict(i="arr"),
        pinv=dict(i=("arr", "damp")),
    )
    # -------------------------------------------------------------------------

    # END arg-parsing =========================================================
    op = cls(shape=shape)
    for p in op.properties():
        for name in p.arithmetic_attributes():
            attr = kwargs.get(name, getattr(op, name))
            setattr(op, name, attr)
        for name in p.arithmetic_methods():
            if name in kwargs:
                func = kwargs[name]
                if name in vectorize:
                    decorate = pycu.vectorize(**vkwargs[name])
                    func = decorate(func)
                if name in enforce_precision:
                    decorate = pycrt.enforce_precision(**ekwargs[name])
                    func = decorate(func)
            else:
                # vectorize() & enforce_precision() do NOT kick in for default-provided methods.
                # (We assume they are Pycsou-compliant from the start.)
                func = getattr(cls, name)
            setattr(op, name, types.MethodType(func, op))
    for (name, attr) in embed.items():
        setattr(op, name, attr)
    return op
