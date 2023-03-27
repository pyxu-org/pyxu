import types

import pycsou.util.ptype as pyct

__all__ = [
    "from_source",
]


def from_source(
    cls: pyct.OpC,
    shape: pyct.OpShape,
    embed: dict = None,
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
    if embed is None:
        embed = dict()

    op = cls(shape=shape)
    for p in op.properties():
        for name in p.arithmetic_attributes():
            attr = kwargs.get(name, getattr(op, name))
            setattr(op, name, attr)
        for name in p.arithmetic_methods():
            func = kwargs.get(name, getattr(cls, name))
            setattr(op, name, types.MethodType(func, op))
    for (name, attr) in embed.items():
        setattr(op, name, attr)
    return op
