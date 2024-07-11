import collections.abc as cabc
import types

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.util as pxu

__all__ = [
    "from_source",
]


def from_source(
    cls: pxt.OpC,
    dim_shape: pxt.NDArrayShape,
    codim_shape: pxt.NDArrayShape,
    embed: dict = None,
    vectorize: pxt.VarName = frozenset(),
    **kwargs,
) -> pxt.OpT:
    r"""
    Define an :py:class:`~pyxu.abc.Operator` from low-level constructs.

    Parameters
    ----------
    cls: OpC
        Operator sub-class to instantiate.
    dim_shape: NDArrayShape
        Operator domain shape (M1,...,MD).
    codim_shape: NDArrayShape
        Operator co-domain shape (N1,...,NK).
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

        `vectorize` is useful if an arithmetic method provided to `kwargs` (ex: :py:meth:`~pyxu.abc.Map.apply`) does not
        support stacking dimensions.

    Returns
    -------
    op: OpT
        Pyxu-compliant operator :math:`A: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1}
        \times\cdots\times N_{K}}`.

    Notes
    -----
    * If provided, arithmetic methods must abide exactly to the Pyxu interface.  In particular, the following arithmetic
      methods, if supplied, **must** have the following interface:

      .. code-block:: python3

         def apply(self, arr: pxt.NDArray) -> pxt.NDArray                   # (..., M1,...,MD) -> (..., N1,...,NK)
         def grad(self, arr: pxt.NDArray) -> pxt.NDArray                    # (..., M1,...,MD) -> (..., M1,...,MD)
         def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray                 # (..., N1,...,NK) -> (..., M1,...,MD)
         def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray     # (..., M1,...,MD) -> (..., M1,...,MD)
         def pinv(self, arr: pxt.NDArray, damp: pxt.Real) -> pxt.NDArray    # (..., N1,...,NK) -> (..., M1,...,MD)

      Moreover, the methods above **must** accept stacking dimensions in ``arr``.  If this does not hold, consider
      populating `vectorize`.

    * Auto-vectorization consists in decorating `kwargs`-specified arithmetic methods with
      :py:func:`~pyxu.util.vectorize`.  Auto-vectorization may be less efficient than explicitly providing a vectorized
      implementation.

    Examples
    --------
    Creation of the custom element-wise differential operator :math:`f(\mathbf{x}) = \mathbf{x}^{2}`.

    .. code-block:: python3

       N = 5
       f = from_source(
           cls=pyxu.abc.DiffMap,
           dim_shape=N,
           codim_shape=N,
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
           dim_shape=N,
           codim_shape=N,
           embed=dict(
               # special form to set (diff-)Lipschitz attributes via from_source()
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

    src = _FromSource(
        cls=cls,
        dim_shape=dim_shape,
        codim_shape=codim_shape,
        embed=embed,
        vectorize=vectorize,
        **kwargs,
    )
    op = src.op()
    return op


class _FromSource:  # See from_source() for a detailed description.
    def __init__(
        self,
        cls: pxt.OpC,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
        embed: dict,
        vectorize: frozenset[str],
        **kwargs,
    ):
        from pyxu.abc.operator import _core_operators

        assert cls in _core_operators(), f"Unknown Operator type: {cls}."
        self._op = cls(  # ensure shape well-formed
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )

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

        # Add-on vectorization functionality.
        self._vkwargs = self._parse_vectorize(vectorize)
        self._vectorize = vectorize

    def op(self) -> pxt.OpT:
        _op = self._op  # shorthand
        for p in _op.properties():
            for name in p.arithmetic_methods():
                if func := self._kwargs.get(name, False):
                    # vectorize() do NOT kick in for default-provided methods.
                    # (We assume they are Pyxu-compliant from the start.)

                    if name in self._vectorize:
                        decorate = pxu.vectorize(**self._vkwargs[name])
                        func = decorate(func)

                    setattr(_op, name, types.MethodType(func, _op))

        # Embed extra attributes
        for name, attr in self._embed.items():
            setattr(_op, name, attr)

        return _op

    def _parse_vectorize(self, vectorize: frozenset[str]):
        vkwargs = dict(  # Parameter hints for vectorize()
            apply=dict(
                i="arr",  # Pyxu arithmetic methods broadcast along parameter `arr`.
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
            ),
            prox=dict(
                i="arr",
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.dim_shape,
            ),
            grad=dict(
                i="arr",
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.dim_shape,
            ),
            adjoint=dict(
                i="arr",
                dim_shape=self._op.codim_shape,
                codim_shape=self._op.dim_shape,
            ),
            pinv=dict(
                i="arr",
                dim_shape=self._op.codim_shape,
                codim_shape=self._op.dim_shape,
            ),
        )

        if not (vectorize <= set(vkwargs)):  # un-recognized arithmetic method
            msg_head = "Can only vectorize arithmetic methods"
            msg_tail = ", ".join([f"{name}()" for name in vkwargs])
            raise ValueError(f"{msg_head} {msg_tail}")
        return vkwargs
