import collections.abc as cabc
import types

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.util as pxu

__all__ = [
    "stack",
    "block_diag",
]


def stack(ops: cabc.Sequence[pxt.OpT]) -> pxt.OpT:
    r"""
    Map operators over the same input.

    A stacked operator :math:`S: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{Q \times N_{1}
    \times\cdots\times N_{K}} is an operator containing (vertically) :math:`Q` blocks of smaller operators :math:`\{
    O_{q}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1} \times\cdots\times N_{K}} \}_{q=1}^{Q}`:

    .. math::

       S
       =
       \left[
           \begin{array}{c}
               O_{1}  \\
               \vdots \\
               O_{Q}  \\
           \end{array}
       \right]

    Each sub-operator :math:`O_{q}` acts on the same input and returns parallel outputs which get stacked along the
    zero-th axis.

    Parameters
    ----------
    ops: :py:class:`~collections.abc.Sequence` ( :py:attr:`~pyxu.info.ptype.OpT` )
        (Q,) identically-shaped operators to map over inputs.

    Returns
    -------
    op: OpT
        Stacked (M1,...,MD) -> (Q, N1,...,NK) operator.

    Examples
    --------

    .. code-block:: python3

       import pyxu.operator as pxo
       import numpy as np

       op = pxo.Sum((3, 4), axis=-1)  # (3,4) -> (3,1)
       A = pxo.stack([op, 2*op])  # (3,4) -> (2,3,1)

       x = np.arange(A.dim_size).reshape(A.dim_shape)  # [[ 0  1  2  3]
                                                       #  [ 4  5  6  7]
                                                       #  [ 8  9 10 11]]
       y = A.apply(x)  # [[[ 6.]
                       #   [22.]
                       #   [38.]]
                       #
                       #  [[12.]
                       #   [44.]
                       #   [76.]]]


    See Also
    --------
    :py:func:`~pyxu.operator.block_diag`
    """
    op = _Stack(ops).op()
    return op


def block_diag(ops: cabc.Sequence[pxt.OpT]) -> pxt.OpT:
    r"""
    Zip operators over parallel inputs.

    A block-diagonal operator :math:`B: \mathbb{R}^{Q \times M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{Q \times
    N_{1} \times\cdots\times N_{K}}` is an operator containing (diagonally) :math:`Q` blocks of smaller operators
    :math:`\{ O_{q}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1} \times\cdots\times N_{K}}
    \}_{q=1}^{Q}`:

    .. math::

       B
       =
       \left[
           \begin{array}{ccc}
               O_{1} &        &       \\
                     & \ddots &       \\
                     &        & O_{Q} \\
           \end{array}
       \right]

    Each sub-operator :math:`O_{q}` acts on the :math:`q`-th slice of the inputs along the zero-th axis.

    Parameters
    ----------
    ops: :py:class:`~collections.abc.Sequence` ( :py:attr:`~pyxu.info.ptype.OpT` )
        (Q,) identically-shaped operators to zip over inputs.

    Returns
    -------
    op: OpT
        Block-diagonal (Q, M1,...,MD) -> (Q, N1,...,NK) operator.

    Examples
    --------

    .. code-block:: python3

       import pyxu.operator as pxo
       import numpy as np

       op = pxo.Sum((3, 4), axis=-1)  # (3,4) -> (3,1)
       A = pxo.block_diag([op, 2*op])  # (2,3,4) -> (2,3,1)

       x = np.arange(A.dim_size).reshape(A.dim_shape)  # [[[ 0  1  2  3]
                                                       #   [ 4  5  6  7]
                                                       #   [ 8  9 10 11]]
                                                       #
                                                       #  [[12 13 14 15]
                                                       #   [16 17 18 19]
                                                       #   [20 21 22 23]]]
       y = A.apply(x)  # [[[  6.]
                       #   [ 22.]
                       #   [ 38.]]
                       #
                       #  [[108.]
                       #   [140.]
                       #   [172.]]]


    See Also
    --------
    :py:func:`~pyxu.operator.stack`
    """
    op = _BlockDiag(ops).op()
    return op


class _BlockDiag:
    # See block_diag() docstrings.
    def __init__(self, ops: cabc.Sequence[pxt.OpT]):
        dim_shape = ops[0].dim_shape
        codim_shape = ops[0].codim_shape

        shape_msg = "All operators must have same dim/codim."
        assert all(_op.dim_shape == dim_shape for _op in ops), shape_msg
        assert all(_op.codim_shape == codim_shape for _op in ops), shape_msg

        self._ops = list(ops)

    def op(self) -> pxt.OpT:
        klass = self._infer_op_klass()
        N_op = len(self._ops)
        dim_shape = self._ops[0].dim_shape
        codim_shape = self._ops[0].codim_shape
        op = klass(
            dim_shape=(N_op, *dim_shape),
            codim_shape=(N_op, *codim_shape),
        )
        op._ops = self._ops  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(op, name, types.MethodType(func, op))
        self._propagate_constants(op)
        return op

    def _infer_op_klass(self) -> pxt.OpC:
        base = {
            pxa.Property.CAN_EVAL,
            pxa.Property.DIFFERENTIABLE,
            pxa.Property.LINEAR,
            pxa.Property.LINEAR_SQUARE,
            pxa.Property.LINEAR_NORMAL,
            pxa.Property.LINEAR_IDEMPOTENT,
            pxa.Property.LINEAR_SELF_ADJOINT,
            pxa.Property.LINEAR_POSITIVE_DEFINITE,
            pxa.Property.LINEAR_UNITARY,
        }
        properties = set.intersection(
            base,
            *[_op.properties() for _op in self._ops],
        )
        klass = pxa.Operator._infer_operator_type(properties)
        return klass

    @staticmethod
    def _propagate_constants(op: pxt.OpT):
        # Propagate (diff-)Lipschitz constants forward via special call to
        # Rule()-overridden `estimate_[diff_]lipschitz()` methods.

        # Important: we write to _[diff_]lipschitz to not overwrite estimate_[diff_]lipschitz() methods.
        if op.has(pxa.Property.CAN_EVAL):
            op._lipschitz = op.estimate_lipschitz(__rule=True)
        if op.has(pxa.Property.DIFFERENTIABLE):
            op._diff_lipschitz = op.estimate_diff_lipschitz(__rule=True)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        N_stack = len(arr.shape[: -self.dim_rank])
        select = lambda i: (slice(None),) * N_stack + (i,)
        parts = [_op.apply(arr[select(i)]) for (i, _op) in enumerate(self._ops)]

        xp = pxu.get_array_module(arr)
        out = xp.stack(parts, axis=-self.codim_rank)
        return out

    def __call__(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        N_stack = len(arr.shape[: -self.codim_rank])
        select = lambda i: (slice(None),) * N_stack + (i,)
        parts = [_op.adjoint(arr[select(i)]) for (i, _op) in enumerate(self._ops)]

        xp = pxu.get_array_module(arr)
        out = xp.stack(parts, axis=-self.dim_rank)
        return out

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        # op.pinv(y, damp) = stack([op1.pinv(y1, damp), ..., opN.pinv(yN, damp)], axis=0)
        N_stack = len(arr.shape[: -self.codim_rank])
        select = lambda i: (slice(None),) * N_stack + (i,)
        parts = [_op.pinv(arr[select(i)], damp) for (i, _op) in enumerate(self._ops)]

        xp = pxu.get_array_module(arr)
        out = xp.stack(parts, axis=-self.dim_rank)
        return out

    def svdvals(self, **kwargs) -> pxt.NDArray:
        # op.svdvals(**kwargs) = top_k([op1.svdvals(**kwargs), ..., opN.svdvals(**kwargs)])
        parts = [_op.svdvals(**kwargs) for _op in self._ops]

        k = kwargs.get("k")
        xp = pxu.get_array_module(parts[0])
        D = xp.sort(xp.concatenate(parts))[-k:]
        return D

    def trace(self, **kwargs) -> pxt.Real:
        # op.trace(**kwargs) = sum([op1.trace(**kwargs), ..., opN.trace(**kwargs)])
        parts = [_op.trace(**kwargs) for _op in self._ops]
        tr = sum(parts)
        return tr

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxa.Property.LINEAR):
            J = self
        else:
            parts = [_op.jacobian(_arr) for (_op, _arr) in zip(self._ops, arr)]
            J = _BlockDiag(ops=parts).op()
        return J

    def asarray(self, **kwargs) -> pxt.NDArray:
        parts = [_op.asarray(**kwargs) for _op in self._ops]

        xp = pxu.get_array_module(parts[0])
        dtype = parts[0].dtype
        A = xp.zeros((*self.codim_shape, *self.dim_shape), dtype=dtype)

        select = (slice(None),) * (self.codim_rank - 1)
        for i, _A in enumerate(parts):
            A[(i,) + select + (i,)] = _A
        return A

    def gram(self) -> pxt.OpT:
        parts = [_op.gram() for _op in self._ops]
        G = _BlockDiag(ops=parts).op()
        return G

    def cogram(self) -> pxt.OpT:
        parts = [_op.cogram() for _op in self._ops]
        CG = _BlockDiag(ops=parts).op()
        return CG

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L_parts = [_op.lipschitz for _op in self._ops]
        elif self.has(pxa.Property.LINEAR):
            L = self.__class__.estimate_lipschitz(self, **kwargs)
            return L
        else:
            L_parts = [_op.estimate_lipschitz(**kwargs) for _op in self._ops]

        # [non-linear case] Upper bound: L <= max(L_k)
        L = max(L_parts)
        return L

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL_parts = [_op.diff_lipschitz for _op in self._ops]
        elif self.has(pxa.Property.LINEAR):
            dL = 0
            return dL
        else:
            dL_parts = [_op.estimate_diff_lipschitz(**kwargs) for _op in self._ops]

        # [non-linear case] Upper bound: dL <= max(dL_k)
        dL = max(dL_parts)
        return dL

    def _expr(self) -> tuple:
        return ("block_diag", *self._ops)


class _Stack:
    # See stack() docstrings.
    def __init__(self, ops: cabc.Sequence[pxt.OpT]):
        dim_shape = ops[0].dim_shape
        codim_shape = ops[0].codim_shape

        shape_msg = "All operators must have same dim/codim."
        assert all(_op.dim_shape == dim_shape for _op in ops), shape_msg
        assert all(_op.codim_shape == codim_shape for _op in ops), shape_msg

        self._ops = list(ops)

    def op(self) -> pxt.OpT:
        klass = self._infer_op_klass()
        N_op = len(self._ops)
        dim_shape = self._ops[0].dim_shape
        codim_shape = self._ops[0].codim_shape
        op = klass(
            dim_shape=dim_shape,
            codim_shape=(N_op, *codim_shape),
        )
        op._ops = self._ops  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name, None)
                if func is not None:
                    setattr(op, name, types.MethodType(func, op))
        self._propagate_constants(op)
        return op

    def _infer_op_klass(self) -> pxt.OpC:
        base = {
            pxa.Property.CAN_EVAL,
            pxa.Property.DIFFERENTIABLE,
            pxa.Property.LINEAR,
        }
        properties = set.intersection(
            base,
            *[_op.properties() for _op in self._ops],
        )
        klass = pxa.Operator._infer_operator_type(properties)
        return klass

    @staticmethod
    def _propagate_constants(op: pxt.OpT):
        # Propagate (diff-)Lipschitz constants forward via special call to
        # Rule()-overridden `estimate_[diff_]lipschitz()` methods.

        # Important: we write to _[diff_]lipschitz to not overwrite estimate_[diff_]lipschitz() methods.
        if op.has(pxa.Property.CAN_EVAL):
            op._lipschitz = op.estimate_lipschitz(__rule=True)
        if op.has(pxa.Property.DIFFERENTIABLE):
            op._diff_lipschitz = op.estimate_diff_lipschitz(__rule=True)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        parts = [_op.apply(arr) for _op in self._ops]

        xp = pxu.get_array_module(arr)
        out = xp.stack(parts, axis=-self.codim_rank)
        return out

    def __call__(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        N_stack = len(arr.shape[: -self.codim_rank])
        select = lambda i: (slice(None),) * N_stack + (i,)
        parts = [_op.adjoint(arr[select(i)]) for (i, _op) in enumerate(self._ops)]

        out = sum(parts)
        return out

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxa.Property.LINEAR):
            J = self
        else:
            parts = [_op.jacobian(arr) for _op in self._ops]
            J = _Stack(ops=parts).op()
        return J

    def asarray(self, **kwargs) -> pxt.NDArray:
        parts = [_op.asarray(**kwargs) for _op in self._ops]
        xp = pxu.get_array_module(parts[0])
        A = xp.stack(parts, axis=0)
        return A

    def gram(self) -> pxt.OpT:
        # [_ops.gram()] should be reduced (via +) to form a single operator.
        # It is inefficient however to chain so many operators together via AddRule().
        # apply() is thus redefined to improve performance.

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            G = [_op.gram() for _op in _._ops]
            parts = [_G.apply(arr) for _G in G]

            out = sum(parts)
            return out

        def op_expr(_) -> tuple:
            return ("gram", self)

        G = px_src.from_source(
            cls=pxa.SelfAdjointOp,
            dim_shape=self.dim_shape,
            codim_shape=self.dim_shape,
            embed=dict(_ops=self._ops),
            apply=op_apply,
            _expr=op_expr,
        )
        return G

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L_parts = [_op.lipschitz for _op in self._ops]
        elif self.has(pxa.Property.LINEAR):
            L = self.__class__.estimate_lipschitz(self, **kwargs)
            return L
        else:
            L_parts = [_op.estimate_lipschitz(**kwargs) for _op in self._ops]

        # [non-linear case] Upper bound: L**2 <= sum(L_k**2)
        L2 = np.r_[L_parts] ** 2
        L = np.sqrt(L2.sum())
        return L

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL_parts = [_op.diff_lipschitz for _op in self._ops]
        elif self.has(pxa.Property.LINEAR):
            dL = 0
            return dL
        else:
            dL_parts = [_op.estimate_diff_lipschitz(**kwargs) for _op in self._ops]

        # [non-linear case] Upper bound: dL**2 <= sum(dL_k**2)
        dL2 = np.r_[dL_parts] ** 2
        dL = np.sqrt(dL2.sum())
        return dL

    def _expr(self) -> tuple:
        return ("stack", *self._ops)
