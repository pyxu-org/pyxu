import collections
import collections.abc as cabc
import itertools
import types

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "stack",
    "vstack",
    "hstack",
    "block_diag",
    "block",
    "coo_block",
]


def stack(
    ops: cabc.Sequence[pyct.OpT],
    axis: pyct.Integer,
) -> pyct.OpT:
    r"""
    Construct a stacked operator.

    A stacked-operator :math:`V: \mathbb{R}^{d} \to \mathbb{R}^{c}` is an operator containing
    (vertically or horizontally) blocks of smaller operators :math:`\{O_{1}, \ldots, O_{N}\}`.

    This is a convenience function around :py:func:`~pycsou.operator.block.hstack` and
    :py:func:`~pycsou.operator.block.vstack`.

    Parameters
    ----------
    ops: cabc.Sequence[pyct.OpT]
        [op1(c1, d1), ..., opN(cN, dN)] operators to join.
    axis: pyct.Integer
        The axis along which operators will be joined, i.e.

        * 0: stack vertically (row-wise)
        * 1: stack horizontally (column-wise)

    Returns
    -------
    op: pyct.OpT
        Stacked operator.

    See Also
    --------
    :py:func:`~pycsou.operator.block.hstack`,
    :py:func:`~pycsou.operator.block.vstack`.
    """
    axis = int(axis)
    assert axis in {0, 1}, f"axis: out-of-bounds axis '{axis}'."

    f = {0: vstack, 1: hstack}[axis]
    op = f(ops)
    return op


def vstack(ops: cabc.Sequence[pyct.OpT]) -> pyct.OpT:
    r"""
    Construct a vertically-stacked operator.

    A vstacked-operator :math:`V: \mathbb{R}^{d} \to \mathbb{R}^{c_{1} + \cdots + c_{N}}` is an
    operator containing (vertically) blocks of smaller operators :math:`\{O_{1}: \mathbb{R}^{d} \to
    \mathbb{R}^{c_{1}}, \ldots, O_{N}: \mathbb{R}^{d} \to \mathbb{R}^{c_{N}}\}`, i.e.

    .. math::

       V
       =
       \left[
           \begin{array}{c}
               O_{1}  \\
               \vdots \\
               O_{N}  \\
           \end{array}
       \right]

    Parameters
    ----------
    ops: cabc.Sequence[pyct.OpT]
        [op1(c1, d), ..., opN(cN, d)] operators to concatenate.

    Returns
    -------
    op: pyct.OpT
        Vertically-stacked (c1+...+cN, d) operator.

    Notes
    -----
    * All sub-operator domains must have compatible shapes, i.e.

      * domain-agnostic operators forbidden, and
      * all integer-valued ``dim`` s must be identical.

    See Also
    --------
    :py:func:`~pycsou.operator.block.hstack`,
    :py:func:`~pycsou.operator.block.stack`.
    """

    def op_expr(_) -> tuple:
        N_row = _._grid_shape[0]
        ops = [_._block[(r, 0)] for r in range(N_row)]
        return ("vstack", *ops)

    N_data = len(ops)
    op = _COOBlock(
        ops=(
            ops,  # data
            (
                tuple(range(N_data)),  # i
                [0] * N_data,  # j
            ),
        ),
        grid_shape=(N_data, 1),
    ).op()
    op._expr = types.MethodType(op_expr, op)
    return op


def hstack(ops: cabc.Sequence[pyct.OpT]) -> pyct.OpT:
    r"""
    Construct a horizontally-stacked operator.

    A hstacked-operator :math:`H: \mathbb{R}^{d_{1} + \cdots + d_{N}} \to \mathbb{R}^{c}` is an
    operator containing (horizontally) blocks of smaller operators :math:`\{O_{1}: \mathbb{R}^{d_{1}} \to
    \mathbb{R}^{c}, \ldots, O_{N}: \mathbb{R}^{d_{N}} \to \mathbb{R}^{c}\}`, i.e.

    .. math::

       H
       =
       \left[
           \begin{array}{ccc}
               O_{1} & \cdots & O_{N}
           \end{array}
       \right]

    Parameters
    ----------
    ops: cabc.Sequence[pyct.OpT]
        [op1(c, d1), ..., opN(c, dN)] operators to concatenate.

    Returns
    -------
    op: pyct.OpT
        Horizontally-stacked (c, d1+....+dN) operator.

    Notes
    -----
    * All sub-operator domains must have compatible shapes, i.e.

      * all ``codim`` s must be identical, and
      * domain-agnostic operators forbidden.

    See Also
    --------
    :py:func:`~pycsou.operator.block.vstack`,
    :py:func:`~pycsou.operator.block.stack`.
    """

    def _infer_op_shape(sh_ops: list[pyct.Shape]) -> pyct.Shape:
        if any(_[1] == None for _ in sh_ops):
            raise ValueError("Domain-agnostic operators are unsupported.")
        assert len(set(_[0] for _ in sh_ops)) == 1, "All sub-operators must have same co-domain size."

        dim, codim = 0, sh_ops[0][0]
        for _ in sh_ops:
            dim += _[1]
        return (codim, dim)

    def _infer_op_klass(ops: list[pyct.OpT]) -> pyct.OpC:
        # CAN_EVAL: always
        # FUNCTIONAL: [op1,...,opN] all functional
        # PROXIMABLE: [op1, ..., opN] all proximable
        # DIFFERENTIABLE: [op1, ..., opN] all differentiable
        # DIFFERENTIABLE_FUNCTION: [op1, ..., opN] all differentiable functions
        # LINEAR: [op1, ..., opN] all linear
        # LINEAR_SQUARE: final shape square & [op1, ..., opN] linear
        # QUADRATIC: [op1, ..., opN] = at least one quad and (rest linear or quad)
        P = pyco.Property
        base = {
            P.CAN_EVAL,
            P.FUNCTIONAL,
            P.PROXIMABLE,
            P.DIFFERENTIABLE,
            P.DIFFERENTIABLE_FUNCTION,
            P.LINEAR,
        }
        properties = frozenset.intersection(*[op.properties() for op in ops])
        properties = set(properties & base)

        sh = _infer_op_shape([op.shape for op in ops])
        if (P.LINEAR in properties) and (sh[0] == sh[1]):
            properties.add(P.LINEAR_SQUARE)

        if any(op.has(P.QUADRATIC) for op in ops):
            # possible quadratic if all other terms are quadratic/linear.
            non_quad = [op for op in ops if not op.has(P.QUADRATIC)]
            if all(op.has(P.LINEAR) for op in non_quad):
                properties.add(P.QUADRATIC)

        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        # op.apply(x) = sum([op1.apply(x1), ..., opN.apply(xN)], axis=0)
        parts, i = [], 0
        for op in _._ops:
            p = op.apply(arr[..., i : i + op.dim])
            parts.append(p)
            i += op.dim

        out = sum(parts)
        return out

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        # op.lipschitz(**kwargs) = max([op1.lipschitz(**kwargs), ..., opN.lipschitz(**kwargs)])
        #                        + update _lipschitz
        if _.has(pyco.Property.LINEAR):
            L = _.__class__.lipschitz(_, **kwargs)
        else:
            L = max([op.lipschitz(**kwargs) for op in _._ops])
        _._lipschitz = float(L)
        return _._lipschitz

    def op_asloss(_, **kwargs) -> pyct.OpT:
        msg = "asloss() is ambiguous for hstack-ed operators."
        raise NotImplementedError(msg)

    def op_jacobian(_, arr: pyct.NDArray) -> pyct.OpT:
        # op.jacobian(x) = hstack([op1.jacobian(x1), ..., opN.jacobian(xN)])
        if not _.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if _.has(pyco.Property.LINEAR):
            out = _
        else:
            parts, i = [], 0
            for op in _._ops:
                p = op.jacobian(arr[i : i + op.dim])
                parts.append(p)
                i += op.dim

            out = hstack(parts)
        return out

    def op_diff_lipschitz(_, **kwargs) -> pyct.Real:
        # op.diff_lipschitz(**kwargs) = max([op1.diff_lipschitz(**kwargs), ..., opN.diff_lipschitz(**kwargs)])
        #                             + update _diff_lipschitz
        if not _.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if _.has(pyco.Property.LINEAR):
            dL = _.__class__.diff_lipschitz(_, **kwargs)
        else:
            dL = max([op.diff_lipschitz(**kwargs) for op in _._ops])
        _._diff_lipschitz = float(dL)
        return _._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def op_grad(_, arr: pyct.NDArray) -> pyct.NDArray:
        # op.grad(x) = concatenate([op1.grad(x1), ..., opN.grad(xN)], axis=-1)
        if not _.has(pyco.Property.DIFFERENTIABLE_FUNCTION):
            raise NotImplementedError

        parts, i = [], 0
        for op in _._ops:
            p = op.grad(arr[..., i : i + op.dim])
            parts.append(p)
            i += op.dim

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def op_prox(_, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # op.prox(x, tau) = concatenate([op1.prox(x1, tau), ..., opN.prox(xN, tau)], axis=-1)
        if not _.has(pyco.Property.PROXIMABLE):
            raise NotImplementedError

        parts, i = [], 0
        for op in _._ops:
            p = op.prox(arr=arr[..., i : i + op.dim], tau=tau)
            parts.append(p)
            i += op.dim

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    def op_hessian(_) -> pyct.OpT:
        # op._hessian() = block_diag([op1._hessian(), ..., opN._hessian()])
        #                 w/ zeros (pos-defed) if needed on diagonal.
        if not _.has(pyco.Property.QUADRATIC):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            if op.has(pyco.Property.QUADRATIC):
                p = op._hessian()
            else:  # op is necessarily LINEAR
                from pycsou.operator.linop import NullOp

                p = NullOp(shape=(op.dim, op.dim)).asop(pyco.PosDefOp)
            parts.append(p)

        H = block_diag(parts)
        return H

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        # op.adjoint(y) = concatenate([op1.adjoint(y), ..., opN.adjoint(y)], axis=-1)
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.adjoint(arr)
            parts.append(p)

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        # op.asarray(**kwargs) = concatenate([op1.asarray(**kwargs), ..., opN.asarray(**kwargs)], axis=-1)
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.asarray(**kwargs)
            parts.append(p)

        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        A = xp.concatenate(parts, axis=-1)
        return A

    def op_gram(_) -> pyct.OpT:
        # op.gram() = \diag([op1.gram(), ..., opN.gram()]) + cross-terms
        #           = constructed via coo_block()
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        def G_expr(__) -> tuple:
            return ("gram", _)

        N = len(_._ops)
        data, i, j = [], [], []
        for _i in range(N):
            for _j in range(N):
                if _i == _j:
                    d = _._ops[_i].gram()
                else:
                    d = _._ops[_i].T * _._ops[_j]
                data.append(d)
                i.append(_i)
                j.append(_j)

        G = coo_block(
            ops=(data, (i, j)),
            grid_shape=(N, N),
        ).asop(pyco.SelfAdjointOp)
        G._expr = types.MethodType(G_expr, G)
        return G

    def op_cogram(_) -> pyct.OpT:
        # op.cogram() = op1.cogram() + ... + opN.cogram()
        #
        # It is inefficient however to chain so many operators together via AddRule().
        # apply() is thus redefined to improve performance.
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        @pycrt.enforce_precision(i="arr")
        def CG_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            parts = []
            for op in _._ops:
                p = op.apply(arr)
                parts.append(p)
            out = sum(parts)
            return out

        CG = pyco.SelfAdjointOp(shape=(_.codim, _.codim))
        CG._ops = [op.cogram() for op in _._ops]  # embed for introspection
        CG.apply = types.MethodType(CG_apply, CG)
        return CG.squeeze()

    def op_expr(_) -> tuple:
        return ("hstack", *_._ops)

    if len(ops) == 1:
        op = ops[0].squeeze()
    else:
        _ops = [op.squeeze() for op in ops]
        klass = _infer_op_klass(_ops)

        _sh_ops = [op.shape for op in ops]
        sh_op = _infer_op_shape(_sh_ops)

        op = klass(shape=sh_op)
        op._ops = _ops  # embed for introspection

        op.apply = types.MethodType(op_apply, op)
        op.lipschitz = types.MethodType(op_lipschitz, op)
        op.asloss = types.MethodType(op_asloss, op)
        op.jacobian = types.MethodType(op_jacobian, op)
        op.diff_lipschitz = types.MethodType(op_diff_lipschitz, op)
        op.grad = types.MethodType(op_grad, op)
        op.prox = types.MethodType(op_prox, op)
        op.hessian = types.MethodType(op_hessian, op)
        op.adjoint = types.MethodType(op_adjoint, op)
        op.asarray = types.MethodType(op_asarray, op)
        op.gram = types.MethodType(op_gram, op)
        op.cogram = types.MethodType(op_cogram, op)
        op._expr = types.MethodType(op_expr, op)
    return op


def block_diag(ops: cabc.Sequence[pyct.OpT]) -> pyct.OpT:
    r"""
    Construct a block-diagonal operator.

    A block-diagonal operator :math:`D: \mathbb{R}^{d_{1} + \cdots + d_{N}} \to \mathbb{R}^{c_{1} +
    \cdots + c_{N}}` is an operator containing (diagonally) blocks of smaller operators
    :math:`\{O_{1}: \mathbb{R}^{d_{1}} \to \mathbb{R}^{c_{1}}, \ldots, O_{N}: \mathbb{R}^{d_{N}}
    \to \mathbb{R}^{c_{N}}\}`, i.e.

    .. math::

       D
       =
       \left[
           \begin{array}{ccc}
               O_{1} &        &       \\
                     & \ddots &       \\
                     &        & O_{N} \\
           \end{array}
       \right]

    Parameters
    ----------
    ops: cabc.Sequence[pyct.OpT]
        [op1(c1, d1), ..., opN(cN, dN)] operators to concatenate.

    Returns
    -------
    op: pyct.OpT
        Block-diagonal (c1+...+cN, d1+...+dN) operator.

    See Also
    --------
    :py:func:`~pycsou.operator.block.block`,
    :py:func:`~pycsou.operator.block.coo_block`.
    """

    def _infer_op_shape(sh_ops: list[pyct.Shape]) -> pyct.Shape:
        if any(_[1] == None for _ in sh_ops):
            raise ValueError("Domain-agnostic operators are unsupported.")

        dim, codim = 0, 0
        for _ in sh_ops:
            codim += _[0]
            dim += _[1]
        return (codim, dim)

    def _infer_op_klass(ops: list[pyct.OpT]) -> pyct.OpC:
        P = pyco.Property
        properties = frozenset.intersection(*[op.properties() for op in ops])
        properties -= {  # all functional-related properties are lost.
            P.FUNCTIONAL,
            P.PROXIMABLE,
            P.DIFFERENTIABLE_FUNCTION,
            P.QUADRATIC,
        }
        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        # op.apply(x) = concatenate([op1.apply(x1), ..., opN.apply(xN)], axis=-1)
        parts, i = [], 0
        for op in _._ops:
            p = op.apply(arr[..., i : i + op.dim])
            parts.append(p)
            i += op.dim

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        # op.lipschitz(**kwargs) = max([op1.lipschitz(**kwargs), ..., opN.lipschitz(**kwargs)])
        #                        + update _lipschitz
        if _.has(pyco.Property.LINEAR):
            L = _.__class__.lipschitz(_, **kwargs)
        else:
            L = max([op.lipschitz(**kwargs) for op in _._ops])
        _._lipschitz = float(L)
        return _._lipschitz

    def op_jacobian(_, arr: pyct.NDArray) -> pyct.OpT:
        # op.jacobian(x) = block_diag([op1.jacobian(x1), ..., opN.jacobian(xN)])
        if not _.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if _.has(pyco.Property.LINEAR):
            out = _
        else:
            parts, i = [], 0
            for op in _._ops:
                p = op.jacobian(arr[..., i : i + op.dim])
                parts.append(p)
                i += op.dim

            out = block_diag(parts)
        return out

    def op_diff_lipschitz(_, **kwargs) -> pyct.Real:
        # op.diff_lipschitz(**kwargs) = max([op1.diff_lipschitz(**kwargs), ..., opN.diff_lipschitz(**kwargs)])
        #                             + update _diff_lipschitz
        if not _.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if _.has(pyco.Property.LINEAR):
            dL = _.__class__.diff_lipschitz(_, **kwargs)
        else:
            dL = max([op.diff_lipschitz(**kwargs) for op in _._ops])
        _._diff_lipschitz = float(dL)
        return _._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        # op.adjoint(y) = concatenate([op1.adjoint(y1), ..., opN.adjoint(yN)], axis=-1)
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts, i = [], 0
        for op in _._ops:
            p = op.adjoint(arr[..., i : i + op.codim])
            parts.append(p)
            i += op.codim

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        # op.asarray(**kwargs) = \diag([op1.asarray(**kwargs), ..., opN.asarray(**kwargs)])
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.asarray(**kwargs)
            parts.append(p)

        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        A, i, j = xp.zeros(_.shape, dtype=dtype), 0, 0
        for op, p in zip(_._ops, parts):
            A[i : i + op.codim, j : j + op.dim] = p
            i += op.codim
            j += op.dim
        return A

    def op_gram(_) -> pyct.OpT:
        # op.gram() = \diag([op1.gram(), ..., opN.gram()])
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.gram()
            parts.append(p)

        G = block_diag(parts)
        return G

    def op_cogram(_) -> pyct.OpT:
        # op.cogram() = \diag([op1.cogram(), ..., opN.cogram()])
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.cogram()
            parts.append(p)

        CG = block_diag(parts)
        return CG

    def op_svdvals(_, **kwargs) -> pyct.NDArray:
        # op.svdvals(**kwargs) = [top|bottom-k]([op1.svdvals(**kwargs), ..., opN.svdvals(**kwargs)])
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        k = kwargs.get("k", 1)
        which = kwargs.get("which", "LM")
        if which.upper() == "SM":
            D = _.__class__.svdvals(_, **kwargs)
        else:
            parts = []
            for op in _._ops:
                p = op.svdvals(**kwargs)
                parts.append(p)

            xp = pycu.get_array_module(parts[0])
            D = xp.sort(xp.concatenate(parts, axis=0), axis=None)[-k:]
        return D

    def op_eigvals(_, **kwargs) -> pyct.NDArray:
        # op.eigvals(**kwargs) = [top|bottom-k]([op1.eigvals(**kwargs), ..., opN.eigvals(**kwargs)])
        if not _.has(pyco.Property.LINEAR_NORMAL):
            raise NotImplementedError

        parts = []
        for op in _._ops:
            p = op.eigvals(**kwargs)
            parts.append(p)

        xp = pycu.get_array_module(parts[0])
        D = xp.concatenate(parts, axis=0)
        D = D[xp.argsort(xp.abs(D))]

        k = kwargs.get("k", 1)
        which = kwargs.get("which", "LM")
        D = D[:k] if (which.upper() == "SM") else D[-k:]
        return D

    @pycrt.enforce_precision(i="arr")
    def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        # op.pinv(y, damp) = concatenate([op1.pinv(y1, damp), ..., opN.pinv(yN, damp)], axis=-1)
        if not _.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts, i = [], 0
        for op in _._ops:
            p = op.pinv(arr[..., i : i + op.codim], **kwargs)
            parts.append(p)
            i += op.codim

        xp = pycu.get_array_module(arr)
        out = xp.concatenate(parts, axis=-1)
        return out

    def op_trace(_, **kwargs) -> pyct.Real:
        # op.trace(**kwargs) = if [op1,...,opN] square
        #                          sum([op1.trace(**kwargs), ..., opN.trace(**kwargs)])
        #                      else
        #                          SquareOp.trace(**kwargs)
        if not _.has(pyco.Property.LINEAR_SQUARE):
            raise NotImplementedError

        if all(op.has(pyco.Property.LINEAR_SQUARE) for op in _._ops):
            parts = []
            for op in _._ops:
                p = op.trace(**kwargs)
                parts.append(p)
            tr = sum(parts)
        else:  # default fallback
            tr = pyco.SquareOp.trace(_, **kwargs)
        return float(tr)

    def op_expr(_) -> tuple:
        return ("block_diag", *_._ops)

    if len(ops) == 1:
        op = ops[0].squeeze()
    else:
        _ops = [op.squeeze() for op in ops]
        klass = _infer_op_klass(_ops)

        _sh_ops = [op.shape for op in ops]
        sh_op = _infer_op_shape(_sh_ops)

        op = klass(shape=sh_op)
        op._ops = _ops  # embed for introspection

        op.apply = types.MethodType(op_apply, op)
        op.lipschitz = types.MethodType(op_lipschitz, op)
        op.jacobian = types.MethodType(op_jacobian, op)
        op.diff_lipschitz = types.MethodType(op_diff_lipschitz, op)
        op.adjoint = types.MethodType(op_adjoint, op)
        op.asarray = types.MethodType(op_asarray, op)
        op.gram = types.MethodType(op_gram, op)
        op.cogram = types.MethodType(op_cogram, op)
        op.svdvals = types.MethodType(op_svdvals, op)
        op.eigvals = types.MethodType(op_eigvals, op)
        op.pinv = types.MethodType(op_pinv, op)
        op.trace = types.MethodType(op_trace, op)
        op._expr = types.MethodType(op_expr, op)
    return op


def block(
    ops: cabc.Sequence[cabc.Sequence[pyct.OpT]],
    order: pyct.Integer,
) -> pyct.OpT:
    r"""
    Construct a (dense) block-defined operator.

    A block-defined operator is an operator containing blocks of smaller operators.
    Blocks are stacked horizontally/vertically in a user-specified order to obtain the final shape.

    Parameters
    ----------
    ops: cabc.Sequence[cabc.Sequence[pyct.OpT]]
        2D nested sequence of (ck, dk)-shaped operators.
    order: 0 | 1
        Order in which the nested operators are specified/concatenated:

        * 0: concatenate inner-most blocks via ``vstack()``, then ``hstack()``.
        * 1: concatenate inner-most blocks via ``hstack()``, then ``vstack()``.

    Returns
    -------
    op: pyct.OpT
        Block-defined operator. (See below for examples.)

    Notes
    -----
    * Domain-agnostic operators, i.e. operators with None-valued ``dim`` s, are unsupported.
    * Each row/column may contain a different number of operators.

    Examples
    --------

    .. code::

       >>> block(
       ...    [
       ...     [A],        # ABEEGGG
       ...     [B, C, D],  # ACEEHHH
       ...     [E, F],     # ADFFHHH
       ...     [G, H],
       ...    ],
       ...    order=0,
       ... )

       >>> block(
       ...    [
       ...     [A, B, C, D],  # ABBCD
       ...     [E],           # EEEEE
       ...     [F, G],        # FFGGG
       ...    ],
       ...    order=1,
       ... )

    See Also
    --------
    :py:func:`~pycsou.operator.block.block_diag`,
    :py:func:`~pycsou.operator.block.coo_block`.
    """
    order = int(order)
    assert order in {0, 1}, f"order: out-of-bounds order '{order}'."

    inner = {0: vstack, 1: hstack}[order]
    outer = {0: hstack, 1: vstack}[order]

    op = outer([inner(row) for row in ops])
    return op


def coo_block(
    ops: tuple[
        cabc.Sequence[pyct.OpT],
        tuple[
            cabc.Sequence[pyct.Integer],
            cabc.Sequence[pyct.Integer],
        ],
    ],
    grid_shape: pyct.Shape,
) -> pyct.OpT:
    r"""
    Constuct a (possibly-sparse) block-defined operator in COOrdinate format.

    A block-defined operator is an operator containing blocks of smaller operators.
    Blocks must align on a coarse grid, akin to the COO format used to define sparse arrays.

    Parameters
    ----------
    ops: ([OpT], ([int], [int]))
        (data, (i, j)) sequences defining block placement, i.e.

        * `data[:]` are the blocks, in any order.
        * `i[:]` are the row indices of the block entries on the coarse grid.
        * `j[:]` are the column indices of the block entries on the coarse grid.

    grid_shape: pyct.Shape
        (M, N) shape of the coarse grid.

    Returns
    -------
    op: pyct.OpT
        Block-defined operator. (See below for examples.)

    Notes
    -----
    * Domain-agnostic operators, i.e. operators with None-valued ``dim`` s, are unsupported.
    * Blocks on the same row/column must have the same ``codim`` / ``dim`` s.
    * Each row/column of the coarse grid **must** contain at least one entry.

    Examples
    --------

    .. code::

       >>> coo_block(
       ...     ([A(500,1000), B(1,1000), C(500,500), D(1,3)],  # data
       ...      [
       ...       [0, 1, 0, 2],  # i
       ...       [0, 0, 2, 1],  # j
       ...      ]),
       ...     grid_shape=(3, 3),
       ... )

       | coarse_idx |      0       |    1    |      2      |
       |------------|--------------|---------|-------------|
       |          0 | A(500, 1000) |         | C(500, 500) |
       |          1 | B(1, 1000)   |         |             |
       |          2 |              | D(1, 3) |             |
    """
    op = _COOBlock(
        ops=ops,
        grid_shape=grid_shape,
    ).op()
    return op


class _COOBlock:  # See coo_block() for a detailed description.
    def __init__(
        self,
        ops: tuple[
            cabc.Sequence[pyct.OpT],
            tuple[
                cabc.Sequence[pyct.Integer],
                cabc.Sequence[pyct.Integer],
            ],
        ],
        grid_shape: pyct.Shape,
    ):
        self._grid_shape = tuple(grid_shape)
        self._init_spec(ops)

        # Default Arithmetic Attributes
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    def op(self) -> pyct.OpT:
        """
        Returns
        -------
        op: pyct.OpT
            Synthesized operator given inputs to
            :py:meth:`~pycsou.operator.blocks._COOBlock.__init__`.
        """
        blk = self._block
        if len(blk) == 1:
            _, op = blk.popitem()
        else:
            op = self._infer_op()
            op._block = self._block  # embed for introspection
            op._block_offset = self._block_offset  # embed for introspection
            op._grid_shape = self._grid_shape  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_attributes():
                    attr = getattr(self, name)
                    setattr(op, name, attr)
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
        return op

    def _init_spec(self, ops):
        """
        Transform input into standardized form.

        Returns
        -------
        block: (int, int) -> pyct.OpT
            coarse_grid index -> Operator at that location.
        block_offset: (int, int) -> (int, int)
            coarse_grid index -> (top-left) fine_grid index.
        """
        data, (i, j) = ops
        N_row, N_col, N_block = *self._grid_shape, len(data)
        msg = "Incorrect COO parametrization"
        assert N_block == len(i) == len(j), msg
        assert 0 < N_block <= N_row * N_col, msg
        assert 0 <= min(i) <= max(i) < N_row, msg
        assert 0 <= min(j) <= max(j) < N_col, msg

        if any(op.dim == None for op in data):
            raise ValueError("Domain-agnostic operators are unsupported.")

        # block dimensions are compatible.
        row = collections.defaultdict(list)  # row_idx -> [block]
        col = collections.defaultdict(list)  # col_idx -> [block]
        for (d, _i, _j) in zip(data, i, j):
            row[_i].append(d)
            col[_j].append(d)
        msg = lambda dim, idx, dom: f"All sub-operators on {dim} {idx} must have same {dom} size."
        for k, v in row.items():
            assert len({_.codim for _ in v}) == 1, msg("row", k, "codomain")
        for k, v in col.items():
            assert len({_.dim for _ in v}) == 1, msg("column", k, "domain")

        # no empty lines/columns in coarse grid.
        msg = lambda _: f"Coarse grid contains empty {_}: cannot infer fine-grid dimensions."
        assert len(row) == N_row, msg("rows")
        assert len(col) == N_col, msg("columns")

        # ---------------------------------------------------------------------
        # create block
        block = dict()  # coarse_grid(int, int) -> pyct.OpT
        for (d, _i, _j) in zip(data, i, j):
            block[(_i, _j)] = d.squeeze()
        self._block = block

        # create block_offset
        block_offset = dict()  # coarse_grid(int, int) -> fine_grid(int, int)
        row_offset = np.cumsum([row[k][0].codim for k in range(N_row)])
        col_offset = np.cumsum([col[k][0].dim for k in range(N_col)])
        for i in range(N_row):
            for j in range(N_col):
                _i = 0 if (i == 0) else row_offset[i - 1]
                _j = 0 if (j == 0) else col_offset[j - 1]
                block_offset[(i, j)] = _i, _j
        self._block_offset = block_offset

    def _infer_op(self) -> pyct.OpT:
        # Create the abstract COO-operator to which methods will be assigned.
        blk = self._block  # shorthand

        row = collections.defaultdict(list)
        col = collections.defaultdict(list)
        for (r, c), op in blk.items():
            row[r].append(op)
            col[c].append(op)
        N_row, N_col = len(row), len(col)
        op_codim = sum(ops[0].codim for ops in row.values())
        op_dim = sum(ops[0].dim for ops in col.values())

        P = pyco.Property
        properties = set.intersection(*[set(op.properties()) for op in blk.values()])
        if op_codim > 1:
            properties -= {
                P.FUNCTIONAL,
                P.PROXIMABLE,
                P.DIFFERENTIABLE_FUNCTION,
                P.QUADRATIC,
            }

        if all(  # block_diag case
            [
                N_row == N_col == len(blk),
                all((r, r) in blk for r in range(N_row)),
            ]
        ):
            pass
        elif op_codim == 1:  # hstack case: special treatment of quadratics
            if all(op.has(P.QUADRATIC) for op in blk.values()):
                properties.add(P.QUADRATIC)
            elif any(op.has(P.QUADRATIC) for op in blk.values()):
                # possible quadratic if all other terms are linear.
                non_quad = [op for op in blk.values() if not op.has(P.QUADRATIC)]
                if all(op.has(P.LINEAR) for op in non_quad):
                    properties.add(P.QUADRATIC)
        else:  # (1) vstack case or (2) arbitrary fill-in case => non-functionals
            properties &= {
                P.CAN_EVAL,
                P.DIFFERENTIABLE,
                P.LINEAR,
            }
            if (op_codim == op_dim) and (P.LINEAR in properties):
                properties.add(P.LINEAR_SQUARE)

        klass = pyco.Operator._infer_operator_type(properties)
        op = klass(shape=(op_codim, op_dim))
        return op

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # compute blocks
        parts = dict()
        for idx, op in self._block.items():
            offset = self._block_offset[idx][1]
            p = op.apply(arr[..., offset : offset + op.dim])
            parts[idx] = p

        # row-oriented reduction (via +)
        rows = collections.defaultdict(list)
        for (r, c), p in parts.items():
            rows[r].append(p)
        cols = [sum(rows[r]) for r in range(len(rows))]

        # concatenate to super-column
        xp = pycu.get_array_module(arr)
        out = xp.concatenate(cols, axis=-1)
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        if self.has(pyco.Property.LINEAR):
            L = self.__class__.lipschitz(self, **kwargs)
        else:
            # Various upper bounds apply depending on how the blocks are organized:
            #   * vertical alignment: L**2 = sum(L_k**2)
            #   * horizontal alignment: L**2 = max(L_k**2)
            #   * block-diagonal alignment: L**2 = max(L_k**2)
            #   * arbitrary 2D alignment: vertical+horizontal composition (or vice-versa)
            # We compute all bounds and take the lowest one.

            # squared Lipschitz constant of each block.
            Ls_all = np.zeros(self._grid_shape)
            for (r, c), op in self._block.items():
                Ls_all[r, c] = op.lipschitz(**kwargs) ** 2

            Ls_1 = Ls_all.sum(axis=0).max()  # upper bound 1
            Ls_2 = Ls_all.max(axis=1).sum()  # upper bound 2
            L = np.sqrt(min(Ls_1, Ls_2))
        self._lipschitz = float(L)
        return self._lipschitz

    def _expr(self) -> tuple:
        class _Block(pyco.Operator):
            def __init__(self, idx: tuple[int, int], op: pyct.OpT):
                super().__init__(shape=op.shape)
                self._name = op._name
                self._idx = tuple(idx)
                self._op = op

            def _expr(self) -> tuple:
                head = f"block[" + ", ".join(map(str, self._idx)) + "]"
                return (head, self._op)

        # head = coo_block[grid_shape]
        head = "coo_block[" + ", ".join(map(str, self._grid_shape)) + "]"

        # tail = sequence of sub-blocks, encapsulated in _Block to obtain hierarchical view.
        tail = [_Block(idx=k, op=op) for (k, op) in self._block.items()]
        return (head, *tail)

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if not self.has(pyco.Property.PROXIMABLE):
            raise NotImplementedError

        parts = dict()
        for idx, op in self._block.items():
            offset = self._block_offset[idx][1]
            p = op.prox(arr=arr[..., offset : offset + op.dim], tau=tau)
            parts[idx] = p

        xp = pycu.get_array_module(arr)
        parts = [parts[(0, c)] for c in range(len(parts))]  # re-ordering
        out = xp.concatenate(parts, axis=-1)
        return out

    def _hessian(self) -> pyct.OpT:
        if not self.has(pyco.Property.QUADRATIC):
            raise NotImplementedError

        parts = dict()
        for idx, op in self._block.items():
            if op.has(pyco.Property.QUADRATIC):
                p = op._hessian()
            else:  # op is necessarily LINEAR
                from pycsou.operator.linop import NullOp

                p = NullOp(shape=(op.dim, op.dim)).asop(pyco.PosDefOp)
            parts[idx] = p

        parts = [parts[k] for k in sorted(parts.keys())]  # re-ordering
        H = block_diag(parts)
        return H

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        if not self.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if self.has(pyco.Property.LINEAR):
            out = self
        else:
            data, i, j = [], [], []
            for (r, c), op in self._block.items():
                offset = self._block_offset[(r, c)][1]
                opJ = op.jacobian(arr[offset : offset + op.dim])

                i.append(r)
                j.append(c)
                data.append(opJ)

            out = _COOBlock(
                ops=(data, (i, j)),
                grid_shape=self._grid_shape,
            ).op()
        return out

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        if not self.has(pyco.Property.DIFFERENTIABLE):
            raise NotImplementedError

        if self.has(pyco.Property.LINEAR):
            dL = self.__class__.diff_lipschitz(self, **kwargs)
        else:
            # Various upper bounds apply depending on how the blocks are organized:
            #   * vertical alignment: dL**2 = sum(dL_k**2)
            #   * horizontal alignment: dL**2 = max(dL_k**2)
            #   * block-diagonal alignment: dL**2 = max(dL_k**2)
            #   * arbitrary 2D alignment: vertical+horizontal composition (or vice-versa)
            # We compute all bounds and take the lowest one.

            # squared diff-Lipschitz constant of each block.
            dLs_all = np.zeros(self._grid_shape)
            for (r, c), op in self._block.items():
                dLs_all[r, c] = op.diff_lipschitz(**kwargs) ** 2

            dLs_1 = dLs_all.sum(axis=0).max()  # upper bound 1
            dLs_2 = dLs_all.max(axis=1).sum()  # upper bound 2
            dL = np.sqrt(min(dLs_1, dLs_2))
        self._diff_lipschitz = float(dL)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not self.has(pyco.Property.DIFFERENTIABLE_FUNCTION):
            raise NotImplementedError

        parts = dict()
        for idx, op in self._block.items():
            offset = self._block_offset[idx][1]
            p = op.grad(arr[..., offset : offset + op.dim])
            parts[idx] = p

        xp = pycu.get_array_module(arr)
        parts = [parts[(0, c)] for c in range(len(parts))]  # re-ordering
        out = xp.concatenate(parts, axis=-1)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not self.has(pyco.Property.LINEAR):
            raise NotImplementedError

        # compute blocks
        parts = dict()
        for idx, op in self._block.items():
            offset = self._block_offset[idx][0]
            p = op.adjoint(arr[..., offset : offset + op.codim])
            parts[idx[::-1]] = p

        # row-oriented reduction (via +)
        rows = collections.defaultdict(list)
        for (r, c), p in parts.items():
            rows[r].append(p)
        cols = [sum(rows[r]) for r in range(len(rows))]

        # concatenate to super-column
        xp = pycu.get_array_module(arr)
        out = xp.concatenate(cols, axis=-1)
        return out

    def asarray(self, **kwargs) -> pyct.NDArray:
        if not self.has(pyco.Property.LINEAR):
            raise NotImplementedError

        parts = dict()
        for idx, op in self._block.items():
            p = op.asarray(**kwargs)
            parts[idx] = p

        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        A = xp.zeros(self.shape, dtype=dtype)
        for idx, p in parts.items():
            r_o, c_o = self._block_offset[idx]  # offsets
            r_s, c_s = self._block[idx].shape  # spans
            A[r_o : r_o + r_s, c_o : c_o + c_s] = p
        return A

    def svdvals(self, **kwargs) -> pyct.NDArray:
        D = self.__class__.svdvals(self, **kwargs)
        return D

    def eigvals(self, **kwargs) -> pyct.NDArray:
        D = self.__class__.eigvals(self, **kwargs)
        return D

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = self.__class__.pinv(self, arr=arr, **kwargs)
        return out

    def gram(self) -> pyct.OpT:
        if not self.has(pyco.Property.LINEAR):
            raise NotImplementedError

        blk = self._block  # shorthand
        N_row, N_col = self._grid_shape

        # Determine operator(s) which will occupy position (r,c) on coarse grid.
        ops = collections.defaultdict(list)  # coarse_grid(int, int) -> [pyct.OpT]
        for r, c in itertools.product(range(N_col), repeat=2):
            for k in range(N_row):
                if ((k, r) in blk) and ((k, c) in blk):
                    if r == c:
                        _op = blk[(k, r)].gram()
                    else:
                        _op = blk[(k, r)].T * blk[(k, c)]
                    ops[(r, c)].append(_op)

        # ops[(r,c)] should be reduced (via +) to form a single operator per (r,c)-entry.
        # It is inefficient however to chain so many operators together via AddRule().
        # G.[apply,adjoint]() are thus redefined to improve performance.
        data, i, j = [], [], []
        for (r, c), _ops in ops.items():
            klass = pyco.SelfAdjointOp if (r == c) else pyco.LinOp
            _op = klass(shape=_ops[0].shape)  # .[apply|adjoint]() overridden below.
            _op._ops = _ops  # embed for introspection

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
                parts = [op.apply(arr) for op in _._ops]
                out = sum(parts)
                return out

            @pycrt.enforce_precision(i="arr")
            def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
                parts = [op.adjoint(arr) for op in _._ops]
                out = sum(parts)
                return out

            _op.apply = types.MethodType(op_apply, _op)
            _op.adjoint = types.MethodType(op_adjoint, _op)

            data.append(_op)
            i.append(r)
            j.append(c)
        G = (
            _COOBlock(
                ops=(data, (i, j)),
                grid_shape=(N_col, N_col),
            )
            .op()
            .asop(pyco.SelfAdjointOp)
        )
        return G.squeeze()

    def cogram(self) -> pyct.OpT:
        if not self.has(pyco.Property.LINEAR):
            raise NotImplementedError

        blk = self._block  # shorthand
        N_row, N_col = self._grid_shape

        # Determine operator(s) which will occupy position (r,c) on coarse grid.
        ops = collections.defaultdict(list)  # coarse_grid(int, int) -> [pyct.OpT]
        for r, c in itertools.product(range(N_row), repeat=2):
            for k in range(N_col):
                if ((r, k) in blk) and ((c, k) in blk):
                    if r == c:
                        _op = blk[(r, k)].cogram()
                    else:
                        _op = blk[(r, k)] * blk[(c, k)].T
                    ops[(r, c)].append(_op)

        # ops[(r,c)] should be reduced (via +) to form a single operator per (r,c)-entry.
        # It is inefficient however to chain so many operators together via AddRule().
        # CG.[apply,adjoint]() are thus redefined to improve performance.
        data, i, j = [], [], []
        for (r, c), _ops in ops.items():
            klass = pyco.SelfAdjointOp if (r == c) else pyco.LinOp
            _op = klass(shape=_ops[0].shape)  # .[apply|adjoint]() overridden below.
            _op._ops = _ops  # embed for introspection

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
                parts = [op.apply(arr) for op in _._ops]
                out = sum(parts)
                return out

            @pycrt.enforce_precision(i="arr")
            def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
                parts = [op.adjoint(arr) for op in _._ops]
                out = sum(parts)
                return out

            _op.apply = types.MethodType(op_apply, _op)
            _op.adjoint = types.MethodType(op_adjoint, _op)

            data.append(_op)
            i.append(r)
            j.append(c)
        CG = (
            _COOBlock(
                ops=(data, (i, j)),
                grid_shape=(N_row, N_row),
            )
            .op()
            .asop(pyco.SelfAdjointOp)
        )
        return CG.squeeze()

    def trace(self, **kwargs) -> pyct.Real:
        tr = self.__class__.trace(self, **kwargs)
        return float(tr)

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        msg = "asloss() is ambiguous for block-defined operators."
        raise NotImplementedError(msg)
