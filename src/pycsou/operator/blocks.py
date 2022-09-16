import collections.abc as cabc
import types

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
    # CAN_EVAL: always
    #     op.apply(x) = concatenate([op1.apply(x), ..., opN.apply(x)], axis=-1)
    #     op._lipschitz = inf
    #     op.lipschitz(**kwargs) = sum([op1.lipschitz(**kwargs), ..., opN.lipschitz(**kwargs)])
    #                            + update _lipschitz
    #     op._expr() = (op._name, op1, ..., opN)
    #     op._name = "vstack"
    # DIFFERENTIABLE: [op1,...,opN] differentiable
    #     op.jacobian(x) = vstack([op1.jacobian(x), ..., opN.jacobian(x)])
    #     op._diff_lipschitz = inf
    #     op.diff_lipschitz(**kwargs) = sum([op1.diff_lipschitz(**kwargs), ..., opN.diff_lipschitz(**kwargs)])
    #                                 + update _diff_lipschitz
    # LINEAR: [op1,...,opN] linear
    #     op.adjoint(y) = concatenate([op1.adjoint(y1), ..., opN.adjoint(yN)], axis=-1)
    #     op.asarray(**kwargs) = concatenate([op1.asarray(**kwargs), ..., opN.asarray(**kwargs)], axis=0)
    #     op.svdvals(**kwargs) = LinOp.svdvals(**kwargs)
    #     op.pinv(y, damp) = LinOp.pinv(y, damp)
    #     op.gram() = op1.gram() + ... + opN.gram()
    #     op.cogram() = \diag([op1.cogram(), ..., opN.cogram()]) + lower-triangular outer-product
    #                 = constructed via block()
    # LINEAR_SQUARE: final shape square & [op1,...,opN] linear
    #     op.trace(**kwargs) = SquareOp.trace(**kwargs)
    pass


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
    # CAN_EVAL: always
    #     op.apply(x) = sum([op1.apply(x1), ..., opN.apply(xN)], axis=0)
    #     op._lipschitz = inf
    #     op.lipschitz(**kwargs) = max([op1.lipschitz(**kwargs), ..., opN.lipschitz(**kwargs)])
    #                            + update _lipschitz
    #     op._expr() = (op._name, op1, ..., opN)
    #     op._name = "hstack"
    # FUNCTIONAL: [op1,...,opN] all functional
    #     op.asloss(x) = NotImplementedError("ambiguous")
    # PROXIMABLE: [op1, ..., opN] all proximable
    #     op.prox(x, tau) = concatenate([op1.prox(x1, tau), ..., opN.prox(xN, tau)], axis=-1)
    # DIFFERENTIABLE: [op1, ..., opN] all differentiable
    #     op.jacobian(x) = hstack([op1.jacobian(x1), ..., opN.jacobian(xN)])
    #     op._diff_lipschitz = inf
    #     op.diff_lipschitz(**kwargs) = max([op1.diff_lipschitz(**kwargs), ..., opN.diff_lipschitz(**kwargs)])
    #                                 + update _diff_lipschitz
    # DIFFERENTIABLE_FUNCTION: [op1, ..., opN] all differentiable functions
    #     op.grad(x) = concatenate([op1.grad(x1), ..., opN.grad(xN)], axis=-1)
    # LINEAR: [op1, ..., opN] all linear
    #     op.adjoint(y) = concatenate([op1.adjoint(y1), ..., opN.adjoint(yN)], axis=-1)
    #     op.asarray(**kwargs) = concatenate([op1.asarray(**kwargs), ..., opN.asarray(**kwargs)], axis=-1)
    #     op.svdvals(**kwargs) = LinOp.svdvals(**kwargs)
    #     op.pinv(y, damp) = LinOp.pinv(y, damp)
    #     op.gram() = \diag([op1.gram(), ..., opN.gram()]) + lower-triangular outer-product
    #               = constructed via block()
    #     op.cogram() = op1.cogram() + ... + opN.cogram()
    # LINEAR_SQUARE: final shape square & [op1, ..., opN] linear
    #     op.trace(**kwargs) = SquareOp.trace(**kwargs)
    # QUADRATIC: [op1, ..., opN] = at least one quad and (rest linear or quad)
    #     op._hessian() = block_diag([op1._hessian(), ..., opN._hessian()])
    #                     w/ zeros (pos-defed) if needed on diagonal.
    pass


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
       ...     grid_shape=(2, 2),
       ... )

       | coarse_idx |      0       |    1    |      2      |
       |------------|--------------|---------|-------------|
       |          0 | A(500, 1000) |         | C(500, 500) |
       |          1 | B(1, 1000)   |         |             |
       |          2 |              | D(1, 3) |             |
    """
    pass
