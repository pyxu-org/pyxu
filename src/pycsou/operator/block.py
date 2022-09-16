import collections.abc as cabc

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
    # CAN_EVAL: always
    #     op.apply(x) = concatenate([op1.apply(x1), ..., opN.apply(xN)], axis=-1)
    #     op._lipschitz = inf
    #     op.lipschitz(**kwargs) = max([op1.lipschitz(**kwargs), ..., opN.lipschitz(**kwargs)])
    #                            + update _lipschitz
    #     op._expr() = (op._name, op1, ..., opN)
    #     op._name = "block_diag"
    # DIFFERENTIABLE: [op1,...,opN] all differentiable
    #     op.jacobian(x) = block_diag([op1.jacobian(x1), ..., opN.jacobian(xN)])
    #     op._diff_lipschitz = inf
    #     op.diff_lipschitz(**kwargs) = max([op1.diff_lipschitz(**kwargs), ..., opN.diff_lipschitz(**kwargs)])
    #                                 + update _diff_lipschitz
    # LINEAR: [op1,...,opN] all linear
    #     op.adjoint(y) = concatenate([op1.adjoint(y1), ..., opN.adjoint(yN)], axis=-1)
    #     op.asarray(**kwargs) = \diag([op1.asarray(**kwargs), ..., opN.asarray(**kwargs)])
    #     op.svdvals(**kwargs) = [top|bottom-k]([op1.svdvals(**kwargs), ..., opN.svdvals(**kwargs)])
    #     op.pinv(y, damp) = concatenate([op1.pinv(y1, damp), ..., opN.pinv(yN, damp)], axis=-1)
    #     op.gram() = \diag([op1.gram(), ..., opN.gram()])
    #     op.cogram() = \diag([op1.cogram(), ..., opN.cogram()])
    # LINEAR_SQUARE: final shape square & all linear
    #     op.trace(**kwargs) = if [op1,...,opN] square
    #                              sum([op1.trace(**kwargs), ..., opN.trace(**kwargs)])
    #                          else
    #                              SquareOp.trace(**kwargs)
    # LINEAR_NORMAL: [op1, ..., opN] all normal
    #     op.eigvals(**kwargs) = [top|bottom-k]([op1.eigvals(**kwargs), ..., opN.eigvals(**kwargs)])
    # LINEAR_IDEMPOTENT: [op1, ..., opN] all idempotent
    # LINEAR_SELF_ADJOINT: [op1, ..., opN] all self-adjoint
    # LINEAR_POSITIVE_DEFINITE: [op1, ..., opN] all positive-definite
    # LINEAR_UNITARY: [op1, ..., opN] all unitary
    pass


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
