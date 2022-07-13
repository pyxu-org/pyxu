"""
Operator Arithmetic.
"""

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.util.ptype as pyct


def add(lhs: pyct.OpT, rhs: pyct.OpT) -> pyct.OpT:
    pass


def scale(op: pyct.OpT, cst: pyct.Real) -> pyct.OpT:
    from pycsou.operator.linop import HomothetyOp, NullFunc, NullOp

    if np.isclose(cst, 0):
        return NullOp(shape=op.shape) if (op.codim > 1) else NullFunc()
    elif np.isclose(cst, 1):
        return op
    else:
        h = HomothetyOp(cst, dim=op.codim)
        return compose(h, op)


def compose(lhs: pyct.OpT, rhs: pyct.OpT) -> pyct.OpT:
    pass


def pow(op: pyct.OpT, k: pyct.Integer) -> pyct.OpT:
    from pycsou.operator.linop import IdentityOp

    assert op.codim == op.dim, f"pow: expected endomorphism, got {op}."
    assert k >= 0, "pow: only non-negative exponents are supported."

    if k == 0:
        return IdentityOp(dim=op.codim)
    else:
        op_pow = op
        if pyco.Property.LINEAR_IDEMPOTENT not in op.properties():
            for _ in range(k - 1):
                op_pow = compose(op, op_pow)
        return op_pow


def argscale(op: pyct.OpT, cst: pyct.Real) -> pyct.OpT:
    from pycsou.operator.linop import HomothetyOp

    if np.isclose(cst, 1):
        return op
    else:
        # Cannot instantiate HomothetyOp if op.dim == None.
        # Trick: create it of size (1, 1), then modify ._shape manually.
        h = HomothetyOp(cst, dim=1)
        h._shape = (1, op.dim)
        return compose(op, h)


def argshift(op: pyct.OpT, cst: pyct.NDArray) -> pyct.OpT:
    pass
