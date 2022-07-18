"""
Operator Arithmetic.
"""

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.util.ptype as pyct


def add(lhs: pyct.OpT, rhs: pyct.OpT) -> pyct.OpT:
    # special case values
    #     _lhs || _rhs = Null[Op|Func] -> _rhs/_lhs
    #     _lhs && _rhs = Null[Op|Func] -> Null[Op|Func]
    #     _lhs == _rhs -> scale(_lhs, cst=2)
    # else
    #     Store _lhs and _rhs
    #     Properties to Keep
    #         Keep (base = _lhs.properties() & _rhs.properties())
    #         Then post-process as such:
    #             Always drop
    #                 LINEAR_UNITARY
    #                 LINEAR_IDEMPOTENT
    #             if (SELF-ADJOINT in base) and ([vice-versa] _lhs/_rhs is POS-DEF/IDEMPOTENT)
    #                 # Case OrthProj + Pos-Def -> Pos-Def
    #                 Add back LINEAR_POSITIVE_DEFINITE
    #             if (PROXIMAL in base)
    #                 if LINEAR not in _lhs or _rhs:
    #                     # prox preserved only for prox + linear
    #                     drop PROXIMAL
    #                 elif QUADRATIC in _lhs or _rhs:
    #                     # quadratic preserved during quad + linear
    #                     add back QUADRATIC
    #     Arithmetic Rule
    #         CAN_EVAL
    #             op.apply(arr) = _lhs.apply(arr) + _rhs.apply(arr)
    #             op._lipschitz = _lhs._lipschitz + _rhs._lipschitz
    #             op.lipschitz()
    #                 = _lhs.lipschitz() + _rhs.lipschitz()
    #                 + update op._lipschitz
    #         PROXIMABLE
    #             op.prox(arr, tau) = _lhs.prox(arr - tau * _rhs.grad(arr), tau)
    #                           OR  = _rhs.prox(arr - tau * _lhs.grad(arr), tau)
    #                 IMPORTANT: the one calling .grad() should be either (lhs, rhs) which has LINEAR property
    #         DIFFERENTIABLE
    #             op._diff_lipschitz = _lhs._diff_lipschitz + _rhs._diff_lipschitz
    #             op.diff_lipschitz()
    #                 = _lhs.diff_lipschitz() + _rhs.diff_lipschitz()
    #                 + update op._diff_lipschitz
    #             op.jacobian(arr) = _lhs.jacobian(arr) + _rhs.jacobian(arr)
    #         DIFFERENTIABLE_FUNCTION
    #             op.grad(arr) = _lhs.grad(arr) + _rhs.grad(arr)
    #         LINEAR
    #             op.adjoint(arr) = _lhs.adjoint(arr) + _rhs.adjoint(arr)
    pass


def scale(op: pyct.OpT, cst: pyct.Real) -> pyct.OpT:
    # special scale values
    #     0: NullOp/NullFunc
    #     1: self
    # else
    #     Store _orig and _scale
    #     Preserve during scale by \alpha != {0, 1}
    #         CAN_EVAL
    #             op_new.apply(arr) = op_old.apply(arr) * \alpha
    #             op_new._lipschitz = op_old._lipschitz * abs(\alpha)
    #             op_new.lipschitz()
    #                 = op_old.lipschitz() * abs(\alpha)
    #                 + update op_new._lipschitz
    #         FUNCTIONAL
    #         PROXIMABLE (if \alpha > 0)
    #             op_new.prox(arr, tau) = op_old.prox(arr, tau * \alpha)
    #         DIFFERENTIABLE
    #             op_new.jacobian(arr) = op_old.jacobian(arr) * \alpha
    #             op_new._diff_lipschitz = op_old._diff_lipschitz * abs(\alpha)
    #             diff_lipschitz()
    #                 = op_old.diff_lipschitz() * abs(\alpha)
    #                 + update op_new._diff_lipschitz
    #         DIFFERENTIABLE_FUNCTION
    #             op_new.grad(arr) = op_old.grad(arr) * \alpha
    #         LINEAR
    #             op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha
    #         LINEAR_SQUARE
    #         LINEAR_NORMAL
    #         LINEAR_SELF_ADJOINT
    #         LINEAR_POSITIVE_DEFINITE (if \alpha > 0)
    #         LINEAR_UNITARY (only if \alpha = -1)
    #         QUADRATIC (only if \alpha > 0)
    pass


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
    # special case values
    #     0: constant-valued function [looks like NullOp/Func, but with different .apply(), and not linear]
    #         op_new.apply(arr) = op_old.apply(0) [cast to right array type]
    #         op_new._lipschitz = 0
    #         op_new.lipschitz() = op_new._lipschitz alias
    #         op_new.prox(arr, tau) = arr
    #         op_new.jacobian(arr) = NullOp/Func
    #         op_new._diff_lipschitz = 0
    #         op_new.diff_lipschitz() = op_new._diff_lipschitz alias
    #         op_new.grad(arr) = zeros [cast to right array type]
    #     1: self
    # else
    #     Store _orig and _scale
    #     Preserve during argscale by \alpha != {0, 1}
    #         CAN_EVAL
    #             op_new.apply(arr) = op_old.apply(arr * \alpha)
    #             op_new._lipschitz = op_old._lipschitz * abs(\alpha)
    #             op_new.lipschitz()
    #                 = op_old.lipschitz() * abs(\alpha)
    #                 + update op_new._lipschitz
    #         FUNCTIONAL
    #         PROXIMABLE
    #             op_new.prox(arr, tau) = op_old.prox(\alpha * arr, \alpha**2 * tau) / \alpha
    #         DIFFERENTIABLE
    #             _diff_lipschitz = op_old._diff_lipschitz * abs(\alpha)
    #             diff_lipschitz()
    #                 = op_old.diff_lipschitz() * abs(\alpha)
    #                 + update op_new._diff_lipschitz
    #             op_new.jacobian(arr) = op_old.jacobian(op_old.apply(arr)) * \alpha
    #         DIFFERENTIABLE_FUNCTION
    #             op_new.grad(arr) = op_old.grad(\alpha * arr) * \alpha
    #         LINEAR
    #             op_new.adjoint(arr) = op_old.adjoint(y) * \alpha
    #         LINEAR_SQUARE
    #         LINEAR_NORMAL
    #         LINEAR_SELF_ADJOINT
    #         LINEAR_POSITIVE_DEFINITE (if \alpha > 0)
    #         LINEAR_UNITARY (only if \alpha = -1)
    #         QUADRATIC
    pass


def argshift(op: pyct.OpT, cst: pyct.NDArray) -> pyct.OpT:
    # cst must have right dimensions
    # special case values
    #     0: self
    # else
    #     Store _orig and _shift
    #     Preserve during argshift by \shift != 0
    #         CAN_EVAL
    #             op_new.apply(arr) = op_old.apply(arr + \shift)
    #             op_new._lipschitz = op_old._lipschitz
    #             op_new.lipschitz() = op_new._lipschitz alias
    #         FUNCTIONAL
    #         PROXIMABLE
    #             op_new.prox(arr, tau) = op_old.prox(arr + \shift, tau) - \shift
    #         DIFFERENTIABLE
    #             op_new._diff_lipschitz = op_old._diff_lipschitz
    #             op_new.diff_lipschitz() = op_new._diff_lipschitz alias
    #             op_new.jacobian(arr) = op_old.jacobian(arr + \shift)
    #         DIFFERENTIABLE_FUNCTION
    #             op_new.grad(arr) = op_old.grad(arr + \shift)
    #         QUADRATIC
    pass
