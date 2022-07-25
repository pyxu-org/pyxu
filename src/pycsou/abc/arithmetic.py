"""
Operator Arithmetic.
"""

import types

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class Rule:
    def op(self) -> pyct.OpT:
        """
        Returns
        -------
        op: pyct.OpT
            Synthesized operator given inputs to :py:meth:`~pycsou.abc.arithmetic.Rule.__init__`.
        """
        raise NotImplementedError


class ScaleRule(Rule):
    r"""
    Special Cases:
        \alpha = 0  => NullOp/NullFunc
        \alpha = 1  => self
    Else:
        |--------------------------|-------------|---------------------------------------------------------------|
        |         Property         |  Preserved? |                   Arithmetic Update Rule(s)                   |
        |--------------------------|-------------|---------------------------------------------------------------|
        | CAN_EVAL                 | yes         | op_new.apply(arr) = op_old.apply(arr) * \alpha                |
        |                          |             | op_new._lipschitz = op_old._lipschitz * abs(\alpha)           |
        |                          |             |                                                               |
        |                          |             | op_new.lipschitz()                                            |
        |                          |             | = op_old.lipschitz() * abs(\alpha)                            |
        |                          |             | + update op_new._lipschitz                                    |
        |--------------------------|-------------|---------------------------------------------------------------|
        | FUNCTIONAL               | yes         |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | PROXIMABLE               | \alpha > 0  | op_new.prox(arr, tau) = op_old.prox(arr, tau * \alpha)        |
        |--------------------------|-------------|---------------------------------------------------------------|
        | DIFFERENTIABLE           | yes         | op_new.jacobian(arr) = op_old.jacobian(arr) * \alpha          |
        |                          |             | op_new._diff_lipschitz = op_old._diff_lipschitz * abs(\alpha) |
        |                          |             |                                                               |
        |                          |             | diff_lipschitz()                                              |
        |                          |             | = op_old.diff_lipschitz() * abs(\alpha)                       |
        |                          |             | + update op_new._diff_lipschitz                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes         | op_new.grad(arr) = op_old.grad(arr) * \alpha                  |
        |--------------------------|-------------|---------------------------------------------------------------|
        | QUADRATIC                | \alpha > 0  |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR                   | yes         | op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha            |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_SQUARE            | yes         |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_NORMAL            | yes         |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_UNITARY           | \alpha = -1 |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_SELF_ADJOINT      | yes         |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_POSITIVE_DEFINITE | \alpha > 0  |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
        | LINEAR_IDEMPOTENT        | no          |                                                               |
        |--------------------------|-------------|---------------------------------------------------------------|
    """

    def __init__(self, op: pyct.OpT, cst: pyct.Real):
        self._op = op._squeeze()
        self._cst = float(cst)

        # Arithmetic Attributes
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    def op(self) -> pyct.OpT:
        if np.isclose(self._cst, 0):
            from pycsou.operator.linop import NullOp

            op = NullOp(shape=self._op.shape)._squeeze()
        elif np.isclose(self._cst, 1):
            op = self._op
        else:
            klass = self._infer_op_klass()
            op = klass(shape=self._op.shape)
            op._op = self._op  # embed for introspection
            op._cst = self._cst  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_attributes():
                    attr = getattr(self, name)
                    setattr(op, name, attr)
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
        return op

    def _infer_op_klass(self) -> pyct.OpC:
        preserved = {
            pyco.Property.CAN_EVAL,
            pyco.Property.FUNCTIONAL,
            pyco.Property.DIFFERENTIABLE,
            pyco.Property.DIFFERENTIABLE_FUNCTION,
            pyco.Property.LINEAR,
            pyco.Property.LINEAR_SQUARE,
            pyco.Property.LINEAR_NORMAL,
            pyco.Property.LINEAR_SELF_ADJOINT,
            pyco.Property.PROXIMABLE,
            # PROXIMABLE should only be preserved if `cst > 0`.
            # (Reason: (\alpha f)(x) is no longer convex otherwise.)
            # However since prox property should be preserved for LinFuncs (i.e., the
            # borderline-convex case), we keep PROXIMABLE regardless and disallow calling .prox().
            # [See .prox() override.]
        }
        if self._cst > 0:
            preserved |= {
                pyco.Property.LINEAR_POSITIVE_DEFINITE,
                pyco.Property.QUADRATIC,
            }
        if np.isclose(self._cst, -1):
            preserved.add(pyco.Property.LINEAR_UNITARY)

        properties = self._op.properties() & preserved
        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = self._op.apply(arr)
        out *= self._cst
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = self._op.lipschitz(**kwargs)
        self._lipschitz *= abs(self._cst)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if self._cst > 0:
            return self._op.prox(arr, tau * self._cst)
        else:
            # See comment in _infer_op_klass()
            raise NotImplementedError

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        return self._op.jacobian(arr) * self._cst

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._op.diff_lipschitz(**kwargs)
        self._diff_lipschitz *= abs(self._cst)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = self._op.grad(arr)
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = self._op.adjoint(arr)
        out *= self._cst
        return out


class ArgScaleRule(Rule):
    r"""
    Special Cases:
        \alpha = 0  => ConstantValued (w/ potential vector-valued output)
        \alpha = 1  => self
    Else:
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        |         Property         |  Preserved? |                          Arithmetic Update Rule(s)                          |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | CAN_EVAL                 | yes         | op_new.apply(arr) = op_old.apply(arr * \alpha)                              |
        |                          |             | op_new._lipschitz = op_old._lipschitz * abs(\alpha)                         |
        |                          |             |                                                                             |
        |                          |             | op_new.lipschitz()                                                          |
        |                          |             | = op_old.lipschitz() * abs(\alpha)                                          |
        |                          |             | + update op_new._lipschitz                                                  |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | FUNCTIONAL               | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | PROXIMABLE               | yes         | op_new.prox(arr, tau) = op_old.prox(\alpha * arr, \alpha**2 * tau) / \alpha |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | DIFFERENTIABLE           | yes         | _diff_lipschitz = op_old._diff_lipschitz * abs(\alpha)                      |
        |                          |             |                                                                             |
        |                          |             | diff_lipschitz()                                                            |
        |                          |             | = op_old.diff_lipschitz() * abs(\alpha)                                     |
        |                          |             | + update op_new._diff_lipschitz                                             |
        |                          |             |                                                                             |
        |                          |             | op_new.jacobian(arr) = op_old.jacobian(arr * \alpha) * \alpha               |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes         | op_new.grad(arr) = op_old.grad(\alpha * arr) * \alpha                       |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | QUADRATIC                | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR                   | yes         | op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha                          |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_SQUARE            | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_NORMAL            | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_UNITARY           | \alpha = -1 |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_SELF_ADJOINT      | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_POSITIVE_DEFINITE | \alpha > 0  |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_IDEMPOTENT        | no          |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
    """

    def __init__(self, op: pyct.OpT, cst: pyct.Real):
        self._op = op._squeeze()
        self._cst = float(cst)

        # Arithmetic Attributes
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    def op(self) -> pyct.OpT:
        if np.isclose(self._cst, 0):
            # ConstantVECTOR output: modify ConstantValued to work.
            from pycsou.operator.map import ConstantValued

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
                xp = pycu.get_array_module(arr)
                arr = xp.zeros(
                    (*arr.shape[:-1], self._op.dim),
                    dtype=arr.dtype,
                )
                out = self._op.apply(arr)
                return out

            op = ConstantValued(
                shape=self._op.shape,
                cst=self._cst,
            )
            op.apply = types.MethodType(op_apply, op)
        elif np.isclose(self._cst, 1):
            op = self._op
        else:
            klass = self._infer_op_klass()
            op = klass(shape=self._op.shape)
            op._op = self._op  # embed for introspection
            op._cst = self._cst  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_attributes():
                    attr = getattr(self, name)
                    setattr(op, name, attr)
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
        return op

    def _infer_op_klass(self) -> pyct.OpC:
        preserved = {
            pyco.Property.CAN_EVAL,
            pyco.Property.FUNCTIONAL,
            pyco.Property.PROXIMABLE,
            pyco.Property.DIFFERENTIABLE,
            pyco.Property.DIFFERENTIABLE_FUNCTION,
            pyco.Property.LINEAR,
            pyco.Property.LINEAR_SQUARE,
            pyco.Property.LINEAR_NORMAL,
            pyco.Property.LINEAR_SELF_ADJOINT,
            pyco.Property.QUADRATIC,
        }
        if self._cst > 0:
            preserved.add(pyco.Property.LINEAR_POSITIVE_DEFINITE)
        if np.isclose(self._cst, -1):
            preserved.add(pyco.Property.LINEAR_UNITARY)

        properties = self._op.properties() & preserved
        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = arr.copy()
        x *= self._cst
        out = self._op.apply(x)
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = self._op.lipschitz(**kwargs)
        self._lipschitz *= abs(self._cst)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        x = arr.copy()
        x *= self._cst
        out = self._op.prox(x, (self._cst**2) * tau)
        out /= self._cst
        return out

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        x = arr.copy()
        x *= self._cst
        op = self._op.jacobian(x) * self._cst
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._op.diff_lipschitz(**kwargs)
        self._diff_lipschitz *= abs(self._cst)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = arr.copy()
        x *= self._cst
        out = self._op.grad(x)
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = self._op.adjoint(arr)
        out *= self._cst
        return out


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
