"""
Operator Arithmetic.
"""

import types

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class Rule:
    def __init__(self):
        # Arithmetic Attributes
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

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
        super().__init__()
        self._op = op._squeeze()
        self._cst = float(cst)

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
        }
        if self._cst > 0:
            preserved |= {
                pyco.Property.LINEAR_POSITIVE_DEFINITE,
                pyco.Property.QUADRATIC,
                pyco.Property.PROXIMABLE,
            }
        if self._op.has(pyco.Property.LINEAR):
            preserved.add(pyco.Property.PROXIMABLE)
        if np.isclose(self._cst, -1):
            preserved.add(pyco.Property.LINEAR_UNITARY)

        properties = self._op.properties() & preserved
        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = pycu.copy_if_unsafe(self._op.apply(arr))
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
        return self._op.prox(arr, tau * self._cst)

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        if self.has(pyco.Property.LINEAR):
            op = self
        else:
            op = self._op.jacobian(arr) * self._cst
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._op.diff_lipschitz(**kwargs)
        self._diff_lipschitz *= abs(self._cst)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = pycu.copy_if_unsafe(self._op.grad(arr))
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = pycu.copy_if_unsafe(self._op.adjoint(arr))
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
        | DIFFERENTIABLE           | yes         | _diff_lipschitz = op_old._diff_lipschitz * (\alpha**2)                      |
        |                          |             |                                                                             |
        |                          |             | diff_lipschitz()                                                            |
        |                          |             | = op_old.diff_lipschitz() * (\alpha**2)                                     |
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
        super().__init__()
        self._op = op._squeeze()
        self._cst = float(cst)

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
        if self.has(pyco.Property.LINEAR):
            op = self
        else:
            x = arr.copy()
            x *= self._cst
            op = self._op.jacobian(x) * self._cst
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._op.diff_lipschitz(**kwargs)
        self._diff_lipschitz *= self._cst**2
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
        out = pycu.copy_if_unsafe(self._op.adjoint(arr))
        out *= self._cst
        return out


class ArgShiftRule(Rule):
    r"""
    Special Cases:
        \shift = 0  => self
    Else:
        |--------------------------|------------|-----------------------------------------------------------------|
        |         Property         | Preserved? |                    Arithmetic Update Rule(s)                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | CAN_EVAL                 | yes        | op_new.apply(arr) = op_old.apply(arr + \shift)                  |
        |                          |            | op_new._lipschitz = op_old._lipschitz                           |
        |                          |            | op_new.lipschitz() = op_new._lipschitz alias                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | FUNCTIONAL               | yes        |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | PROXIMABLE               | yes        | op_new.prox(arr, tau) = op_old.prox(arr + \shift, tau) - \shift |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE           | yes        | op_new._diff_lipschitz = op_old._diff_lipschitz                 |
        |                          |            | op_new.diff_lipschitz() = op_new._diff_lipschitz alias          |
        |                          |            | op_new.jacobian(arr) = op_old.jacobian(arr + \shift)            |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes        | op_new.grad(arr) = op_old.grad(arr + \shift)                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | QUADRATIC                | yes        |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR                   | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_SQUARE            | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_NORMAL            | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_UNITARY           | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_SELF_ADJOINT      | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_POSITIVE_DEFINITE | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | LINEAR_IDEMPOTENT        | no         |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
    """

    def __init__(self, op: pyct.OpT, cst: pyct.NDArray):
        super().__init__()
        self._op = op._squeeze()
        assert cst.size == len(cst), f"cst: expected 1D array, got {cst.shape}."
        self._cst = cst

    def op(self) -> pyct.OpT:
        xp = pycu.get_array_module(self._cst)
        norm = pycu.compute(xp.linalg.norm(self._cst))
        if np.isclose(float(norm), 0):
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
            pyco.Property.QUADRATIC,
        }

        properties = self._op.properties() & preserved
        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.apply(x)
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = self._op.lipschitz(**kwargs)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.prox(x, tau)
        out -= self._cst
        return out

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        x = arr.copy()
        x += self._cst
        op = self._op.jacobian(x)
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._op.diff_lipschitz(**kwargs)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.grad(x)
        return out


class AddRule(Rule):
    r"""
    The output type of AddRule(A._squeeze(), B._squeeze()) is summarized in the table below (LHS/RHS
    commute):

        |---------------|-----|------|---------|----------|----------|--------------|-----------|---------|--------------|------------|------------|------------|---------------|---------------|------------|---------------|
        |   LHS / RHS   | Map | Func | DiffMap | DiffFunc | ProxFunc | ProxDiffFunc | Quadratic |  LinOp  |   LinFunc    |  SquareOp  |  NormalOp  |   UnitOp   | SelfAdjointOp |    PosDefOp   |   ProjOp   |   OrthProjOp  |
        |---------------|-----|------|---------|----------|----------|--------------|-----------|---------|--------------|------------|------------|------------|---------------|---------------|------------|---------------|
        | Map           | Map | Map  | Map     | Map      | Map      | Map          | Map       | Map     | Map          | Map        | Map        | Map        | Map           | Map           | Map        | Map           |
        | Func          |     | Func | Map     | Func     | Func     | Func         | Func      | Map     | Func         | Map        | Map        | Map        | Map           | Map           | Map        | Map           |
        | DiffMap       |     |      | DiffMap | DiffMap  | Map      | DiffMap      | DiffMap   | DiffMap | DiffMap      | DiffMap    | DiffMap    | DiffMap    | DiffMap       | DiffMap       | DiffMap    | DiffMap       |
        | DiffFunc      |     |      |         | DiffFunc | Func     | DiffFunc     | DiffFunc  | DiffMap | DiffFunc     | DiffMap    | DiffMap    | DiffMap    | DiffMap       | DiffMap       | DiffMap    | DiffMap       |
        | ProxFunc      |     |      |         |          | Func     | Func         | Func      | Map     | ProxFunc     | Map        | Map        | Map        | Map           | Map           | Map        | Map           |
        | ProxDiffFunc  |     |      |         |          |          | DiffFunc     | DiffFunc  | DiffMap | ProxDiffFunc | DiffMap    | DiffMap    | DiffMap    | DiffMap       | DiffMap       | DiffMap    | DiffMap       |
        | Quadratic     |     |      |         |          |          |              | Quadratic | DiffMap | Quadratic    | DiffMap    | DiffMap    | DiffMap    | DiffMap       | DiffMap       | DiffMap    | DiffMap       |
        | LinOp         |     |      |         |          |          |              |           | LinOp   | LinOp        | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE    | IMPOSSIBLE    | IMPOSSIBLE | IMPOSSIBLE    |
        | LinFunc       |     |      |         |          |          |              |           |         | LinFunc      | SquareOp   | SquareOp   | SquareOp   | SquareOp      | SquareOp      | SquareOp   | SquareOp      |
        | SquareOp      |     |      |         |          |          |              |           |         |              | SquareOp   | SquareOp   | SquareOp   | SquareOp      | SquareOp      | SquareOp   | SquareOp      |
        | NormalOp      |     |      |         |          |          |              |           |         |              |            | NormalOp   | NormalOp   | NormalOp      | NormalOp      | SquareOp   | NormalOp      |
        | UnitOp        |     |      |         |          |          |              |           |         |              |            |            | NormalOp   | NormalOp      | NormalOp      | SquareOp   | NormalOp      |
        | SelfAdjointOp |     |      |         |          |          |              |           |         |              |            |            |            | SelfAdjointOp | SelfAdjointOp | SquareOp   | SelfAdjointOp |
        | PosDefOp      |     |      |         |          |          |              |           |         |              |            |            |            |               | PosDefOp      | SquareOp   | PosDefOp      |
        | ProjOp        |     |      |         |          |          |              |           |         |              |            |            |            |               |               | SquareOp   | SquareOp      |
        | OrthProjOp    |     |      |         |          |          |              |           |         |              |            |            |            |               |               |            | SelfAdjointOp |
        |---------------|-----|------|---------|----------|----------|--------------|-----------|---------|--------------|------------|------------|------------|---------------|---------------|------------|---------------|


    The output properties however can be inferred based on the following simplified dispatch table:

        |--------------------------|---------------------|-------------------------------------|
        |      LHS Properties      |    RHS Properties   |          Output Properties          |
        |--------------------------|---------------------|-------------------------------------|
        | LHS                      | RHS                 | LHS.properties() & RHS.properties() |
        |--------------------------|---------------------|-------------------------------------|
        | PROXIMABLE               | PROXIMABLE &        | - PROXIMABLE                        |
        |                          | NOT LINEAR          |                                     |
        |--------------------------|---------------------|-------------------------------------|
        | QUADRATIC                | LINEAR &            | + QUADRATIC                         |
        |                          | FUNCTIONAL          |                                     |
        |--------------------------|---------------------|-------------------------------------|
        | LINEAR_UNITARY           | LINEAR_UNITARY      | - LINEAR_UNITARY                    |
        |                          |                     |                                     |
        |--------------------------|---------------------|-------------------------------------|
        | LINEAR_IDEMPOTENT        | LINEAR_IDEMPOTENT   | - LINEAR_IDEMPOTENT                 |
        |                          |                     |                                     |
        |--------------------------|---------------------|-------------------------------------|
        | LINEAR_POSITIVE_DEFINITE | LINEAR_IDEMPOTENT & | + LINEAR_POSITIVE_DEFINITE          |
        |                          | LINEAR_SELF_ADJOINT |                                     |
        |--------------------------|---------------------|-------------------------------------|


    Caution: the dispatch table should be read top-down and all rows must be executed.
    Ex: if LHS/RHS satisfy last row, then the rule is:

        (LHS.properties() & RHS.properties()) + LINEAR_POSITIVE_DEFINITE


    Arithmetic Update Rule(s)
    -------------------------
    * CAN_EVAL
        op.apply(arr) = _lhs.apply(arr) + _rhs.apply(arr)
        op._lipschitz = _lhs._lipschitz + _rhs._lipschitz
        op.lipschitz()
            = _lhs.lipschitz() + _rhs.lipschitz()
            + update op._lipschitz
        IMPORTANT: if range-broadcasting takes place (ex: LHS(1,) + RHS(M,)), then the broadcasted
                   operand's Lipschitz constant must be magnified by \sqrt{M}.

    * PROXIMABLE
        op.prox(arr, tau) = _lhs.prox(arr - tau * _rhs.grad(arr), tau)
                      OR  = _rhs.prox(arr - tau * _lhs.grad(arr), tau)
            IMPORTANT: the one calling .grad() should be either (lhs, rhs) which has LINEAR property

    * DIFFERENTIABLE
        op._diff_lipschitz = _lhs._diff_lipschitz + _rhs._diff_lipschitz
        op.diff_lipschitz()
            = _lhs.diff_lipschitz() + _rhs.diff_lipschitz()
            + update op._diff_lipschitz
        op.jacobian(arr) = _lhs.jacobian(arr) + _rhs.jacobian(arr)

    * DIFFERENTIABLE_FUNCTION
        op.grad(arr) = _lhs.grad(arr) + _rhs.grad(arr)

    * LINEAR
        op.adjoint(arr) = _lhs.adjoint(arr) + _rhs.adjoint(arr)
    """

    def __init__(self, lhs: pyct.OpT, rhs: pyct.OpT):
        super().__init__()
        self._lhs = lhs._squeeze()
        self._rhs = rhs._squeeze()

    def op(self) -> pyct.OpT:
        klass = self._infer_op_klass()
        sh_op = pycu.infer_sum_shape(self._lhs.shape, self._rhs.shape)
        op = klass(shape=sh_op)
        op._lhs = self._lhs  # embed for introspection
        op._rhs = self._rhs  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_attributes():
                attr = getattr(self, name)
                setattr(op, name, attr)
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(op, name, types.MethodType(func, op))
        return op

    def _infer_op_klass(self) -> pyct.OpC:
        lhs_p = self._lhs.properties()
        rhs_p = self._rhs.properties()
        base = set(lhs_p & rhs_p)
        base.discard(pyco.Property.LINEAR_UNITARY)
        base.discard(pyco.Property.LINEAR_IDEMPOTENT)
        if pyco.Property.LINEAR_SELF_ADJOINT in base:
            # orth-proj + pos-def => pos-def
            posd = pyco.Property.LINEAR_POSITIVE_DEFINITE
            idem = pyco.Property.LINEAR_IDEMPOTENT
            if any(
                [
                    (posd in lhs_p) and (idem in rhs_p),
                    (idem in lhs_p) and (posd in rhs_p),
                ]
            ):
                base.add(posd)
        if pyco.Property.PROXIMABLE in base:
            if pyco.Property.LINEAR not in (lhs_p | rhs_p):
                # .prox() only preserved when doing (prox + linfunc)
                base.discard(pyco.Property.PROXIMABLE)
            elif pyco.Property.QUADRATIC in (lhs_p | rhs_p):
                # quadraticity preserved when doing (quadratic + linear)
                base.add(pyco.Property.QUADRATIC)

        klass = pyco.Operator._infer_operator_type(base)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # ranges may broadcast, so can't do in-place updates.
        out_lhs = self._lhs.apply(arr)
        out_rhs = self._rhs.apply(arr)
        out = out_lhs + out_rhs
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        L_lhs = self._lhs.lipschitz(**kwargs)
        L_rhs = self._rhs.lipschitz(**kwargs)
        if self._lhs.codim < self._rhs.codim:
            # LHS broadcasts
            L_lhs *= np.sqrt(self._rhs.codim)
        elif self._lhs.codim > self._rhs.codim:
            # RHS broadcasts
            L_rhs *= np.sqrt(self._lhs.codim)

        self._lipschitz = L_lhs + L_rhs
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if pyco.Property.LINEAR in self._lhs.properties():
            P, G = self._rhs, self._lhs
        elif pyco.Property.LINEAR in self._rhs.properties():
            P, G = self._lhs, self._rhs
        else:
            raise NotImplementedError

        x = pycu.copy_if_unsafe(G.grad(arr))
        x *= -tau
        x += arr
        out = P.prox(x, tau)
        return out

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        op_lhs = self._lhs.jacobian(arr)
        op_rhs = self._rhs.jacobian(arr)
        op = op_lhs + op_rhs
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        self._diff_lipschitz = self._lhs.diff_lipschitz(**kwargs)
        self._diff_lipschitz += self._rhs.diff_lipschitz(**kwargs)
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        # ranges may broadcast, so can't do in-place updates.
        out_lhs = self._lhs.grad(arr)
        out_rhs = self._rhs.grad(arr)
        out = out_lhs + out_rhs
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        # ranges may broadcast, so can't do in-place updates.
        out_lhs = self._lhs.adjoint(arr)
        out_rhs = self._rhs.adjoint(arr)
        out = out_lhs + out_rhs
        return out


class ChainRule(Rule):
    def __init__(self, lhs: pyct.OpT, rhs: pyct.OpT):
        super().__init__()
        self._lhs = lhs._squeeze()
        self._rhs = rhs._squeeze()

    def op(self) -> pyct.OpT:
        identity_p = {  # identity matrix properties
            pyco.Property.LINEAR_POSITIVE_DEFINITE,
            pyco.Property.LINEAR_UNITARY,
        }
        if self._lhs.has(identity_p):
            op = self._rhs
        elif self._rhs.has(identity_p):
            op = self._lhs
        else:
            klass = self._infer_op_klass()
            sh_op = pycu.infer_composition_shape(self._lhs.shape, self._rhs.shape)
            op = klass(shape=sh_op)
            op._lhs = self._lhs  # embed for introspection
            op._rhs = self._rhs  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_attributes():
                    attr = getattr(self, name)
                    setattr(op, name, attr)
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
        return op

    def _infer_op_klass(self) -> pyct.OpC:
        base = set(self._lhs.properties())
        if self._lhs.has(pyco.Map.properties()):
            pass
        elif self._lhs.has(pyco.Func.properties()):
            pass
        elif self._lhs.has(pyco.DiffMap.properties()):
            if not self._rhs.has(pyco.Property.DIFFERENTIABLE):
                base.discard(pyco.Property.DIFFERENTIABLE)
        elif self._lhs.has(pyco.DiffFunc.properties()):
            if not self._rhs.has(pyco.Property.DIFFERENTIABLE):
                base.discard(pyco.Property.DIFFERENTIABLE)
                base.discard(pyco.Property.DIFFERENTIABLE_FUNCTION)
        elif self._lhs.has(pyco.ProxFunc.properties()):
            if not self._rhs.has(pyco.Property.LINEAR_UNITARY):
                base.discard(pyco.Property.PROXIMABLE)
        elif self._lhs.has(pyco.ProxDiffFunc.properties()):
            if self._rhs.has(pyco.Property.LINEAR_UNITARY):
                pass
            elif self._rhs.has(pyco.Property.DIFFERENTIABLE):
                base.discard(pyco.Property.PROXIMABLE)
            else:
                base = {
                    pyco.Property.CAN_EVAL,
                    pyco.Property.FUNCTIONAL,
                }
        elif self._lhs.has(pyco.QuadraticFunc.properties()):
            if self._rhs.has(pyco.Property.LINEAR):
                pass
            elif self._rhs.has(pyco.Property.DIFFERENTIABLE):
                base.discard(pyco.Property.PROXIMABLE)
                base.discard(pyco.Property.QUADRATIC)
            else:
                base = {
                    pyco.Property.CAN_EVAL,
                    pyco.Property.FUNCTIONAL,
                }
        elif self._lhs.has(pyco.LinOp.properties()):
            if self._rhs.has(pyco.LinOp.LINEAR):
                pass
            elif self._rhs.has(pyco.Property.DIFFERENTIABLE):
                base.discard(pyco.Property.LINEAR)
            else:
                base = {
                    pyco.Property.CAN_EVAL,
                }
        elif self._lhs.has(pyco.LinFunc.properties()):
            if self._rhs.has(pyco.Property.LINEAR):
                pass
            elif self._rhs.has(pyco.Property.QUADRATIC):
                base.discard(pyco.Property.LINEAR)
                base.add(pyco.Property.QUADRATIC)
            else:
                base.discard(pyco.Property.PROXIMABLE)
                base.discard(pyco.Property.LINEAR)
                base &= self._rhs.properties()
            base.add(pyco.Property.FUNCTIONAL)
        else:  # Necessarily a SquareOp sub-class
            if not self._rhs.has(pyco.Property.SQUARE):
                base &= self._rhs.properties()
            elif self._lhs.has(pyco.Property.LINEAR_UNITARY) and self._rhs.has(pyco.Property.LINEAR_UNITARY):
                base = pyco.UnitOp.properties()
            else:
                base = pyco.SquareOp.properties()

        klass = pyco.Operator._infer_operator_type(base)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._rhs.apply(arr)
        out = self._lhs.apply(x)
        return out

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = self._lhs.lipschitz(**kwargs)
        self._lipschitz *= self._rhs.lipschitz(**kwargs)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if all(
            [
                self._lhs.has(pyco.Property.PROXIMABLE),
                self._rhs.has(pyco.Property.LINEAR_UNITARY),
            ]
        ):
            x = self._rhs.apply(arr)
            y = self._lhs.prox(x, tau)
            out = self._rhs.adjoint(y)
            return out
        else:
            raise NotImplementedError

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        x = self._rhs.apply(arr)
        J_lhs = self._lhs.jacobian(x)
        J_rhs = self._rhs.jacobian(arr)
        J = J_lhs * J_rhs
        return J

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        if self._lhs.has(pyco.Property.LINEAR) and self._rhs.has(pyco.Property.LINEAR):
            self._diff_lipschitz = 0
        elif self._lhs.has(pyco.Property.LINEAR) and self._rhs.has(pyco.Property.DIFFERENTIABLE):
            self._diff_lipschitz = self._lhs.lipschitz(**kwargs)
            self._diff_lipschitz *= self._rhs.diff_lipschitz(**kwargs)
        elif self._lhs.has(pyco.Property.DIFFERENTIABLE) and self._rhs.has(pyco.Property.LINEAR):
            self._diff_lipschitz = self._lhs.diff_lipschitz(**kwargs)
            self._diff_lipschitz *= self._rhs.lipschitz(**kwargs) ** 2
        else:
            # @Matthieu: is this correct?
            self._diff_lipschitz = np.inf
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._rhs.apply(arr)
        y = self._lhs.grad(x)
        J_rhs = self._rhs.jacobian(arr)
        out = J_rhs.adjoint(y)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._lhs.adjoint(arr)
        out = self._rhs.adjoint(x)
        return out


class PowerRule(Rule):
    r"""
    Special Cases:
        k = 0  => IdentityOp
    Else:
        B = A \circ ... \circ A  (k-1 compositions)
    """

    def __init__(self, op: pyct.OpT, k: pyct.Integer):
        super().__init__()
        assert op.codim == op.dim, f"PowerRule: expected endomorphism, got {op}."
        assert int(k) >= 0, "PowerRule: only non-negative exponents are supported."
        self._op = op._squeeze()
        self._k = int(k)

    def op(self) -> pyct.OpT:
        if self._k == 0:
            from pycsou.operator.linop import IdentityOp

            op = IdentityOp(dim=self._op.codim)
        else:
            op = self._op
            if pyco.Property.LINEAR_IDEMPOTENT not in self._op.properties():
                for _ in range(self._k - 1):
                    op = ChainRule(self._op, op).op()
        return op
