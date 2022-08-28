"""
Operator Arithmetic.
"""

import copy
import types
import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class Rule:
    def __init__(self):
        # Default Arithmetic Attributes
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

    # Default Arithmetic Methods ----------------------------------------------
    # Fallback on these when no simple form in terms of Rule.__init__() args is known.
    # If a method from Property.arithmetic_methods() is not listed here, then all Rule subclasses
    # provide an overloaded implementation.

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

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

    def trace(self, **kwargs) -> pyct.Real:
        tr = self.__class__.trace(self, **kwargs)
        return float(tr)


class ScaleRule(Rule):
    r"""
    Special Cases:
        \alpha = 0  => NullOp/NullFunc
        \alpha = 1  => self
    Else:
        |--------------------------|-------------|--------------------------------------------------------------------|
        |         Property         |  Preserved? |                     Arithmetic Update Rule(s)                      |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | CAN_EVAL                 | yes         | op_new.apply(arr) = op_old.apply(arr) * \alpha                     |
        |                          |             | op_new._lipschitz = op_old._lipschitz * abs(\alpha)                |
        |                          |             |                                                                    |
        |                          |             | op_new.lipschitz()                                                 |
        |                          |             | = op_old.lipschitz() * abs(\alpha)                                 |
        |                          |             | + update op_new._lipschitz                                         |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | FUNCTIONAL               | yes         | op_new.asloss(\beta) = op_old.asloss(\beta) * \alpha               |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | PROXIMABLE               | \alpha > 0  | op_new.prox(arr, tau) = op_old.prox(arr, tau * \alpha)             |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | DIFFERENTIABLE           | yes         | op_new.jacobian(arr) = op_old.jacobian(arr) * \alpha               |
        |                          |             | op_new._diff_lipschitz = op_old._diff_lipschitz * abs(\alpha)      |
        |                          |             |                                                                    |
        |                          |             | diff_lipschitz()                                                   |
        |                          |             | = op_old.diff_lipschitz() * abs(\alpha)                            |
        |                          |             | + update op_new._diff_lipschitz                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes         | op_new.grad(arr) = op_old.grad(arr) * \alpha                       |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | QUADRATIC                | \alpha > 0  | op_new._hessian() = op_old._hessian() * \alpha                     |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR                   | yes         | op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha                 |
        |                          |             | op_new.asarray() = op_old.asarray() * \alpha                       |
        |                          |             | op_new.svdvals() = op_old.svdvals() * abs(\alpha)                  |
        |                          |             | op_new.pinv(x, damp) = op_old.pinv(x, damp / (\alpha**2)) / \alpha |
        |                          |             | op_new.gram() = op_old.gram() * (\alpha**2)                        |
        |                          |             | op_new.cogram() = op_old.cogram() * (\alpha**2)                        |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_SQUARE            | yes         | op_new.trace() = op_old.trace() * \alpha                           |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_NORMAL            | yes         | op_new.eigvals() = op_old.eigvals() * \alpha                       |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_UNITARY           | \alpha = -1 |                                                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_SELF_ADJOINT      | yes         |                                                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_POSITIVE_DEFINITE | \alpha > 0  |                                                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_IDEMPOTENT        | no          |                                                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
    """

    def __init__(self, op: pyct.OpT, cst: pyct.Real):
        super().__init__()
        self._op = op.squeeze()
        self._cst = float(cst)

    def op(self) -> pyct.OpT:
        if np.isclose(self._cst, 0):
            from pycsou.operator.linop import NullOp

            op = NullOp(shape=self._op.shape).squeeze()
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

    def _expr(self) -> tuple:
        return ("scale", self._op, self._cst)

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

    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = self._op.lipschitz(**kwargs)
        self._lipschitz *= abs(self._cst)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return self._op.prox(arr, tau * self._cst)

    def _hessian(self) -> pyct.OpT:
        return ScaleRule(op=self._op._hessian(), cst=self._cst).op()

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

    def asarray(self, **kwargs) -> pyct.NDArray:
        A = pycu.copy_if_unsafe(self._op.asarray(**kwargs))
        A *= self._cst
        return A

    def svdvals(self, **kwargs) -> pyct.NDArray:
        D = pycu.copy_if_unsafe(self._op.svdvals(**kwargs))
        D *= abs(self._cst)
        return D

    def eigvals(self, **kwargs) -> pyct.NDArray:
        D = pycu.copy_if_unsafe(self._op.eigvals(**kwargs))
        D *= self._cst
        return D

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        scale = kwargs.get("damp", 0) / (self._cst**2)
        out = pycu.copy_if_unsafe(self._op.pinv(arr, damp=scale))
        out /= self._cst
        return out

    def gram(self) -> pyct.OpT:
        op = self._op.gram() * (self._cst**2)
        return op

    def cogram(self) -> pyct.OpT:
        op = self._op.cogram() * (self._cst**2)
        return op

    def trace(self, **kwargs) -> pyct.Real:
        tr = self._op.trace(**kwargs) * self._cst
        return float(tr)

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        if self.has(pyco.Property.FUNCTIONAL):
            if data is None:
                op = self
            else:
                op = self._op.asloss(data) * self._cst
            return op
        else:
            raise NotImplementedError


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
        | FUNCTIONAL               | yes         | op_new.asloss(\beta) = op_old.argscale(\alpha).asloss(\beta)                |
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
        | QUADRATIC                | yes         | op_new._hessian() = op_old._hessian() * (\alpha**2)                         |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR                   | yes         | op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha                          |
        |                          |             | op_new.asarray() = op_old.asarray() * \alpha                                |
        |                          |             | op_new.svdvals() = op_old.svdvals() * abs(\alpha)                           |
        |                          |             | op_new.pinv(x, damp) = op_old.pinv(x, damp / (\alpha**2)) / \alpha          |
        |                          |             | op_new.gram() = op_old.gram() * (\alpha**2)                                 |
        |                          |             | op_new.cogram() = op_old.cogram() * (\alpha**2)                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_SQUARE            | yes         | op_new.trace() = op_old.trace() * \alpha                                    |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | LINEAR_NORMAL            | yes         | op_new.eigvals() = op_old.eigvals() * \alpha                                |
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
        self._op = op.squeeze()
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

    def _expr(self) -> tuple:
        return ("argscale", self._op, self._cst)

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

    def _hessian(self) -> pyct.OpT:
        return ScaleRule(op=self._op._hessian(), cst=self._cst**2).op()

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

    def asarray(self, **kwargs) -> pyct.NDArray:
        A = pycu.copy_if_unsafe(self._op.asarray(**kwargs))
        A *= self._cst
        return A

    def svdvals(self, **kwargs) -> pyct.NDArray:
        D = pycu.copy_if_unsafe(self._op.svdvals(**kwargs))
        D *= abs(self._cst)
        return D

    def eigvals(self, **kwargs) -> pyct.NDArray:
        D = pycu.copy_if_unsafe(self._op.eigvals(**kwargs))
        D *= self._cst
        return D

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        scale = kwargs.get("damp", 0) / (self._cst**2)
        out = pycu.copy_if_unsafe(self._op.pinv(arr, damp=scale))
        out /= self._cst
        return out

    def gram(self) -> pyct.OpT:
        op = self._op.gram() * (self._cst**2)
        return op

    def cogram(self) -> pyct.OpT:
        op = self._op.cogram() * (self._cst**2)
        return op

    def trace(self, **kwargs) -> pyct.Real:
        tr = self._op.trace(**kwargs) * self._cst
        return float(tr)

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        if self.has(pyco.Property.FUNCTIONAL):
            op = self._op.__class__.asloss(self, data=data)
            return op
        else:
            raise NotImplementedError


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
        | FUNCTIONAL               | yes        | op_new.asloss(\beta) = op_old.argshift(\shift).asloss(\beta)    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | PROXIMABLE               | yes        | op_new.prox(arr, tau) = op_old.prox(arr + \shift, tau) - \shift |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE           | yes        | op_new._diff_lipschitz = op_old._diff_lipschitz                 |
        |                          |            | op_new.diff_lipschitz() = op_new._diff_lipschitz alias          |
        |                          |            | op_new.jacobian(arr) = op_old.jacobian(arr + \shift)            |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes        | op_new.grad(arr) = op_old.grad(arr + \shift)                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | QUADRATIC                | yes        | op_new._hessian() = op_old._hessian()                           |
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

    def __init__(self, op: pyct.OpT, cst: typ.Union[pyct.Real, pyct.NDArray]):
        super().__init__()
        self._op = op.squeeze()
        self._scalar = isinstance(cst, pyct.Real)
        if self._scalar:
            cst = float(cst)
        else:  # pyct.NDArray
            assert cst.size == len(cst), f"cst: expected 1D array, got {cst.shape}."
        self._cst = cst

    def op(self) -> pyct.OpT:
        kwargs = dict(fallback=np if self._scalar else None)
        xp = pycu.get_array_module(self._cst, **kwargs)
        norm = pycu.compute(xp.linalg.norm(self._cst))
        if np.isclose(float(norm), 0):
            op = self._op
        else:
            klass = self._infer_op_klass()
            shape = self._infer_op_shape()
            op = klass(shape=shape)
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

    def _expr(self) -> tuple:
        sh = (None,) if isinstance(self._cst, pyct.Real) else self._cst.shape
        return ("argshift", self._op, sh)

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

    def _infer_op_shape(self) -> pyct.Shape:
        if self._scalar:
            return self._op.shape
        else:  # pyct.NDArray
            dim_op = self._op.dim
            dim_cst = self._cst.size
            if (dim_op is None) or (dim_op == dim_cst):
                return (self._op.codim, dim_cst)
            else:
                raise ValueError(f"Shifting {self._op} by {self._cst.shape} forbidden.")

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.apply(x)
        return out

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

    def _hessian(self) -> pyct.OpT:
        return self._op._hessian()

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

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        if self.has(pyco.Property.FUNCTIONAL):
            op = self._op.__class__.asloss(self, data=data)
            return op
        else:
            raise NotImplementedError


class AddRule(Rule):
    r"""
    The output type of AddRule(A.squeeze(), B.squeeze()) is summarized in the table below (LHS/RHS
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
        | NormalOp      |     |      |         |          |          |              |           |         |              |            | SquareOp   | SquareOp   | SquareOp      | SquareOp      | SquareOp   | SquareOp      |
        | UnitOp        |     |      |         |          |          |              |           |         |              |            |            | SquareOp   | SquareOp      | SquareOp      | SquareOp   | SquareOp      |
        | SelfAdjointOp |     |      |         |          |          |              |           |         |              |            |            |            | SelfAdjointOp | SelfAdjointOp | SquareOp   | SelfAdjointOp |
        | PosDefOp      |     |      |         |          |          |              |           |         |              |            |            |            |               | PosDefOp      | SquareOp   | PosDefOp      |
        | ProjOp        |     |      |         |          |          |              |           |         |              |            |            |            |               |               | SquareOp   | SquareOp      |
        | OrthProjOp    |     |      |         |          |          |              |           |         |              |            |            |            |               |               |            | SelfAdjointOp |
        |---------------|-----|------|---------|----------|----------|--------------|-----------|---------|--------------|------------|------------|------------|---------------|---------------|------------|---------------|


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

    * FUNCTIONAL
        op.asloss(\beta) = _lhs.asloss(\beta) + _rhs.asloss(\beta)

    * PROXIMABLE
        op.prox(arr, tau) = _lhs.prox(arr - tau * _rhs.grad(arr), tau)
                      OR  = _rhs.prox(arr - tau * _lhs.grad(arr), tau)
            IMPORTANT: the one calling .grad() should be either (lhs, rhs) which has LINEAR property

    * DIFFERENTIABLE
        op.jacobian(arr) = _lhs.jacobian(arr) + _rhs.jacobian(arr)
        op._diff_lipschitz = _lhs._diff_lipschitz + _rhs._diff_lipschitz
        op.diff_lipschitz()
            = _lhs.diff_lipschitz() + _rhs.diff_lipschitz()
            + update op._diff_lipschitz
        IMPORTANT: if range-broadcasting takes place (ex: LHS(1,) + RHS(M,)), then the broadcasted
                   operand's diff-Lipschitz constant must be magnified by \sqrt{M}.

    * DIFFERENTIABLE_FUNCTION
        op.grad(arr) = _lhs.grad(arr) + _rhs.grad(arr)

    * LINEAR
        op.adjoint(arr) = _lhs.adjoint(arr) + _rhs.adjoint(arr)
        IMPORTANT: if range-broadcasting takes place (ex: LHS(1,) + RHS(M,)), then the broadcasted
                   operand's adjoint-input must be averaged.
        op.asarray() = _lhs.asarray() + _rhs.asarray()
        op.gram() = _lhs.gram() + _rhs.gram() + (_lhs.T * _rhs) + (_rhs.T * _lhs)
        op.cogram() = _lhs.cogram() + _rhs.cogram() + (_lhs * _rhs.T) + (_rhs * _lhs.T)

    * LINEAR_SQUARE
        op.trace() = _lhs.trace() + _rhs.trace()

    * QUADRATIC
        op._hessian() = _lhs._hessian() + _rhs.hessian()
    """

    def __init__(self, lhs: pyct.OpT, rhs: pyct.OpT):
        super().__init__()
        self._lhs = lhs.squeeze()
        self._rhs = rhs.squeeze()

    def op(self) -> pyct.OpT:
        sh_op = pycu.infer_sum_shape(self._lhs.shape, self._rhs.shape)
        klass = self._infer_op_klass()
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

    def _expr(self) -> tuple:
        return ("add", self._lhs, self._rhs)

    def _infer_op_klass(self) -> pyct.OpC:
        P = pyco.Property
        lhs_p = self._lhs.properties()
        rhs_p = self._rhs.properties()
        base = set(lhs_p & rhs_p)
        base.discard(pyco.Property.LINEAR_NORMAL)
        base.discard(pyco.Property.LINEAR_UNITARY)
        base.discard(pyco.Property.LINEAR_IDEMPOTENT)
        base.discard(pyco.Property.PROXIMABLE)

        # Exceptions ----------------------------------------------------------
        # normality preserved for self-adjoint addition
        if P.LINEAR_SELF_ADJOINT in base:
            base.add(P.LINEAR_NORMAL)

        # orth-proj + pos-def => pos-def
        if (({P.LINEAR_IDEMPOTENT, P.LINEAR_SELF_ADJOINT} < lhs_p) and (P.LINEAR_POSITIVE_DEFINITE in rhs_p)) or (
            ({P.LINEAR_IDEMPOTENT, P.LINEAR_SELF_ADJOINT} < rhs_p) and (P.LINEAR_POSITIVE_DEFINITE in lhs_p)
        ):
            base.add(P.LINEAR_SQUARE)
            base.add(P.LINEAR_NORMAL)
            base.add(P.LINEAR_SELF_ADJOINT)
            base.add(P.LINEAR_POSITIVE_DEFINITE)

        # linfunc + (square-shape) => square
        if P.LINEAR in base:
            sh = pycu.infer_sum_shape(self._lhs.shape, self._rhs.shape)
            if (sh[0] == sh[1]) and (sh[0] > 1):
                base.add(P.LINEAR_SQUARE)

        # quadratic + quadratic => quadratic
        if P.QUADRATIC in base:
            base.add(P.PROXIMABLE)

        # quadratic + linfunc => quadratic
        if (P.PROXIMABLE in (lhs_p & rhs_p)) and ({P.QUADRATIC, P.LINEAR} < (lhs_p | rhs_p)):
            base.add(P.QUADRATIC)

        # prox(-diff) + linfunc => prox(-diff)
        if (P.PROXIMABLE in (lhs_p & rhs_p)) and (P.LINEAR in (lhs_p | rhs_p)):
            base.add(P.PROXIMABLE)

        klass = pyco.Operator._infer_operator_type(base)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        # ranges may broadcast, so can't do in-place updates.
        out_lhs = self._lhs.apply(arr)
        out_rhs = self._rhs.apply(arr)
        out = out_lhs + out_rhs
        return out

    def lipschitz(self, **kwargs) -> pyct.Real:
        if self.has(pyco.Property.LINEAR):
            if self.has(pyco.Property.FUNCTIONAL):
                self._lipschitz = pyco.LinFunc.lipschitz(self, **kwargs)
            else:
                kwargs = copy.copy(kwargs)
                kwargs.update(recompute=True)
                self._lipschitz = pyco.LinOp.lipschitz(self, **kwargs)
        else:
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
        P_LHS = self._lhs.properties()
        P_RHS = self._rhs.properties()
        if pyco.Property.LINEAR in (P_LHS | P_RHS):
            # linear + proximable
            if pyco.Property.LINEAR in P_LHS:
                P, G = self._rhs, self._lhs
            elif pyco.Property.LINEAR in P_RHS:
                P, G = self._lhs, self._rhs
            x = pycu.copy_if_unsafe(G.grad(arr))
            x *= -tau
            x += arr
            out = P.prox(x, tau)
        elif pyco.Property.QUADRATIC in (P_LHS & P_RHS):
            # quadratic + quadratic
            from pycsou.operator.func import QuadraticFunc

            x = np.zeros(shape=(self.dim), dtype=arr.dtype, like=arr)
            out = QuadraticFunc(
                Q=self._hessian(),
                c=self.jacobian(x),
                init_lipschitz=False,
            ).prox(arr, tau)
        else:
            raise NotImplementedError
        return out

    def _hessian(self) -> pyct.OpT:
        if self.has(pyco.Property.QUADRATIC):
            from pycsou.operator.linop import NullOp

            op_lhs = op_rhs = NullOp(shape=(self.dim, self.dim))
            if self._lhs.has(pyco.Property.QUADRATIC):
                op_lhs = self._lhs._hessian()
            if self._rhs.has(pyco.Property.QUADRATIC):
                op_rhs = self._rhs._hessian()
            op = op_lhs + op_rhs
            return op
        else:
            raise NotImplementedError

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        if self.has(pyco.Property.LINEAR):
            op = self
        else:
            op_lhs = self._lhs.jacobian(arr)
            op_rhs = self._rhs.jacobian(arr)
            op = op_lhs + op_rhs
        return op

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        L_lhs = self._lhs.diff_lipschitz(**kwargs)
        L_rhs = self._rhs.diff_lipschitz(**kwargs)
        if self._lhs.codim < self._rhs.codim:
            # LHS broadcasts
            L_lhs *= np.sqrt(self._rhs.codim)
        elif self._lhs.codim > self._rhs.codim:
            # RHS broadcasts
            L_rhs *= np.sqrt(self._lhs.codim)

        self._diff_lipschitz = L_lhs + L_rhs
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = self._lhs.grad(arr)
        out = pycu.copy_if_unsafe(out)
        out += self._rhs.grad(arr)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr_lhs = arr_rhs = arr
        if self._lhs.codim < self._rhs.codim:
            # LHS broadcasts
            arr_lhs = arr.sum(axis=-1, keepdims=True)
        elif self._lhs.codim > self._rhs.codim:
            # RHS broadcasts
            arr_rhs = arr.sum(axis=-1, keepdims=True)

        out = self._lhs.adjoint(arr_lhs)
        out = pycu.copy_if_unsafe(out)
        out += self._rhs.adjoint(arr_rhs)
        return out

    def asarray(self, **kwargs) -> pyct.NDArray:
        # broadcast may be involved, so can't do in-place updates.
        A_lhs = self._lhs.asarray(**kwargs)
        A_rhs = self._rhs.asarray(**kwargs)
        A = A_lhs + A_rhs
        return A

    def gram(self) -> pyct.OpT:
        lhs, rhs = self._lhs, self._rhs
        if self._lhs.codim == self._rhs.codim:
            # No broadcasting
            lhs_F = rhs_F = False
        else:
            # Broadcasting
            lhs_F = self._lhs.has(pyco.Property.FUNCTIONAL)
            rhs_F = self._rhs.has(pyco.Property.FUNCTIONAL)
            if lhs_F:
                lhs = _Sum(M=self._rhs.codim, N=1) * self._lhs
            if rhs_F:
                rhs = _Sum(M=self._lhs.codim, N=1) * self._rhs

        op1 = self._lhs.gram() * (self._rhs.codim if lhs_F else 1)
        op2 = self._rhs.gram() * (self._lhs.codim if rhs_F else 1)
        op3 = lhs.T * rhs
        op4 = rhs.T * lhs
        op = op1 + op2 + (op3 + op4).asop(pyco.SelfAdjointOp)
        return op.squeeze()

    def cogram(self) -> pyct.OpT:
        lhs, rhs = self._lhs, self._rhs
        if self._lhs.codim == self._rhs.codim:
            # No broadcasting
            lhs_F = rhs_F = False
        else:
            # Broadcasting
            lhs_F = self._lhs.has(pyco.Property.FUNCTIONAL)
            rhs_F = self._rhs.has(pyco.Property.FUNCTIONAL)
            if lhs_F:
                lhs = _Sum(M=self._rhs.codim, N=1) * self._lhs
            if rhs_F:
                rhs = _Sum(M=self._lhs.codim, N=1) * self._rhs

        if lhs_F:
            scale = float(self._lhs.cogram().asarray())
            op1 = _Sum(M=self._rhs.codim, N=self._rhs.codim) * scale
        else:
            op1 = self._lhs.cogram()
        if rhs_F:
            scale = float(self._rhs.cogram().asarray())
            op2 = _Sum(M=self._lhs.codim, N=self._lhs.codim) * scale
        else:
            op2 = self._rhs.cogram()
        op3 = lhs * rhs.T
        op4 = rhs * lhs.T
        op = op1 + op2 + (op3 + op4).asop(pyco.SelfAdjointOp)
        return op.squeeze()

    def trace(self, **kwargs) -> pyct.Real:
        tr = 0
        for side in (self._lhs, self._rhs):
            if side.has(pyco.Property.FUNCTIONAL):
                tr += float(side.asarray().sum())
            else:
                tr += float(side.trace(**kwargs))
        return tr

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        if self.has(pyco.Property.FUNCTIONAL):
            if data is None:
                op = self
            else:
                op_lhs = self._lhs.asloss(data=data)
                op_rhs = self._rhs.asloss(data=data)
                op = op_lhs + op_rhs
            return op
        else:
            raise NotImplementedError


class ChainRule(Rule):
    r"""
    The output type of ChainRule(A.squeeze(), B.squeeze()) is summarized in the table below:

        |---------------|------|------------|----------|------------|------------|----------------|----------------------|------------------|------------|-----------|-----------|--------------|---------------|-----------|-----------|------------|
        |   LHS / RHS   | Map  |    Func    | DiffMap  |  DiffFunc  |  ProxFunc  |  ProxDiffFunc  |      Quadratic       |      LinOp       |  LinFunc   |  SquareOp |  NormalOp |    UnitOp    | SelfAdjointOp |  PosDefOp |   ProjOp  | OrthProjOp |
        |---------------|------|------------|----------|------------|------------|----------------|----------------------|------------------|------------|-----------|-----------|--------------|---------------|-----------|-----------|------------|
        | Map           | Map  | Map        | Map      | Map        | Map        | Map            | Map                  | Map              | Map        | Map       | Map       | Map          | Map           | Map       | Map       | Map        |
        | Func          | Func | Func       | Func     | Func       | Func       | Func           | Func                 | Func             | Func       | Func      | Func      | Func         | Func          | Func      | Func      | Func       |
        | DiffMap       | Map  | Map        | DiffMap  | DiffMap    | Map        | DiffMap        | DiffMap              | DiffMap          | DiffMap    | DiffMap   | DiffMap   | DiffMap      | DiffMap       | DiffMap   | DiffMap   | DiffMap    |
        | DiffFunc      | Func | Func       | DiffFunc | DiffFunc   | Func       | DiffFunc       | DiffFunc             | DiffFunc         | DiffFunc   | DiffFunc  | DiffFunc  | DiffFunc     | DiffFunc      | DiffFunc  | DiffFunc  | DiffFunc   |
        | ProxFunc      | Func | Func       | Func     | Func       | Func       | Func           | Func                 | Func             | Func       | Func      | Func      | ProxFunc     | Func          | Func      | Func      | Func       |
        | ProxDiffFunc  | Func | Func       | DiffFunc | DiffFunc   | Func       | DiffFunc       | DiffFunc             | DiffFunc         | DiffFunc   | DiffFunc  | DiffFunc  | ProxDiffFunc | DiffFunc      | DiffFunc  | DiffFunc  | DiffFunc   |
        | Quadratic     | Func | Func       | DiffFunc | DiffFunc   | Func       | DiffFunc       | DiffFunc             | Quadratic        | Quadratic  | Quadratic | Quadratic | Quadratic    | Quadratic     | Quadratic | Quadratic | Quadratic  |
        | LinOp         | Map  | Func       | DiffMap  | DiffMap    | Map        | DiffMap        | DiffMap              | LinOp / SquareOp | LinOp      | LinOp     | LinOp     | LinOp        | LinOp         | LinOp     | LinOp     | LinOp      |
        | LinFunc       | Func | Func       | DiffFunc | DiffFunc   | [Prox]Func | [Prox]DiffFunc | DiffFunc / Quadratic | LinFunc          | LinFunc    | LinFunc   | LinFunc   | LinFunc      | LinFunc       | LinFunc   | LinFunc   | LinFunc    |
        | SquareOp      | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | NormalOp      | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | UnitOp        | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | UnitOp       | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | SelfAdjointOp | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | PosDefOp      | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | ProjOp        | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        | OrthProjOp    | Map  | IMPOSSIBLE | DiffMap  | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE     | IMPOSSIBLE           | LinOp            | IMPOSSIBLE | SquareOp  | SquareOp  | SquareOp     | SquareOp      | SquareOp  | SquareOp  | SquareOp   |
        |---------------|------|------------|----------|------------|------------|----------------|----------------------|------------------|------------|-----------|-----------|--------------|---------------|-----------|-----------|------------|


    Arithmetic Update Rule(s)
    -------------------------
    * CAN_EVAL
        op.apply(arr) = _lhs.apply(_rhs.apply(arr))
        op._lipschitz = _lhs._lipschitz * _rhs._lipschitz
        op.lipschitz()
            = _lhs.lipschitz() * _rhs.lipschitz()
            + update op._lipschitz

    * FUNCTIONAL
        op.asloss(\beta) = _lhs.asloss(\beta) * _rhs

    * PROXIMABLE (RHS Unitary only)
        op.prox(arr, tau) = _rhs.adjoint(_lhs.prox(_rhs.apply(arr), tau))

    * DIFFERENTIABLE
        op.jacobian(arr) = _lhs.jacobian(_rhs.apply(arr)) * _rhs.jacobian(arr)
        op._diff_lipschitz =
            linear \comp linear  => 0
            linear \comp diff    => _lhs._lipschitz * _rhs.diff_lipschitz
            diff   \comp linear  => _lhs._diff_lipschitz * (_rhs.lipschitz ** 2)
            diff   \comp diff    => \infty
        op.diff_lipschitz()
            = COMPLEX; see above.
            + update op._diff_lipschitz

    * DIFFERENTIABLE_FUNCTION (1D input)
        op.grad(arr) = _lhs.grad(_rhs.apply(arr)) @ _rhs.jacobian(arr).asarray()

    * LINEAR
        op.adjoint(arr) = _rhs.adjoint(_lhs.adjoint(arr))
        op.asarray() = _lhs.asarray() @ _rhs.asarray()
        op.gram() = _rhs.T @ _lhs.gram() @ _rhs
        op.cogram() = _lhs @ _rhs.cogram() @ _lhs.T

    * QUADRATIC
        op._hessian() = _rhs.T \comp _lhs._hessian() \comp _rhs [Positive-Definite]
    """

    def __init__(self, lhs: pyct.OpT, rhs: pyct.OpT):
        super().__init__()
        self._lhs = lhs.squeeze()
        self._rhs = rhs.squeeze()

    def op(self) -> pyct.OpT:
        sh_op = pycu.infer_composition_shape(self._lhs.shape, self._rhs.shape)
        klass = self._infer_op_klass()
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

    def _expr(self) -> tuple:
        return ("compose", self._lhs, self._rhs)

    def _infer_op_klass(self) -> pyct.OpC:
        # |--------------------------|------------------------------------------------------|
        # |         Property         |                      Preserved?                      |
        # |--------------------------|------------------------------------------------------|
        # | CAN_EVAL                 | (LHS CAN_EVAL) & (RHS CAN_EVAL)                      |
        # |--------------------------|------------------------------------------------------|
        # | FUNCTIONAL               | LHS FUNCTIONAL                                       |
        # |--------------------------|------------------------------------------------------|
        # | PROXIMABLE               | * (LHS PROXIMABLE) & (RHS LINEAR_UNITARY)            |
        # |                          | * (LHS LINEAR) & (RHS LINEAR)                        |
        # |                          | * (LHS LINEAR FUNCTIONAL [> 0]) & (RHS PROXIMABLE)   |
        # |--------------------------|------------------------------------------------------|
        # | DIFFERENTIABLE           | (LHS DIFFERENTIABLE) & (RHS DIFFERENTIABLE)          |
        # |--------------------------|------------------------------------------------------|
        # | DIFFERENTIABLE_FUNCTION  | (LHS DIFFERENTIABLE_FUNCTION) & (RHS DIFFERENTIABLE) |
        # |--------------------------|------------------------------------------------------|
        # | QUADRATIC                | * (LHS QUADRATIC) & (RHS LINEAR)                     |
        # |                          | * (LHS LINEAR FUNCTIONAL [> 0]) & (RHS QUADRATIC)    |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR                   | (LHS LINEAR) & (RHS LINEAR)                          |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_SQUARE            | (Shape[LHS * RHS] square) & (LHS.codim > 1)          |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_NORMAL            | no                                                   |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_UNITARY           | (LHS LINEAR_UNITARY) & (RHS LINEAR_UNITARY)          |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_SELF_ADJOINT      | no                                                   |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_POSITIVE_DEFINITE | no                                                   |
        # |--------------------------|------------------------------------------------------|
        # | LINEAR_IDEMPOTENT        | no                                                   |
        # |--------------------------|------------------------------------------------------|
        lhs_p = self._lhs.properties()
        rhs_p = self._rhs.properties()
        P = pyco.Property
        properties = {P.CAN_EVAL}
        if P.FUNCTIONAL in lhs_p:
            properties.add(P.FUNCTIONAL)
        # Proximal ------------------------------
        if (P.PROXIMABLE in lhs_p) and (P.LINEAR_UNITARY in rhs_p):
            properties.add(P.PROXIMABLE)
        elif ({P.LINEAR, P.FUNCTIONAL} < lhs_p) and (P.PROXIMABLE in rhs_p):
            cst = self._lhs.asarray().item()
            if cst > 0:
                properties.add(P.PROXIMABLE)
                if P.QUADRATIC in rhs_p:
                    properties.add(P.QUADRATIC)
        # ---------------------------------------
        if P.DIFFERENTIABLE in (lhs_p & rhs_p):
            properties.add(P.DIFFERENTIABLE)
        if (P.DIFFERENTIABLE_FUNCTION in lhs_p) and (P.DIFFERENTIABLE in rhs_p):
            properties.add(P.DIFFERENTIABLE_FUNCTION)
        if ((P.QUADRATIC in lhs_p) and (P.LINEAR in rhs_p)) or (
            ({P.LINEAR, P.FUNCTIONAL} < lhs_p) and (P.QUADRATIC in rhs_p)
        ):
            properties.add(P.PROXIMABLE)
            properties.add(P.QUADRATIC)
        if P.LINEAR in (lhs_p & rhs_p):
            properties.add(P.LINEAR)
            if self._lhs.codim == 1:
                properties.add(P.PROXIMABLE)
            if self._lhs.codim == self._rhs.dim > 1:
                properties.add(P.LINEAR_SQUARE)
        if P.LINEAR_UNITARY in (lhs_p & rhs_p):
            properties.add(P.LINEAR_NORMAL)
            properties.add(P.LINEAR_UNITARY)

        klass = pyco.Operator._infer_operator_type(properties)
        return klass

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._rhs.apply(arr)
        out = self._lhs.apply(x)
        return out

    def lipschitz(self, **kwargs) -> pyct.Real:
        if self.has(pyco.Property.LINEAR):
            if self.has(pyco.Property.FUNCTIONAL):
                self._lipschitz = pyco.LinFunc.lipschitz(self, **kwargs)
            else:
                kwargs = copy.copy(kwargs)
                kwargs.update(recompute=True)
                self._lipschitz = pyco.LinOp.lipschitz(self, **kwargs)
        else:
            self._lipschitz = self._lhs.lipschitz(**kwargs)
            self._lipschitz *= self._rhs.lipschitz(**kwargs)
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if self.has(pyco.Property.PROXIMABLE):
            out = None
            if self._lhs.has(pyco.Property.PROXIMABLE) and self._rhs.has(pyco.Property.LINEAR_UNITARY):
                # prox[diff]func() \comp unitop() => prox[diff]func()
                x = self._rhs.apply(arr)
                y = self._lhs.prox(x, tau)
                out = self._rhs.adjoint(y)
            elif self._lhs.has(pyco.Property.QUADRATIC) and self._rhs.has(pyco.Property.LINEAR):
                # quadratic \comp linop => quadratic
                from pycsou.operator.func import QuadraticFunc

                x = np.zeros(shape=(self._lhs.dim), dtype=arr.dtype, like=arr)
                out = QuadraticFunc(
                    Q=self._hessian(),
                    c=self._lhs.jacobian(x) * self._rhs,
                    init_lipschitz=False,
                ).prox(arr, tau)
            elif self._lhs.has(pyco.Property.LINEAR) and self._rhs.has(pyco.Property.PROXIMABLE):
                # linfunc() \comp prox[diff]func() => prox[diff]func()
                #                                  = (\alpha * prox[diff]func())
                op = ScaleRule(op=self._rhs, cst=self._lhs.asarray().item()).op()
                out = op.prox(arr, tau)
            elif pyco.Property.LINEAR in (self._lhs.properties() & self._rhs.properties()):
                # linfunc() \comp linop() => linfunc()
                out = pyco.LinFunc.prox(self, arr, tau)

            if out is not None:
                return out
        raise NotImplementedError

    def _hessian(self) -> pyct.OpT:
        if self.has(pyco.Property.QUADRATIC):
            if self._lhs.has(pyco.Property.LINEAR):
                # linfunc (scalar) \comp quadratic
                op = ScaleRule(op=self._rhs._hessian(), cst=self._lhs.asarray().item()).op()
            elif self._rhs.has(pyco.Property.LINEAR):
                # quadratic \comp linop
                op = (self._rhs.T * self._lhs._hessian() * self._rhs).asop(pyco.PosDefOp)
            return op
        else:
            raise NotImplementedError

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        if self.has(pyco.Property.LINEAR):
            op = self
        else:
            J_rhs = self._rhs.jacobian(arr)
            J_lhs = self._lhs.jacobian(self._rhs.apply(arr))
            op = J_lhs * J_rhs
        return op

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
            self._diff_lipschitz = np.inf
        return self._diff_lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._lhs.grad(self._rhs.apply(arr))
        if (arr.ndim == 1) or self._rhs.has(pyco.Property.LINEAR):
            out = self._rhs.jacobian(arr).adjoint(x)
        else:
            xp = pycu.get_array_module(arr)
            out = xp.stack(
                [
                    self._rhs.jacobian(a).adjoint(b)
                    for (a, b) in zip(
                        arr.reshape((np.prod(arr.shape[:-1]), -1)),
                        x.reshape((np.prod(x.shape[:-1]), -1)),
                        # zip() above safer than
                        #       zip(
                        #         arr.reshape((-1, self._rhs.dim)),
                        #         x.reshape((-1, self._lhs.dim)),
                        #       )
                        # Due to potential _lhs/_rhs domain-agnosticity, i.e. `[lhs|rhs].dim=None`.
                    )
                ],
                axis=0,
            ).reshape(arr.shape)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        x = self._lhs.adjoint(arr)
        out = self._rhs.adjoint(x)
        return out

    def asarray(self, **kwargs) -> pyct.NDArray:
        A_lhs = self._lhs.asarray(**kwargs)
        A_rhs = self._rhs.asarray(**kwargs)
        A = A_lhs @ A_rhs
        return A

    def gram(self) -> pyct.OpT:
        op = self._rhs.T * self._lhs.gram() * self._rhs
        return op.asop(pyco.SelfAdjointOp).squeeze()

    def cogram(self) -> pyct.OpT:
        op = self._lhs * self._rhs.cogram() * self._lhs.T
        return op.asop(pyco.SelfAdjointOp).squeeze()

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        if self.has(pyco.Property.FUNCTIONAL):
            if data is None:
                op = self
            else:
                op_lhs = self._lhs.asloss(data=data)
                op = op_lhs * self._rhs
            return op
        else:
            raise NotImplementedError


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
        self._op = op.squeeze()
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

            # Needed due to implicit PowerRule definition in terms of ChainRule.
            op._expr = self._expr
        return op

    def _expr(self) -> tuple:
        return ("exp", self._op, self._k)


# Helper Class/Functions ------------------------------------------------------
def _Sum(M: int, N: int) -> pyct.OpT:
    # f: \bR^{N} -> \bR^{M}
    #      x     -> [sum(x), ..., sum(x)]  (M times)
    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.broadcast_to(
            arr.sum(axis=-1, keepdims=True),
            (*arr.shape[:-1], _.codim),
        )
        return out

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.broadcast_to(
            arr.sum(axis=-1, keepdims=True),
            (*arr.shape[:-1], _.dim),
        )
        return out

    def op_gram(_) -> pyct.OpT:
        op = _Sum(M=_.dim, N=_.dim) * _.codim
        return op

    def op_cogram(_) -> pyct.OpT:
        op = _Sum(M=_.codim, N=_.codim) * _.dim
        return op

    if M == 1:
        klass = pyco.LinFunc
    elif M == N:
        klass = pyco.SelfAdjointOp
    else:
        klass = pyco.LinOp
    op = klass(shape=(M, N))
    op._lipschitz = np.sqrt(M * N)
    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.gram = types.MethodType(op_gram, op)
    op.cogram = types.MethodType(op_cogram, op)
    return op
