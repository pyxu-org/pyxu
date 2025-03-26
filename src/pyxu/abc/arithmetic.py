# Arithmetic Rules

import types

import numpy as np

import pyxu.abc.operator as pxo
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu


class Rule:
    """
    General arithmetic rule.

    This class defines default arithmetic rules applicable unless re-defined by sub-classes.
    """

    def op(self) -> pxt.OpT:
        """
        Returns
        -------
        op: OpT
            Synthesize operator.
        """
        raise NotImplementedError

    # Helper Methods ----------------------------------------------------------

    @staticmethod
    def _propagate_constants(op: pxt.OpT):
        # Propagate (diff-)Lipschitz constants forward via special call to
        # Rule()-overridden `estimate_[diff_]lipschitz()` methods.

        # Important: we write to _[diff_]lipschitz to not overwrite estimate_[diff_]lipschitz() methods.
        if op.has(pxo.Property.CAN_EVAL):
            op._lipschitz = op.estimate_lipschitz(__rule=True)
        if op.has(pxo.Property.DIFFERENTIABLE):
            op._diff_lipschitz = op.estimate_diff_lipschitz(__rule=True)

    # Default Arithmetic Methods ----------------------------------------------
    # Fallback on these when no simple form in terms of Rule.__init__() args is known.
    # If a method from Property.arithmetic_methods() is not listed here, then all Rule subclasses
    # provide an overloaded implementation.

    def __call__(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = self.__class__.svdvals(self, **kwargs)
        return D

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.__class__.pinv(self, arr=arr, damp=damp, **kwargs)
        return out

    def trace(self, **kwargs) -> pxt.Real:
        tr = self.__class__.trace(self, **kwargs)
        return tr


class ScaleRule(Rule):
    r"""
    Arithmetic rules for element-wise scaling: :math:`B(x) = \alpha A(x)`.

    Special Cases::

        \alpha = 0  => NullOp/NullFunc
        \alpha = 1  => self

    Else::

        |--------------------------|-------------|--------------------------------------------------------------------|
        |         Property         |  Preserved? |                     Arithmetic Update Rule(s)                      |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | CAN_EVAL                 | yes         | op_new.apply(arr) = op_old.apply(arr) * \alpha                     |
        |                          |             | op_new.lipschitz = op_old.lipschitz * abs(\alpha)                  |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | FUNCTIONAL               | yes         |                                                                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | PROXIMABLE               | \alpha > 0  | op_new.prox(arr, tau) = op_old.prox(arr, tau * \alpha)             |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | DIFFERENTIABLE           | yes         | op_new.jacobian(arr) = op_old.jacobian(arr) * \alpha               |
        |                          |             | op_new.diff_lipschitz = op_old.diff_lipschitz * abs(\alpha)        |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes         | op_new.grad(arr) = op_old.grad(arr) * \alpha                       |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | QUADRATIC                | \alpha > 0  | Q, c, t = op_old._quad_spec()                                      |
        |                          |             | op_new._quad_spec() = (\alpha * Q, \alpha * c, \alpha * t)         |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR                   | yes         | op_new.adjoint(arr) = op_old.adjoint(arr) * \alpha                 |
        |                          |             | op_new.asarray() = op_old.asarray() * \alpha                       |
        |                          |             | op_new.svdvals() = op_old.svdvals() * abs(\alpha)                  |
        |                          |             | op_new.pinv(x, damp) = op_old.pinv(x, damp / (\alpha**2)) / \alpha |
        |                          |             | op_new.gram() = op_old.gram() * (\alpha**2)                        |
        |                          |             | op_new.cogram() = op_old.cogram() * (\alpha**2)                    |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_SQUARE            | yes         | op_new.trace() = op_old.trace() * \alpha                           |
        |--------------------------|-------------|--------------------------------------------------------------------|
        | LINEAR_NORMAL            | yes         |                                                                    |
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

    def __init__(self, op: pxt.OpT, cst: pxt.Real):
        super().__init__()
        self._op = op
        self._cst = float(cst)

    def op(self) -> pxt.OpT:
        if np.isclose(self._cst, 0):
            from pyxu.operator import NullOp

            op = NullOp(
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
            )
        elif np.isclose(self._cst, 1):
            op = self._op
        else:
            klass = self._infer_op_klass()
            op = klass(
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
            )
            op._op = self._op  # embed for introspection
            op._cst = self._cst  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
            self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("scale", self._op, self._cst)

    def _infer_op_klass(self) -> pxt.OpC:
        preserved = {
            pxo.Property.CAN_EVAL,
            pxo.Property.FUNCTIONAL,
            pxo.Property.DIFFERENTIABLE,
            pxo.Property.DIFFERENTIABLE_FUNCTION,
            pxo.Property.LINEAR,
            pxo.Property.LINEAR_SQUARE,
            pxo.Property.LINEAR_NORMAL,
            pxo.Property.LINEAR_SELF_ADJOINT,
        }
        if self._cst > 0:
            preserved |= {
                pxo.Property.LINEAR_POSITIVE_DEFINITE,
                pxo.Property.QUADRATIC,
                pxo.Property.PROXIMABLE,
            }
        if self._op.has(pxo.Property.LINEAR):
            preserved.add(pxo.Property.PROXIMABLE)
        if np.isclose(self._cst, -1):
            preserved.add(pxo.Property.LINEAR_UNITARY)

        properties = self._op.properties() & preserved
        klass = pxo.Operator._infer_operator_type(properties)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self._op.apply(arr))
        out *= self._cst
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L = float(self._op.lipschitz)
        else:
            L = self._op.estimate_lipschitz(**kwargs)
        L *= abs(self._cst)
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        return self._op.prox(arr, tau * self._cst)

    def _quad_spec(self):
        Q1, c1, t1 = self._op._quad_spec()
        Q2 = ScaleRule(op=Q1, cst=self._cst).op()
        c2 = ScaleRule(op=c1, cst=self._cst).op()
        t2 = t1 * self._cst
        return (Q2, c2, t2)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxo.Property.LINEAR):
            op = self
        else:
            op = self._op.jacobian(arr) * self._cst
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL = float(self._op.diff_lipschitz)
        else:
            dL = self._op.estimate_diff_lipschitz(**kwargs)
        dL *= abs(self._cst)
        return dL

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self._op.grad(arr))
        out *= self._cst
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self._op.adjoint(arr))
        out *= self._cst
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        A = pxu.copy_if_unsafe(self._op.asarray(**kwargs))
        A *= self._cst
        return A

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxu.copy_if_unsafe(self._op.svdvals(**kwargs))
        D *= abs(self._cst)
        return D

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        scale = damp / (self._cst**2)
        out = pxu.copy_if_unsafe(self._op.pinv(arr, damp=scale, **kwargs))
        out /= self._cst
        return out

    def gram(self) -> pxt.OpT:
        op = self._op.gram() * (self._cst**2)
        return op

    def cogram(self) -> pxt.OpT:
        op = self._op.cogram() * (self._cst**2)
        return op

    def trace(self, **kwargs) -> pxt.Real:
        tr = self._op.trace(**kwargs) * self._cst
        return tr


class ArgScaleRule(Rule):
    r"""
    Arithmetic rules for element-wise parameter scaling: :math:`B(x) = A(\alpha x)`.

    Special Cases::

        \alpha = 0  => ConstantValued (w/ potential vector-valued output)
        \alpha = 1  => self

    Else::

        |--------------------------|-------------|-----------------------------------------------------------------------------|
        |         Property         |  Preserved? |                          Arithmetic Update Rule(s)                          |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | CAN_EVAL                 | yes         | op_new.apply(arr) = op_old.apply(arr * \alpha)                              |
        |                          |             | op_new.lipschitz = op_old.lipschitz * abs(\alpha)                           |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | FUNCTIONAL               | yes         |                                                                             |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | PROXIMABLE               | yes         | op_new.prox(arr, tau) = op_old.prox(\alpha * arr, \alpha**2 * tau) / \alpha |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | DIFFERENTIABLE           | yes         | op_new.diff_lipschitz = op_old.diff_lipschitz * (\alpha**2)                 |
        |                          |             | op_new.jacobian(arr) = op_old.jacobian(arr * \alpha) * \alpha               |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes         | op_new.grad(arr) = op_old.grad(\alpha * arr) * \alpha                       |
        |--------------------------|-------------|-----------------------------------------------------------------------------|
        | QUADRATIC                | yes         | Q, c, t = op_old._quad_spec()                                               |
        |                          |             | op_new._quad_spec() = (\alpha**2 * Q, \alpha * c, t)                        |
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

    def __init__(self, op: pxt.OpT, cst: pxt.Real):
        super().__init__()
        self._op = op
        self._cst = float(cst)

    def op(self) -> pxt.OpT:
        if np.isclose(self._cst, 0):
            # ConstantVECTOR output: modify ConstantValued to work.
            from pyxu.operator import ConstantValued

            def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
                xp = pxu.get_array_module(arr)
                arr = xp.zeros_like(arr)
                out = self._op.apply(arr)
                return out

            op = ConstantValued(
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
                cst=self._cst,
            )
            op.apply = types.MethodType(op_apply, op)
            op._name = "ConstantVector"
        elif np.isclose(self._cst, 1):
            op = self._op
        else:
            klass = self._infer_op_klass()
            op = klass(
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
            )
            op._op = self._op  # embed for introspection
            op._cst = self._cst  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
            self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("argscale", self._op, self._cst)

    def _infer_op_klass(self) -> pxt.OpC:
        preserved = {
            pxo.Property.CAN_EVAL,
            pxo.Property.FUNCTIONAL,
            pxo.Property.PROXIMABLE,
            pxo.Property.DIFFERENTIABLE,
            pxo.Property.DIFFERENTIABLE_FUNCTION,
            pxo.Property.LINEAR,
            pxo.Property.LINEAR_SQUARE,
            pxo.Property.LINEAR_NORMAL,
            pxo.Property.LINEAR_SELF_ADJOINT,
            pxo.Property.QUADRATIC,
        }
        if self._cst > 0:
            preserved.add(pxo.Property.LINEAR_POSITIVE_DEFINITE)
        if np.isclose(self._cst, -1):
            preserved.add(pxo.Property.LINEAR_UNITARY)

        properties = self._op.properties() & preserved
        klass = pxo.Operator._infer_operator_type(properties)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = arr.copy()
        x *= self._cst
        out = self._op.apply(x)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L = float(self._op.lipschitz)
        else:
            L = self._op.estimate_lipschitz(**kwargs)
        L *= abs(self._cst)
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        x = arr.copy()
        x *= self._cst
        y = self._op.prox(x, (self._cst**2) * tau)
        out = pxu.copy_if_unsafe(y)
        out /= self._cst
        return out

    def _quad_spec(self):
        Q1, c1, t1 = self._op._quad_spec()
        Q2 = ScaleRule(op=Q1, cst=self._cst**2).op()
        c2 = ScaleRule(op=c1, cst=self._cst).op()
        t2 = t1
        return (Q2, c2, t2)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxo.Property.LINEAR):
            op = self
        else:
            x = arr.copy()
            x *= self._cst
            op = self._op.jacobian(x) * self._cst
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL = self._op.diff_lipschitz
        else:
            dL = self._op.estimate_diff_lipschitz(**kwargs)
        dL *= self._cst**2
        return dL

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = arr.copy()
        x *= self._cst
        out = pxu.copy_if_unsafe(self._op.grad(x))
        out *= self._cst
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self._op.adjoint(arr))
        out *= self._cst
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        A = pxu.copy_if_unsafe(self._op.asarray(**kwargs))
        A *= self._cst
        return A

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxu.copy_if_unsafe(self._op.svdvals(**kwargs))
        D *= abs(self._cst)
        return D

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        scale = damp / (self._cst**2)
        out = pxu.copy_if_unsafe(self._op.pinv(arr, damp=scale, **kwargs))
        out /= self._cst
        return out

    def gram(self) -> pxt.OpT:
        op = self._op.gram() * (self._cst**2)
        return op

    def cogram(self) -> pxt.OpT:
        op = self._op.cogram() * (self._cst**2)
        return op

    def trace(self, **kwargs) -> pxt.Real:
        tr = self._op.trace(**kwargs) * self._cst
        return tr


class ArgShiftRule(Rule):
    r"""
    Arithmetic rules for parameter shifting: :math:`B(x) = A(x + c)`.

    Special Cases::

        [NUMPY,CUPY] \shift = 0  => self
        [DASK]       \shift = 0  => rules below apply ...
                                    ... because we don't force evaluation of \shift for performance reasons.

    Else::

        |--------------------------|------------|-----------------------------------------------------------------|
        |         Property         | Preserved? |                    Arithmetic Update Rule(s)                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | CAN_EVAL                 | yes        | op_new.apply(arr) = op_old.apply(arr + \shift)                  |
        |                          |            | op_new.lipschitz = op_old.lipschitz                             |
        |--------------------------|------------|-----------------------------------------------------------------|
        | FUNCTIONAL               | yes        |                                                                 |
        |--------------------------|------------|-----------------------------------------------------------------|
        | PROXIMABLE               | yes        | op_new.prox(arr, tau) = op_old.prox(arr + \shift, tau) - \shift |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE           | yes        | op_new.diff_lipschitz = op_old.diff_lipschitz                   |
        |                          |            | op_new.jacobian(arr) = op_old.jacobian(arr + \shift)            |
        |--------------------------|------------|-----------------------------------------------------------------|
        | DIFFERENTIABLE_FUNCTION  | yes        | op_new.grad(arr) = op_old.grad(arr + \shift)                    |
        |--------------------------|------------|-----------------------------------------------------------------|
        | QUADRATIC                | yes        | Q, c, t = op_old._quad_spec()                                   |
        |                          |            | op_new._quad_spec() = (Q, c + Q @ \shift, op_old.apply(\shift)) |
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

    def __init__(self, op: pxt.OpT, cst: pxt.NDArray):
        super().__init__()
        self._op = op

        xp = pxu.get_array_module(cst)
        try:
            xp.broadcast_to(cst, op.dim_shape)
        except ValueError:
            error_msg = "`cst` must be broadcastable with operator dimensions: "
            error_msg += f"expected broadcastable-to {op.dim_shape}, got {cst.shape}."
            raise ValueError(error_msg)
        self._cst = cst

    def op(self) -> pxt.OpT:
        N = pxd.NDArrayInfo  # short-hand
        ndi = N.from_obj(self._cst)
        if ndi == N.DASK:
            no_op = False
        else:  # NUMPY/CUPY
            xp = ndi.module()
            norm = xp.sum(self._cst ** 2)
            no_op = xp.allclose(norm, 0)

        if no_op:
            op = self._op
        else:
            klass = self._infer_op_klass()
            op = klass(
                dim_shape=self._op.dim_shape,
                codim_shape=self._op.codim_shape,
            )
            op._op = self._op  # embed for introspection
            op._cst = self._cst  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
            self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("argshift", self._op, self._cst.shape)

    def _infer_op_klass(self) -> pxt.OpC:
        preserved = {
            pxo.Property.CAN_EVAL,
            pxo.Property.FUNCTIONAL,
            pxo.Property.PROXIMABLE,
            pxo.Property.DIFFERENTIABLE,
            pxo.Property.DIFFERENTIABLE_FUNCTION,
            pxo.Property.QUADRATIC,
        }

        properties = self._op.properties() & preserved
        klass = pxo.Operator._infer_operator_type(properties)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.apply(x)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L = self._op.lipschitz
        else:
            L = self._op.estimate_lipschitz(**kwargs)
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.prox(x, tau)
        out -= self._cst
        return out

    def _quad_spec(self):
        Q1, c1, t1 = self._op._quad_spec()

        xp = pxu.get_array_module(self._cst)
        cst = xp.broadcast_to(self._cst, self._op.dim_shape)

        Q2 = Q1
        c2 = c1 + pxo.LinFunc.from_array(
            A=Q1.apply(cst)[np.newaxis, ...],
            dim_rank=self._op.dim_rank,
            enable_warnings=False,
            # [enable_warnings] API users have no reason to call _quad_spec().
            # If they choose to use `c2`, then we assume they know what they are doing.
        )
        t2 = float(self._op.apply(cst)[0])

        return (Q2, c2, t2)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        x = arr.copy()
        x += self._cst
        op = self._op.jacobian(x)
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL = self._op.diff_lipschitz
        else:
            dL = self._op.estimate_diff_lipschitz(**kwargs)
        return dL

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = arr.copy()
        x += self._cst
        out = self._op.grad(x)
        return out


class AddRule(Rule):
    r"""
    Arithmetic rules for operator addition: :math:`C(x) = A(x) + B(x)`.

    The output type of ``AddRule(A, B)`` is summarized in the table below (LHS/RHS commute)::

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

    Arithmetic Update Rule(s)::

        * CAN_EVAL
            op.apply(arr) = _lhs.apply(arr) + _rhs.apply(arr)
            op.lipschitz = _lhs.lipschitz + _rhs.lipschitz
            IMPORTANT: if range-broadcasting takes place (ex: LHS(1,) + RHS(M,)), then the broadcasted
                       operand's Lipschitz constant must be magnified by \sqrt{M}.

        * PROXIMABLE
            op.prox(arr, tau) = _lhs.prox(arr - tau * _rhs.grad(arr), tau)
                          OR  = _rhs.prox(arr - tau * _lhs.grad(arr), tau)
                IMPORTANT: the one calling .grad() should be either (lhs, rhs) which has LINEAR property

        * DIFFERENTIABLE
            op.jacobian(arr) = _lhs.jacobian(arr) + _rhs.jacobian(arr)
            op.diff_lipschitz = _lhs.diff_lipschitz + _rhs.diff_lipschitz
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
            lhs = rhs = quadratic
              Q_l, c_l, t_l = lhs._quad_spec()
              Q_r, c_r, t_r = rhs._quad_spec()
              op._quad_spec() = (Q_l + Q_r, c_l + c_r, t_l + t_r)
            lhs, rhs = quadratic, linear
              Q, c, t = lhs._quad_spec()
              op._quad_spec() = (Q, c + rhs, t)
    """

    def __init__(self, lhs: pxt.OpT, rhs: pxt.OpT):
        assert lhs.dim_shape == rhs.dim_shape, "Operator dimensions are not compatible."
        try:
            codim_bcast = np.broadcast_shapes(lhs.codim_shape, rhs.codim_shape)
        except ValueError:
            error_msg = "`lhs/rhs` codims must be broadcastable: "
            error_msg += f"got {lhs.codim_shape}, {rhs.codim_shape}."
            raise ValueError(error_msg)

        if codim_bcast != lhs.codim_shape:
            from pyxu.operator import BroadcastAxes

            bcast = BroadcastAxes(
                dim_shape=lhs.codim_shape,
                codim_shape=codim_bcast,
            )
            lhs = bcast * lhs
        if codim_bcast != rhs.codim_shape:
            from pyxu.operator import BroadcastAxes

            bcast = BroadcastAxes(
                dim_shape=rhs.codim_shape,
                codim_shape=codim_bcast,
            )
            rhs = bcast * rhs

        super().__init__()
        self._lhs = lhs
        self._rhs = rhs

    def op(self) -> pxt.OpT:
        # LHS/RHS have same dim/codim following __init__()
        dim_shape = self._rhs.dim_shape
        codim_shape = self._rhs.codim_shape
        klass = self._infer_op_klass(dim_shape, codim_shape)

        if klass.has(pxo.Property.QUADRATIC):
            # Quadratic additive arithmetic differs substantially from other arithmetic operations.
            # To avoid tedious redefinitions of arithmetic methods to handle QuadraticFunc
            # specifically, the code-path below delegates additive arithmetic directly to
            # QuadraticFunc.
            lin = lambda _: _.has(pxo.Property.LINEAR)
            quad = lambda _: _.has(pxo.Property.QUADRATIC)

            if quad(self._lhs) and quad(self._rhs):
                lQ, lc, lt = self._lhs._quad_spec()
                rQ, rc, rt = self._rhs._quad_spec()
                op = klass(
                    dim_shape=dim_shape,
                    codim_shape=1,
                    Q=lQ + rQ,
                    c=lc + rc,
                    t=lt + rt,
                )
            elif quad(self._lhs) and lin(self._rhs):
                lQ, lc, lt = self._lhs._quad_spec()
                op = klass(
                    dim_shape=dim_shape,
                    codim_shape=1,
                    Q=lQ,
                    c=lc + self._rhs,
                    t=lt,
                )
            elif lin(self._lhs) and quad(self._rhs):
                rQ, rc, rt = self._rhs._quad_spec()
                op = klass(
                    dim_shape=dim_shape,
                    codim_shape=1,
                    Q=rQ,
                    c=self._lhs + rc,
                    t=rt,
                )
            else:
                raise ValueError("Impossible scenario: something went wrong during klass inference.")
        else:
            op = klass(
                dim_shape=dim_shape,
                codim_shape=codim_shape,
            )
            op._lhs = self._lhs  # embed for introspection
            op._rhs = self._rhs  # embed for introspection
            for p in op.properties():
                for name in p.arithmetic_methods():
                    func = getattr(self.__class__, name)
                    setattr(op, name, types.MethodType(func, op))
        self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("add", self._lhs, self._rhs)

    def _infer_op_klass(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ) -> pxt.OpC:
        P = pxo.Property
        lhs_p = self._lhs.properties()
        rhs_p = self._rhs.properties()
        base = set(lhs_p & rhs_p)
        base.discard(pxo.Property.LINEAR_NORMAL)
        base.discard(pxo.Property.LINEAR_UNITARY)
        base.discard(pxo.Property.LINEAR_IDEMPOTENT)
        base.discard(pxo.Property.PROXIMABLE)

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
            dim_size = np.prod(dim_shape)
            codim_size = np.prod(codim_shape)
            if (dim_size == codim_size) and (codim_shape != (1,)):
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

        klass = pxo.Operator._infer_operator_type(base)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self._lhs.apply(arr))
        out += self._rhs.apply(arr)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L_lhs = self._lhs.lipschitz
            L_rhs = self._rhs.lipschitz
        elif self.has(pxo.Property.LINEAR):
            L = self.__class__.estimate_lipschitz(self, **kwargs)
            return L
        else:
            L_lhs = self._lhs.estimate_lipschitz(**kwargs)
            L_rhs = self._rhs.estimate_lipschitz(**kwargs)

        L = L_lhs + L_rhs
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        P_LHS = self._lhs.properties()
        P_RHS = self._rhs.properties()
        if pxo.Property.LINEAR in (P_LHS | P_RHS):
            # linear + proximable
            if pxo.Property.LINEAR in P_LHS:
                P, G = self._rhs, self._lhs
            elif pxo.Property.LINEAR in P_RHS:
                P, G = self._lhs, self._rhs
            x = pxu.copy_if_unsafe(G.grad(arr))
            x *= -tau
            x += arr
            out = P.prox(x, tau)
        else:
            raise NotImplementedError
        return out

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxo.Property.LINEAR):
            op = self
        else:
            op_lhs = self._lhs.jacobian(arr)
            op_rhs = self._rhs.jacobian(arr)
            op = op_lhs + op_rhs
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            dL_lhs = self._lhs.diff_lipschitz
            dL_rhs = self._rhs.diff_lipschitz
        elif self.has(pxo.Property.LINEAR):
            dL_lhs = 0
            dL_rhs = 0
        else:
            dL_lhs = self._lhs.estimate_diff_lipschitz(**kwargs)
            dL_rhs = self._rhs.estimate_diff_lipschitz(**kwargs)

        dL = dL_lhs + dL_rhs
        return dL

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = self._lhs.grad(arr)
        out = pxu.copy_if_unsafe(out)
        out += self._rhs.grad(arr)
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = self._lhs.adjoint(arr)
        out = pxu.copy_if_unsafe(out)
        out += self._rhs.adjoint(arr)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        A = self._lhs.asarray(**kwargs)
        A = pxu.copy_if_unsafe(A)
        A += self._rhs.asarray(**kwargs)
        return A

    def gram(self) -> pxt.OpT:
        op1 = self._lhs.gram()
        op2 = self._rhs.gram()
        op3 = self._lhs.T * self._rhs
        op4 = self._rhs.T * self._lhs
        op = op1 + op2 + (op3 + op4).asop(pxo.SelfAdjointOp)
        return op

    def cogram(self) -> pxt.OpT:
        op1 = self._lhs.cogram()
        op2 = self._rhs.cogram()
        op3 = self._lhs * self._rhs.T
        op4 = self._rhs * self._lhs.T
        op = op1 + op2 + (op3 + op4).asop(pxo.SelfAdjointOp)
        return op

    def trace(self, **kwargs) -> pxt.Real:
        tr = 0
        for side in (self._lhs, self._rhs):
            tr += side.trace(**kwargs)
        return float(tr)


class ChainRule(Rule):
    r"""
    Arithmetic rules for operator composition: :math:`C(x) = (A \circ B)(x)`.

    The output type of ``ChainRule(A, B)`` is summarized in the table below::

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

    Arithmetic Update Rule(s)::

        * CAN_EVAL
            op.apply(arr) = _lhs.apply(_rhs.apply(arr))
            op.lipschitz = _lhs.lipschitz * _rhs.lipschitz

        * PROXIMABLE (RHS Unitary only)
            op.prox(arr, tau) = _rhs.adjoint(_lhs.prox(_rhs.apply(arr), tau))

        * DIFFERENTIABLE
            op.jacobian(arr) = _lhs.jacobian(_rhs.apply(arr)) * _rhs.jacobian(arr)
            op.diff_lipschitz =
                quadratic            => _quad_spec().Q.lipschitz
                linear \comp linear  => 0
                linear \comp diff    => _lhs.lipschitz * _rhs.diff_lipschitz
                diff   \comp linear  => _lhs.diff_lipschitz * (_rhs.lipschitz ** 2)
                diff   \comp diff    => \infty

        * DIFFERENTIABLE_FUNCTION (1D input)
            op.grad(arr) = _lhs.grad(_rhs.apply(arr)) @ _rhs.jacobian(arr).asarray()

        * LINEAR
            op.adjoint(arr) = _rhs.adjoint(_lhs.adjoint(arr))
            op.asarray() = _lhs.asarray() @ _rhs.asarray()
            op.gram() = _rhs.T @ _lhs.gram() @ _rhs
            op.cogram() = _lhs @ _rhs.cogram() @ _lhs.T

        * QUADRATIC
            Q, c, t = _lhs._quad_spec()
            op._quad_spec() = (_rhs.T * Q * _rhs, _rhs.T * c, t)
    """

    def __init__(self, lhs: pxt.OpT, rhs: pxt.OpT):
        assert lhs.dim_shape == rhs.codim_shape, "Operator dimensions are not compatible."

        super().__init__()
        self._lhs = lhs
        self._rhs = rhs

    def op(self) -> pxt.OpT:
        klass = self._infer_op_klass()
        op = klass(
            dim_shape=self._rhs.dim_shape,
            codim_shape=self._lhs.codim_shape,
        )
        op._lhs = self._lhs  # embed for introspection
        op._rhs = self._rhs  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(op, name, types.MethodType(func, op))
        self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("compose", self._lhs, self._rhs)

    def _infer_op_klass(self) -> pxt.OpC:
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
        P = pxo.Property
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
        if (P.QUADRATIC in lhs_p) and (P.LINEAR in rhs_p):
            properties.add(P.PROXIMABLE)
            properties.add(P.QUADRATIC)
        if P.LINEAR in (lhs_p & rhs_p):
            properties.add(P.LINEAR)
            if self._lhs.codim_shape == (1,):
                for p in pxo.LinFunc.properties():
                    properties.add(p)
            if (self._lhs.codim_size == self._rhs.dim_size) and (self._rhs.dim_size > 1):
                properties.add(P.LINEAR_SQUARE)
        if P.LINEAR_UNITARY in (lhs_p & rhs_p):
            properties.add(P.LINEAR_NORMAL)
            properties.add(P.LINEAR_UNITARY)

        klass = pxo.Operator._infer_operator_type(properties)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = self._rhs.apply(arr)
        out = self._lhs.apply(x)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L_lhs = self._lhs.lipschitz
            L_rhs = self._rhs.lipschitz
        elif self.has(pxo.Property.LINEAR):
            L = self.__class__.estimate_lipschitz(self, **kwargs)
            return L
        else:
            L_lhs = self._lhs.estimate_lipschitz(**kwargs)
            L_rhs = self._rhs.estimate_lipschitz(**kwargs)

        zeroQ = lambda _: np.isclose(_, 0)
        if zeroQ(L_lhs) or zeroQ(L_rhs):
            L = 0
        else:
            L = L_lhs * L_rhs
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        if self.has(pxo.Property.PROXIMABLE):
            out = None
            if self._lhs.has(pxo.Property.PROXIMABLE) and self._rhs.has(pxo.Property.LINEAR_UNITARY):
                # prox[diff]func() \comp unitop() => prox[diff]func()
                x = self._rhs.apply(arr)
                y = self._lhs.prox(x, tau)
                out = self._rhs.adjoint(y)
            elif self._lhs.has(pxo.Property.QUADRATIC) and self._rhs.has(pxo.Property.LINEAR):
                # quadratic \comp linop => quadratic
                Q, c, t = self._quad_spec()
                op = pxo.QuadraticFunc(
                    dim_shape=self.dim_shape,
                    codim_shape=self.codim_shape,
                    Q=Q,
                    c=c,
                    t=t,
                )
                out = op.prox(arr, tau)
            elif self._lhs.has(pxo.Property.LINEAR) and self._rhs.has(pxo.Property.PROXIMABLE):
                # linfunc() \comp prox[diff]func() => prox[diff]func()
                #                                  = (\alpha * prox[diff]func())
                op = ScaleRule(op=self._rhs, cst=self._lhs.asarray().item()).op()
                out = op.prox(arr, tau)
            elif pxo.Property.LINEAR in (self._lhs.properties() & self._rhs.properties()):
                # linfunc() \comp linop() => linfunc()
                out = pxo.LinFunc.prox(self, arr, tau)

            if out is not None:
                return out
        raise NotImplementedError

    def _quad_spec(self):
        if self.has(pxo.Property.QUADRATIC):
            if self._lhs.has(pxo.Property.LINEAR):
                # linfunc (scalar) \comp quadratic
                op = ScaleRule(op=self._rhs, cst=self._lhs.asarray().item()).op()
                Q2, c2, t2 = op._quad_spec()
            elif self._rhs.has(pxo.Property.LINEAR):
                # quadratic \comp linop
                Q1, c1, t1 = self._lhs._quad_spec()
                Q2 = (self._rhs.T * Q1 * self._rhs).asop(pxo.PosDefOp)
                c2 = c1 * self._rhs
                t2 = t1
            return (Q2, c2, t2)
        else:
            raise NotImplementedError

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        if self.has(pxo.Property.LINEAR):
            op = self
        else:
            J_rhs = self._rhs.jacobian(arr)
            J_lhs = self._lhs.jacobian(self._rhs.apply(arr))
            op = J_lhs * J_rhs
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if self.has(pxo.Property.QUADRATIC):
            Q, c, t = self._quad_spec()
            op = pxo.QuadraticFunc(
                dim_shape=self.dim_shape,
                codim_shape=self.codim_shape,
                Q=Q,
                c=c,
                t=t,
            )
            if no_eval:
                dL = op.diff_lipschitz
            else:
                dL = op.estimate_diff_lipschitz(**kwargs)
        elif self._lhs.has(pxo.Property.LINEAR) and self._rhs.has(pxo.Property.LINEAR):
            dL = 0
        elif self._lhs.has(pxo.Property.LINEAR) and self._rhs.has(pxo.Property.DIFFERENTIABLE):
            if no_eval:
                L_lhs = self._lhs.lipschitz
                dL_rhs = self._rhs.diff_lipschitz
            else:
                L_lhs = self._lhs.estimate_lipschitz(**kwargs)
                dL_rhs = self._rhs.estimate_diff_lipschitz(**kwargs)
            dL = L_lhs * dL_rhs
        elif self._lhs.has(pxo.Property.DIFFERENTIABLE) and self._rhs.has(pxo.Property.LINEAR):
            if no_eval:
                dL_lhs = self._lhs.diff_lipschitz
                L_rhs = self._rhs.lipschitz
            else:
                dL_lhs = self._lhs.estimate_diff_lipschitz(**kwargs)
                L_rhs = self._rhs.estimate_lipschitz(**kwargs)
            dL = dL_lhs * (L_rhs**2)
        else:
            dL = np.inf
        return dL

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]
        if (len(sh) == 0) or self._rhs.has(pxo.Property.LINEAR):
            x = self._lhs.grad(self._rhs.apply(arr))
            out = self._rhs.jacobian(arr).adjoint(x)

            # RHS.adjoint() may change core-chunks if (codim->dim) changes are involved.
            # This is problematic since grad() should preserve core-chunks by default.
            ndi = pxd.NDArrayInfo.from_obj(arr)
            if ndi == pxd.NDArrayInfo.DASK:
                if out.chunks != arr.chunks:
                    out = out.rechunk(arr.chunks)
        else:
            # We need to evaluate the Jacobian seperately per stacked input.

            @pxu.vectorize(
                i="arr",
                dim_shape=self.dim_shape,
                codim_shape=self.dim_shape,
            )
            def f(arr: pxt.NDArray) -> pxt.NDArray:
                x = self._lhs.grad(self._rhs.apply(arr))
                out = self._rhs.jacobian(arr).adjoint(x)

                # RHS.adjoint() may change core-chunks if (codim->dim) changes are involved.
                # This is problematic since grad() should preserve core-chunks by default.
                ndi = pxd.NDArrayInfo.from_obj(arr)
                if ndi == pxd.NDArrayInfo.DASK:
                    if out.chunks != arr.chunks:
                        out = out.rechunk(arr.chunks)
                return out

            out = f(arr)
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        x = self._lhs.adjoint(arr)
        out = self._rhs.adjoint(x)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        A_lhs = self._lhs.asarray(**kwargs)
        A_rhs = self._rhs.asarray(**kwargs)

        xp = pxu.get_array_module(A_lhs)
        A = xp.tensordot(A_lhs, A_rhs, axes=self._lhs.dim_rank)
        return A

    def gram(self) -> pxt.OpT:
        op = self._rhs.T * self._lhs.gram() * self._rhs
        return op.asop(pxo.SelfAdjointOp)

    def cogram(self) -> pxt.OpT:
        op = self._lhs * self._rhs.cogram() * self._lhs.T
        return op.asop(pxo.SelfAdjointOp)


class TransposeRule(Rule):
    # Not strictly-speaking an arithmetic method, but the logic behind constructing transposed
    # operators is identical to arithmetic methods.
    # LinOp.T() rules are hence summarized here.
    r"""
    Arithmetic rules for :py:class:`~pyxu.abc.LinOp` transposition: :math:`B(x) = A^{T}(x)`.

    Arithmetic Update Rule(s)::

        * CAN_EVAL
            opT.apply(arr) = op.adjoint(arr)
            opT.lipschitz = op.lipschitz

        * PROXIMABLE
            opT.prox(arr, tau) = LinFunc.prox(arr, tau)

        * DIFFERENTIABLE
            opT.jacobian(arr) = opT
            opT.diff_lipschitz = 0

        * DIFFERENTIABLE_FUNCTION
            opT.grad(arr) = LinFunc.grad(arr)

        * LINEAR
            opT.adjoint(arr) = op.apply(arr)
            opT.asarray() = op.asarray().T [block-reorder dim/codim]
            opT.gram() = op.cogram()
            opT.cogram() = op.gram()
            opT.svdvals() = op.svdvals()

        * LINEAR_SQUARE
            opT.trace() = op.trace()
    """

    def __init__(self, op: pxt.OpT):
        super().__init__()
        self._op = op

    def op(self) -> pxt.OpT:
        klass = self._infer_op_klass()
        op = klass(
            dim_shape=self._op.codim_shape,
            codim_shape=self._op.dim_shape,
        )
        op._op = self._op  # embed for introspection
        for p in op.properties():
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(op, name, types.MethodType(func, op))
        self._propagate_constants(op)
        return op

    def _expr(self) -> tuple:
        return ("transpose", self._op)

    def _infer_op_klass(self) -> pxt.OpC:
        # |--------------------------------|--------------------------------|
        # |      op_klass(codim; dim)      |     opT_klass(codim; dim)      |
        # |--------------------------------|--------------------------------|
        # | LINEAR(1; 1)                   | LinFunc(1; 1)                  |
        # | LinFunc(1; M1,...,MD)          | LinOp(M1,...,MD; 1)            |
        # | LinOp(N1,...,ND; 1)            | LinFunc(1; N1,...,ND)          |
        # | op_klass(N1,...,ND; M1,...,MD) | op_klass(M1,...,MD; N1,...,ND) |
        # |--------------------------------|--------------------------------|
        single_dim = self._op.dim_shape == (1,)
        single_codim = self._op.codim_shape == (1,)

        if single_dim and single_codim:
            klass = pxo.LinFunc
        elif single_codim:
            klass = pxo.LinOp
        elif single_dim:
            klass = pxo.LinFunc
        else:
            prop = self._op.properties()
            klass = pxo.Operator._infer_operator_type(prop)
        return klass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = self._op.adjoint(arr)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            L = self._op.lipschitz
        else:
            L = self._op.estimate_lipschitz(**kwargs)
        return L

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        out = pxo.LinFunc.prox(self, arr, tau)
        return out

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return self

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        return 0

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = pxo.LinFunc.grad(self, arr)
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = self._op.apply(arr)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        A = self._op.asarray(**kwargs)
        B = A.transpose(
            *range(-self._op.dim_rank, 0),
            *range(self._op.codim_rank),
        )
        return B

    def gram(self) -> pxt.OpT:
        op = self._op.cogram()
        return op

    def cogram(self) -> pxt.OpT:
        op = self._op.gram()
        return op

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = self._op.svdvals(**kwargs)
        return D

    def trace(self, **kwargs) -> pxt.Real:
        tr = self._op.trace(**kwargs)
        return tr
