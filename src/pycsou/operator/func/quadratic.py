import collections.abc as cabc

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "QuadraticFunc",
]


class QuadraticFunc(pyca._QuadraticFunc):
    r"""
    Quadratic functional.

    The quadratic functional is defined as:

    .. math::

        f(\mathbf{x})
        =
        \frac{1}{2} \langle\mathbf{x}, \mathbf{Q}\mathbf{x}\rangle
        +
        \mathbf{c}^T\mathbf{x}
        +
        t,
        \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    where
    :math:`Q` is a positive-definite operator :math:`\mathbf{Q}:\mathbb{R}^N\rightarrow\mathbb{R}^N`,
    :math:`\mathbf{c}\in\mathbb{R}^N`, and
    :math:`t>0`.

    Its gradient is given by:

    .. math::

        \nabla f(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}

    and proximity operator:

    .. math::

        \text{prox}_{\tau f}(x)
        =
        \left(
            \mathbf{Q} + \tau^{-1} \mathbf{Id}
        \right)^{-1}
        \left(
            \tau^{-1}\mathbf{x} - \mathbf{c}
        \right).

    In practice the proximity operator is evaluated via :py:class:`~pycsou.opt.solver.cg.CG`.

    The Lipschitz constant of a quadratic on an unbounded domain is unbounded.
    The Lipschitz constant of the gradient is given by the spectral norm of :math:`\mathbf{Q}`.

    Notes
    -----
    The ``QuadraticFunc`` type (QF) is preserved by the following operations:

    * ``QF(Q1, c1, t1) + QF(Q2, c2, t2) = QF(Q1+Q2, c1+c2, t1+t2)``
    * ``QF(Q, c, t) + LinFunc(d) = QF(Q, c+d, t)``
    * ``QF(Q, c, t) * a = QF(a*Q, a*c, a*t)``, :math:`a \in \mathbb{R}_{+}.
    * ``QF(Q, c, t) * A = QF(A*Q*A, A*c, t)``, :math:`A` positive-definite.
    * ``QF(Q, c, t).argscale(a) = QF(a**2 * Q, a*c, t)``, :math:`a \in \mathbb{R}`.
    * ``QF(Q, c, t).argshift(y) = QF(Q, c + Q*y, QF(Q, c, t).apply(y))``, :math:`y \in \mathbb{R}^{N}`.
    """

    def __init__(
        self,
        Q: pyca.PosDefOp,
        c: pyca.LinFunc,
        t: pyct.Real = 0,
        init_lipschitz: bool = True,
    ):
        r"""
        Parameters
        ----------
        Q: pyca.PosDefOp
            (N, N) positive-definite operator.
        c: pyca.LinFunc
            (1, N) linear functional
        t: pyct.Real
            offset
        init_lipschitz: bool
            Explicitly evaluate the Lipschitz constants.
        """
        super().__init__(shape=c.shape)
        self._Q = Q
        self._c = c
        self._t = pycrt.coerce(t)

        self._lipschitz = np.inf
        self._diff_lipschitz = self._Q.lipschitz() if init_lipschitz else np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = (arr * self._Q.apply(arr)).sum(axis=-1, keepdims=True)
        out /= 2
        out += self._c.apply(arr)
        out += self._t
        return out

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = pycu.copy_if_unsafe(self._Q.apply(arr))
        out += self._c.grad(arr)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        from pycsou.operator.linop import HomothetyOp
        from pycsou.opt.solver import CG

        A = self._Q + HomothetyOp(cst=1 / tau, dim=self._Q.dim)
        b = arr.copy()
        b /= tau
        b -= self._c.grad(arr)

        slvr = CG(A=A)
        slvr.fit(b=b)
        return slvr.solution()

    def _hessian(self) -> pyct.OpT:
        return self._Q
