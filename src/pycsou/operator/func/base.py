import typing as typ
import warnings

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.linop.base as pycl
import pycsou.opt.solver.cg as pycg
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class QuadraticFunc(pyco.ProxDiffFunc):
    r"""
    Quadratic functional.

    The quadratic functional is defined as:

    .. math::

        f(\mathbf{x})= \frac{1}{2} \langle\mathbf{x}, \mathbf{Q}\mathbf{x}\rangle + \mathbf{c}^T\mathbf{x} + t, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    for a given self-adjoint, positive, nonzero, operator :math:`\mathbf{Q}:\mathbb{R}^N\rightarrow \mathbb{R}^N`, a vector
    :math:`\mathbf{c}\in\mathbb{R}^N` and a scalar :math:`t>0`.

    Its gradient is given by:

    .. math::

        \nabla f(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}

    and proximity operator:

    .. math::

        \text{prox}_{\tau f}(x) = \left(\mathbf{Q} + \tau^{-1} \mathbf{Id}\right)^{-1} \left( \tau^{-1}\mathbf{x} - \mathbf{c}\right).

    In practice, the proximity operator is computed via conjugate gradient (:py:class:`pycsou.opt.solver.cg`).

    The lipschitz constant (``_lipschitz``) of a quadratic functional is unbounded when the domain of :math:`\mathbf{Q}`, is not bounded,
    and thus, it is set to infinity. The lipschitz constant of the gradient (``_diff_lipschitz``) is given by
    the spectral norm of :math:`\mathbf{Q}`.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinOp
    >>> from pycsou.operator.func.base import QuadraticFunc
    >>> import numpy as np
    >>> mat = np.arange(10).reshape(2,5)
    >>> A = ExplicitLinOp(mat)
    >>> b = np.arange(5).reshape(1,5)
    >>> LeastSquares = 2 * QuadraticFunc(Q=A.gram(), c= -A.adjoint(b), t=(1/2) * b.dot(b))

    Notes
    -----
    :: todo test with an example that calls the prox

    :py:class:`~pycsou.operator.func.base.QuadraticFunc` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``c`` used to initialize the :py:class:`~pycsou.operator.func.base.QuadraticFunc` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.func.base.QuadraticFunc` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``c`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    The ``QuadraticFunc`` type is preserved by the following operations:

    * The addition of two quadratic functionals ``(Q1, c1, t1)`` and ``(Q2, c2, t2)`` yiels another quadratic functional with
      parameters ``Q=Q1+Q2``, ``c=c1+c2`` and ``t=t1+t2``.
    * The addition of a quadratic functional and an :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc`
      with parameters ``(Q1, c1, t1)`` and ``vec`` yields another quadratic functional with parameters ``Q=Q1``, ``c=c1+vec`` and ``t=t1``.
    * The precomposition of a quadratic functional with parameters ``(Q1, c1, t1)`` and a :py:class:`~pycsou.abc.operator.SelfAdjointOp` ``A``
      yields another quadratic functional with parameters ``Q=A.T*Q1*A``, ``c=A.T(c1)`` and ``t=t1``.
    * Scaling a quadratic functional with parameters ``(Q1, c1, t1)`` by a scalar ``a``
      yields another quadratic functional with parameters ``Q=a*Q1``, ``c=a*c1`` and ``t=a*t1``.
    * Scaling the argument of a quadratic functional with parameters ``(Q1, c1, t1)`` by a scalar ``a``
      yields another quadratic functional with parameters ``Q=(a**2)*Q1``, ``c=a*c1`` and ``t=t1``.
    * Shifting a quadratic functional ``f`` with parameters ``(Q1, c1, t1)`` by a vector ``y``
      yields another quadratic functional with parameters ``Q=Q1``, ``c=c1 + Q1(y)`` and ``t=f(y)``.

    """

    def __init__(
        self, Q: pyco.SelfAdjointOp, c: pyct.NDArray, t: typ.Optional[pyct.Real] = None, enable_warnings: bool = True
    ):
        r"""

        Parameters
        ----------
        Q: Self-adjoint positive linear operator
            Self-adjoint, positive, nonzero, linear operator.
        c: NDArray
            (N,) vector. N-D input arrays are flattened. This is the vectorial representation of the linear term of the
             quadratic functional.
        t: Real
            Scalar term of the quadratic functional.
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.
        """
        super(QuadraticFunc, self).__init__(shape=(1, c.size))
        self._Q = Q
        self._c = pycrt.coerce(c)
        self._t = pycrt.coerce(t)
        self._enable_warnings = enable_warnings

        self._lipschitz = np.inf
        self._diff_lipschitz = self._Q._lipschitz

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not (self._c.dtype == pycrt.getPrecision().value or (self._c is None)) and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return (
            (self._Q(arr) * arr).sum(axis=-1, keepdims=True) / 2
            + (self._c * arr).sum(axis=-1, keepdims=True)
            + pycrt.coerce(self._t)
        )

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        if not (self._c.dtype == pycrt.getPrecision().value or (self._c is None)) and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return self._Q(arr) + self._c

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if not (self._c.dtype == pycrt.getPrecision().value or (self._c is None)) and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return pycg.CG(self._Q + (1 / tau) * pycl.IdentityOp(self._Q.shape)).fit(arr / tau - self._c)

    def __add__(self: "QuadraticFunc", other: pyco.MapLike) -> typ.Union[pyco.MapLike, "QuadraticFunc"]:

        if isinstance(other, QuadraticFunc):
            f = QuadraticFunc(
                Q=(self._Q + other._Q).specialize(cast_to=pyco.SelfAdjointOp),
                c=self._c + other._c,
                t=self._t + other._t,
            )
        elif isinstance(other, pycl.ExplicitLinFunc):
            f = QuadraticFunc(Q=self._Q, c=self._c + other.vec, t=self._t)
        else:
            f = pyco.ProxDiffFunc.__add__(self, other)
        return f.squeeze()

    def __mul__(self: "QuadraticFunc", other: pyco.MapLike) -> typ.Union[pyco.MapLike, "QuadraticFunc"]:
        if isinstance(other, pyct.Real):
            assert other > 0
            f = QuadraticFunc(
                Q=(self._Q * other).specialize(cast_to=pyco.SelfAdjointOp), c=self._c * other, t=self._t * other
            )
        elif isinstance(other, pyco.SelfAdjointOp):
            f = QuadraticFunc(
                Q=other.T * (self._Q * other).specialize(cast_to=pyco.SelfAdjointOp),
                c=other.adjoint(self._c),
                t=self._t,
            )
        else:
            f = pyco.ProxDiffFunc.__mul__(self, other)
        return f.squeeze()

    @pycrt.enforce_precision(i="scalar")
    def argscale(self: "QuadraticFunc", scalar: pyct.Real) -> "QuadraticFunc":
        return QuadraticFunc(
            Q=(self._Q * (scalar**2)).specialize(cast_to=pyco.SelfAdjointOp), c=self._c * scalar, t=self._t
        )

    @pycrt.enforce_precision(i="arr")
    def argshift(self: "QuadraticFunc", arr: pyct.NDArray) -> "QuadraticFunc":
        return QuadraticFunc(Q=self._Q, c=self._c + self._Q(arr), t=self(arr))
