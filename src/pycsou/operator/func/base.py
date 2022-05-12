import typing as typ
import warnings

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class QuadraticFunc(pyco.ProxDiffFunc):
    r"""
    Quadratic functional.

    The quadratic functional is defined as:

    .. math::
        f(\mathbf{x})= \frac{1}{2} <\mathbf{x}, \mathbf{Q}\mathbf{x}> + \mathbf{c}^T\mathbf{x} + t, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    From a given self-adjoint, positive, nonzero, operator :math:`\mathbf{Q}:\mathbb{R}^N\rightarrow \mathbb{R}^N`, a vector
    :math:`\mathbf{c}\in\mathbb{R}^N` and a scalar :math:`t`.

    Its gradient is given by:

    .. math::
        \nabla f(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}

    and proximity operator:

    .. math::
        prox_{\tau f}(x) = \left(\mathbf{Q} + \frac{1}{tau} \mathcal{I}\right)^{-1} \left( \frac{\mathbf{x}}{tau} - \mathbf{c}\right).

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
    :: todo test documentation

    :py:class:`~pycsou.operator.func.base.QuadraticFunc` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``c`` used to initialize the :py:class:`~pycsou.operator.func.base.QuadraticFunc` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.func.base.QuadraticFunc` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``mat`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

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

        self._lipschitz = np.infty
        self._diff_lipschitz = self._Q._lipschitz

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self._c, self._t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return (self._Q(arr) * arr).sum(axis=-1, keepdims=True) / 2 + self._c(arr) + self._t

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self._c, self._t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return self._Q(arr) + self._c

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self._c, self._t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return pycs.cg.CG(self._Q + (1 / tau) * pycl.IdentityOp(self._Q.shape)).fit(arr / tau - self._c)
