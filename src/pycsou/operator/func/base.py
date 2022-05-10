import typing as typ
import warnings

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class QuadraticFunc(pyco.ProxDiffFunc):
    r"""
    Build a quadratic functional of the form:

    .. math::
        f(\mathbf{x})=<\mathbf{x}, \mathbf{Q}\mathbf{x}> + \mathbf{c}^T\mathbf{x} + t, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    From a given self-adjoint, positive operator :math:`\mathbf{Q}:\mathbb{R}^N\rightarrow \mathbb{R}^N`, a vector
    :math:`\mathbf{c}\in\mathbb{R}^N` and a scalar :math:`t`.

    with gradient given by:
    .. math::
        \nabla_{f}(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}

    and proximity operator given by:
    .. math::
        prox_{\tau f}(x) = (\mathbf{Q} + \frac{1}{tau} \mathcal{I})^{-1} ( \frac{\mathbf{x}}{tau} - \mathbf{c})

    Which in practice is solved via the conjugate gradient method (:py:class:`pycsou.opt.solver.cg`)

    The lipschitz constant (`_lipschitz`) of a quadratic functional is unbounded when the domain of :math:`\mathbf{Q}`, is not bounded,
    and thus, it is set to infinity. The lipschitz constant of the gradient (`_diff_lipschitz`) is given by the operator norm of
    :math:`\mathbf{Q}`.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinOp
    >>> from pycsou.operator.func.base import QuadraticFunc
    >>> import numpy as np
    >>> mat = np.arange(10).reshape(2,5)
    >>> A = ExplicitLinOp(mat)
    >>> b = np.arange(5).reshape(1,5)
    >>> LeastSquares = QuadraticFunc(Q=A.gram(), c=A.adjoint(b), t=b.dot(b))

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

    @pycrt.enforce_precision(i="vec")
    def __init__(
        self, Q: pyco.SelfAdjointOp, c: pyct.NDArray, t: typ.Optional[pyct.Real] = None, enable_warnings: bool = True
    ):
        r"""

        Parameters
        ----------
        Q: Self-adjoint positive linear operator
            Self-adjoint, positive, nonzero, bounded linear operator
        c: NDArray
            (N,) vector. N-D input arrays are flattened. This is the vectorial representation of the linear term of the
             quadratic functional.
        t: Real
            Scalar term of the quadratic functional.
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.
        """
        super(QuadraticFunc, self).__init__(shape=(1, vec.size))
        self.Q = Q
        self.c = c
        self.t = t
        self.enable_warnings = enable_warnings

        self._lipschitz = np.infty

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self.Q, self.c, self.t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return self.Q(arr).dot(arr) + self.c(arr) + self.t

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self.Q, self.c, self.t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return self.Q(arr) + self.c

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if (
            not all([(elem.dtype == pycrt.getPrecision().value) or (elem is None) for elem in [self.Q, self.c, self.t]])
            and self._enable_warnings
        ):
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return pycs.cg.CG(Q + tau * IdentityOp(Q.shape)).fit(arr - self.c)

    def dif_lipschitz(self, **kwargs) -> float:
        return self.Q.lipschitz()
