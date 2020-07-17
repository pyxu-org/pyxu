from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, LpNorm, IndicatorFunctional
from pycsou.core.linop import LinearOperator, DenseLinearOperator
from pycsou.util.math import soft, proj_l1_ball, proj_l2_ball, proj_linfty_ball, proj_nonnegative_orthant, proj_segment
from typing import Union, Optional, Iterable, Any
from numbers import Number
import numpy as np
import scipy.optimize as sciop


class L2Norm(LpNorm):
    r"""
    :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N |x_i|^2}`.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L2Norm

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = L2Norm(dim=x.size)
       >>> norm(x)
       16.881943016134134
       >>> tau=1.2; np.allclose(norm.prox(x, tau=tau),np.clip(1 - tau / norm(x), a_min=0, a_max=None) * x)
       True
       >>> lambda_=3; scaled_norm = lambda_ * norm; scaled_norm(x)
       50.645829048402405
       >>> np.allclose(scaled_norm.prox(x, tau=tau),np.clip(1 - tau * lambda_ / norm(x), a_min=0, a_max=None) * x)
       True

    See Also
    --------
    :py:func:`~pycsou.func.loss.L2Loss`, `~pycsou.core.functional.LpNorm`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(L2Norm, self).__init__(dim=dim, proj_lq_ball=proj_l2_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.linalg.norm(x)


class SquaredL2Norm(DifferentiableFunctional):
    r"""
    Squared :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert^2_2:=\sum_{i=1}^N |x_i|^2`.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.core.linop import DenseLinearOperator

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = SquaredL2Norm(dim=x.size)
       >>> norm(x)
       285.00000000000006
       >>> np.allclose(norm.gradient(x), 2 * x)
       True
       >>> lambda_=3; scaled_norm = lambda_ * norm
       >>> scaled_norm(x)
       855.0000000000002
       >>> np.allclose(scaled_norm.gradient(x), 2 * lambda_ *  x)
       True
       >>> Gmat = np.arange(100).reshape(10, 10)
       >>> G = DenseLinearOperator(Gmat, is_symmetric=False)
       >>> weighted_norm = norm * G
       >>> np.allclose(weighted_norm.gradient(x), 2 * Gmat.transpose() @ (Gmat @ x))
       True

    See Also
    --------
    :py:func:`~pycsou.func.loss.SquaredL2Loss`, :py:class:`~pycsou.core.functional.DifferentiableFunctional`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(SquaredL2Norm, self).__init__(dim=dim, data=None, is_linear=False, lipschitz_cst=np.infty,
                                            diff_lipschitz_cst=2)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.linalg.norm(x) ** 2

    def jacobianT(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        r"""Gradient of the squared L2 norm at x."""
        return 2 * x


def L2Ball(dim: int, radius: Number) -> IndicatorFunctional:
    r"""
    Constructs the indicator function of an :math:`\ell_2`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \,\Vert\mathbf{x}\Vert_2\leq \text{radius}\}`
    with prescribed radius.

    Parameters
    ----------
    dim : int
        Dimension of the domain.
    radius: Number
        Radius of the :math:`\ell_2`-ball.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the :math:`\ell_2`-ball.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L2Ball
       from pycsou.util.math import proj_l2_ball

    .. doctest::

       >>> x1 = np.arange(10); x2 = x1 / np.linalg.norm(x1)
       >>> radius=10; ball = L2Ball(dim=x1.size, radius=radius)
       >>> ball(x1), ball(x2)
       (inf, 0)
       >>> np.allclose(ball.prox(x1,tau=1), proj_l2_ball(x1, radius=radius)), np.linalg.norm(ball.prox(x1,tau=1))
       (True, 10.0)
       >>> np.allclose(ball.prox(x2,tau=1), x2)
       True

    See Also
    --------
    :py:func:`~pycsou.func.loss.L2BallLoss`, py:class:`pycsou.core.functional.IndicatorFunctional`.
    """
    condition_func = lambda x: np.linalg.norm(x) <= radius
    projection_func = lambda x: proj_l2_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class L1Norm(LpNorm):
    r"""
    Class for the :math:`\ell_1`-norm, defined as :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|`.

    This class inherits from the base class :py:class:`~pycsou.core.functional.LpNorm`.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L1Norm
       from pycsou.util.math import soft
       
    .. doctest::

       >>> x = np.arange(10)
       >>> norm = L1Norm(dim=x.size)
       >>> norm(x)
       45
       >>> tau=1.2; np.allclose(norm.prox(x, tau=tau),soft(x,tau=tau))
       True
       >>> lambda_=3; scaled_norm = lambda_ * norm; scaled_norm(x)
       135
       >>> np.allclose(scaled_norm.prox(x, tau=tau),soft(x,tau=tau * lambda_))
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:func:`~pycsou.func.loss.L1Loss`, :py:func:`~pycsou.func.loss.L1BallLoss`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(L1Norm, self).__init__(dim=dim, proj_lq_ball=proj_linfty_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x))

    def soft(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        """Soft thresholding operator (see :py:func:`~pycsou.util.math.soft`)."""
        return soft(x=x, tau=tau)


class SquaredL1Norm(ProximableFunctional):
    def __init__(self, dim: int, prox_computation='sort'):
        self.prox_computation = prox_computation
        super(SquaredL1Norm, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x)) ** 2

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        if self.prox_computation == 'root':
            if np.linalg.norm(x) > 0:
                mu_max = np.max(np.abs(x) ** 2) / (4 * tau)
                mu_min = 1e-12
                func = lambda mu: np.sum(np.clip(np.abs(x) * np.sqrt(tau / mu) - 2 * tau, a_min=0, a_max=None)) - 1
                mu_star = sciop.brentq(func, a=mu_min, b=mu_max)
                lambda_ = np.clip(np.abs(x) * np.sqrt(tau / mu_star) - 2 * tau, a_min=0, a_max=None)
                return lambda_ * x / (lambda_ + 2 * tau)
            else:
                return x
        elif self.prox_computation == 'sort':
            y = np.sort(np.abs(x))[::-1]
            cumsum_y = np.cumsum(y)
            test_array = y - (2 * tau / (1 + (np.arange(y.size) + 1) * 2 * tau)) * cumsum_y
            max_nzi = np.max(np.nonzero(test_array > 0))
            threshold = (2 * tau / (1 + (max_nzi + 1) * 2 * tau)) * cumsum_y[max_nzi]
            return soft(x, threshold)


def L1Ball(dim: int, radius: Number) -> IndicatorFunctional:
    condition_func = lambda x: np.sum(np.abs(x)) <= radius
    projection_func = lambda x: proj_l1_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class LInftyNorm(LpNorm):
    def __init__(self, dim: int):
        super(LInftyNorm, self).__init__(dim=dim, proj_lq_ball=proj_l1_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.max(np.abs(x))


def LInftyBall(dim: int, radius: Number) -> IndicatorFunctional:
    condition_func = lambda x: np.max(np.abs(x)) <= radius
    projection_func = lambda x: proj_linfty_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def NonNegativeOrthant(dim: int) -> IndicatorFunctional:
    condition_func = lambda x: np.alltrue(x >= 0)
    projection_func = lambda x: proj_nonnegative_orthant(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def Segment(dim: int, a: Number = 0, b: Number = 1):
    condition_func = lambda x: np.alltrue((x >= a) & (x <= b))
    projection_func = lambda x: proj_segment(x, a=a, b=b)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def RealLine(dim: int):
    condition_func = lambda x: np.alltrue(np.isreal(x))
    projection_func = lambda x: np.real(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def ImagLine(dim: int):
    condition_func = lambda x: np.alltrue(np.real(x) == 0)
    projection_func = lambda x: np.imag(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class LogBarrier(ProximableFunctional):
    def __init__(self, dim: int):
        super(LogBarrier, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return -np.sum(np.log(x))

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return (x + np.sqrt(x ** 2 + 4 * tau)) / 2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
