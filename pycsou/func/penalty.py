from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, LpNorm, IndicatorFunctional
from pycsou.core.linop import LinearOperator, IdentityOperator
from pycsou.util.math import soft, proj_l1_ball, proj_l2_ball, proj_linfty_ball, proj_nonnegative_orthant, proj_segment
from typing import Union, Optional, Iterable, Any
from numbers import Number
import numpy as np
import scipy.optimize as sciop


class L2Norm(DifferentiableFunctional):
    r"""
    Class for the :math:`\ell_2`-norm, defined as :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N x^2_i}`.

    This class inherits from the base class :py:class:`~pycsou.core.functional.DifferentiableFunctional`.

    Examples
    --------
    .. testsetup::
       import numpy as np
       from pycsou.func.penalty import L2Norm

    .. doctest::
       >>> x=np.arange(10)
       >>> l2_norm=L2Norm(dim=x.size)
       >>> l2_norm(x)
       16.881943016134134
       >>> np.allclose(l2_norm.gradient(x)(x),2*x)
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L2Ball`, :py:func:`~pycsou.func.loss.L2Loss`, :py:func:`~pycsou.func.loss.L2BallLoss`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(L2Norm, self).__init__(dim=dim, data=None, is_linear=False, lipschitz_cst=np.infty, diff_lipschitz_cst=2)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.linalg.norm(x)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> LinearOperator:
        return 2 * IdentityOperator(size=self.dim)


def L2Ball(dim: int, radius: Number) -> IndicatorFunctional:
    r"""
    Constructs the indicator function of an :math:`\ell_2`-ball with prescribed radius.

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

    .. doctest::
       >>> x=np.arange(10)
       >>> l2_ball=L2Ball(dim=x.size,radius=10)
       >>> l2_ball(x)
       inf
       >>> np.linalg.norm(l2_ball.prox(x,tau=1))
       10.0

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L2Norm`, :py:func:`~pycsou.func.loss.L2Loss`, :py:func:`~pycsou.func.loss.L2BallLoss`.
    """
    condition_func = lambda x: np.linalg.norm(x) <= radius
    projection_func = lambda x: proj_l2_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class L1Norm(LpNorm):
    r"""
    Class for the :math:`\ell_1`-norm, defined as :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|}`.

    This class inherits from the base class :py:class:`~pycsou.core.functional.LpNorm`.

    Examples
    --------
    .. testsetup::
       import numpy as np
       from pycsou.func.penalty import L1Norm
       from pycsou.util.math import soft

    .. doctest::
       >>> x=np.arange(10)
       >>> l1_norm=L1Norm(dim=x.size)
       >>> l1_norm(x)
       45
       >>> np.allclose(l1_norm.prox(x,tau=1),soft(x,tau=1))
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:func:`~pycsou.func.loss.L1Loss`, :py:func:`~pycsou.func.loss.L1BallLoss`.
    """

    def __init__(self, dim: int):
        super(L1Norm, self).__init__(dim=dim, proj_lq_ball=proj_linfty_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x))

    def soft(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return soft(x=x, tau=tau)


class SquaredL1Norm(ProximableFunctional):
    def __init__(self, dim: int, prox_computation='root'):
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
