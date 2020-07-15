from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, LpNorm, IndicatorFunctional
from pycsou.core.linop import LinearOperator, IdentityOperator
from pycsou.util.math import soft, proj_l1_ball, proj_l2_ball, proj_linfty_ball, proj_nonnegative_orthant, proj_segment
from typing import Union, Optional, Iterable, Any
from numbers import Number
import numpy as np
import scipy.optimize as sciop


class L2Norm(DifferentiableFunctional):
    def __init__(self, dim: int):
        super(L2Norm, self).__init__(dim=dim, data=None, is_linear=False, lipschitz_cst=np.infty, diff_lipschitz_cst=2)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x) ** 2).reshape(-1)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> LinearOperator:
        return 2 * IdentityOperator(size=self.dim)


def L2Ball(dim: int, radius: Number) -> IndicatorFunctional:
    condition_func = lambda x: np.linalg.norm(x) <= radius
    projection_func = lambda x: proj_l2_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class L1Norm(LpNorm):
    def __init__(self, dim: int):
        super(L1Norm, self).__init__(dim=dim, proj_lq_ball=proj_linfty_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x)).reshape(-1)

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
        return np.max(np.abs(x)).reshape(-1)


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
        return -np.sum(np.log(x)).reshape(-1)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return (x + np.sqrt(x ** 2 + 4 * tau)) / 2
