from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, ProxFuncPreComp, IndicatorFunctional
from pycsou.core.linop import LinearOperator, IdentityOperator
from pycsou.func.penalty import L2Norm, L1Norm, LInftyNorm, L2Ball, L1Ball, LInftyBall, LogBarrier
from typing import Union, Optional, Iterable, Any
from numbers import Number
import numpy as np


def ProximableLoss(Loss: ProximableFunctional, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    return ProxFuncPreComp(Loss, scale=1, shift=-data)


def DifferentiableLoss(Loss: DifferentiableFunctional, data: Union[Number, np.ndarray]) -> DifferentiableFunctional:
    return Loss * (IdentityOperator(size=Loss.dim, dtype=data.dtype) - data)


def L2Loss(dim: int, data: Union[Number, np.ndarray]) -> DifferentiableFunctional:
    L2_norm = L2Norm(dim=dim)
    return DifferentiableLoss(L2_norm, data=data)


def L2BallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    L2_ball = L2Ball(dim=dim, radius=radius)
    return ProximableLoss(L2_ball, data=data)


def L1Loss(dim: int, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    L1_norm = L1Norm(dim=dim)
    return ProximableLoss(L1_norm, data=data)


def L1BallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    L1_ball = L1Ball(dim=dim, radius=radius)
    return ProximableLoss(L1_ball, data=data)


def LInftyLoss(dim: int, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    LInfty_norm = LInftyNorm(dim=dim)
    return ProximableLoss(LInfty_norm, data=data)


def LInftyBallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    LInfty_ball = LInftyBall(dim=dim, radius=radius)
    return ProximableLoss(LInfty_ball, data=data)


def ConsistencyLoss(dim: int, data: Union[Number, np.ndarray]):
    condition_func = lambda x: np.allclose(x, data)
    projection_func = lambda x: data
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class KLDivergence(ProximableFunctional):
    def __init__(self, dim: int, data: Union[Number, np.ndarray]):
        super(KLDivergence, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.data = data

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(self.data * np.log(self.data / x) - self.data + x).reshape(-1)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return (x - tau + np.sqrt((x - tau) ** 2 + 4 * tau * self.data)) / 2