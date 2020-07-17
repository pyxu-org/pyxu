import numpy as np
from typing import Union
from numbers import Number
import scipy.optimize as sciop


def sign(x: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
    y = 0 * x
    y[np.abs(x) != 0] = np.conj(x[np.abs(x) != 0]) / np.abs(x[np.abs(x) != 0])
    return y


def soft(x: Union[np.ndarray, Number], tau: Number) -> Union[np.ndarray, Number]:
    return np.clip(np.abs(x) - tau, a_min=0, a_max=None) * sign(x)


def proj_l1_ball(x: Union[np.ndarray, Number], radius: Number) -> Union[np.ndarray, Number]:
    if np.sum(np.abs(x)) <= radius:
        return x
    else:
        mu_max = np.max(np.abs(x))
        func = lambda mu: np.sum(np.clip(np.abs(x) - mu, a_min=0, a_max=None)) - radius
        mu_star = sciop.brentq(func, a=0, b=mu_max)
        return soft(x, mu_star)


def proj_l2_ball(x: Union[np.ndarray, Number], radius: Number) -> Union[np.ndarray, Number]:
    if np.linalg.norm(x) <= radius:
        return x
    else:
        return radius * x / np.linalg.norm(x)


def proj_linfty_ball(x: Union[np.ndarray, Number], radius: Number) -> Union[np.ndarray, Number]:
    y = x
    y[y > radius] = radius
    y[y < -radius] = -radius
    return y


def proj_nonnegative_orthant(x: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
    y = np.real(x)
    y[y < 0] = 0
    return y


def proj_segment(x: Union[np.ndarray, Number], a: Number = 0, b: Number = 1) -> Union[np.ndarray, Number]:
    y = np.real(x)
    y[y < a] = a
    y[y > b] = b
    return y
