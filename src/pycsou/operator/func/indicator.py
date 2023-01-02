import numpy as np
import scipy.optimize as sopt

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
from pycsou.operator.func.norm import ShiftLossMixin

__all__ = [
    "L1Ball",
    "L2Ball",
    "LInfinityBall",
]


class L1Ball(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_1`-ball.

    It is defined as:

    .. math::

       \iota_{1}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{1} \le r \\
           \infty & \text{otherwise}.
       \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Ball radius. (Default: unit ball.)
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)
        condition = pylinalg.norm(arr, ord=1) <= self._radius
        return xp.zeros(1, dtype=arr.dtype) if condition else np.inf * xp.ones(1, dtype=arr.dtype)

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if pylinalg.norm(arr, ord=1) <= self._radius:
            return arr
        else:
            mu_max = xp.max(xp.fabs(arr))
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) - mu, 0)) - self._radius
            mu_star = sopt.brentq(func, a=0, b=mu_max)
            y = xp.fmax(0, xp.fabs(arr) - mu_star)
            y *= xp.sign(arr)
        return y


class L2Ball(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_2`-ball.

    It is defined as:

    .. math::

       \iota_{2}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{2} \le r \\
           \infty & \text{otherwise}.
       \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Ball radius. (Default: unit ball.)
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        condition = xp.linalg.norm(arr) <= self._radius
        return xp.zeros(1, dtype=arr.dtype) if condition else np.inf * xp.ones(1, dtype=arr.dtype)

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr_norm = xp.linalg.norm(arr).astype(arr.dtype)
        if arr_norm <= self._radius:
            return arr
        else:
            return self._radius * arr / arr_norm


class LInfinityBall(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_\infty`-ball.

    It is defined as:

    .. math::

       \iota_{\infty}^{r}(\mathbf{x})
       :=
       \begin{cases}
           0 & \|\mathbf{x}\|_{\infty} \le r \\
           \infty & \text{otherwise}.
       \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Ball radius. (Default: unit ball.)
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        condition = xp.max(xp.fabs(arr)) <= self._radius
        return xp.zeros(1, dtype=arr.dtype) if condition else np.inf * xp.ones(1, dtype=arr.dtype)

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if xp.max(xp.fabs(arr)) <= self._radius:
            return arr
        else:
            y = xp.fmin(xp.fabs(arr), self._radius)
            y *= xp.sign(arr)
            return y
