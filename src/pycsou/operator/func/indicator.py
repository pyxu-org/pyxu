import numpy as np

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.operator.func.norm as pycofn
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "L1Ball",
    "L2Ball",
    "LInfinityBall",
]


class L1Ball(pycofn.ShiftLossMixin, pyca.ProxFunc):
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
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)
        out = xp.zeros((*arr.shape[:-1], self.codim), dtype=arr.dtype)
        norm = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        out[norm > self._radius] = np.inf
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        out = arr.copy()
        out -= pycofn.LInfinityNorm().prox(arr, tau=self._radius)
        return out


class L2Ball(pycofn.ShiftLossMixin, pyca.ProxFunc):
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
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.zeros((*arr.shape[:-1], self.codim), dtype=arr.dtype)
        norm = pylinalg.norm(arr, ord=2, axis=-1, keepdims=True)
        out[norm > self._radius] = np.inf
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        out = arr.copy()
        out -= pycofn.L2Norm().prox(arr, tau=self._radius)
        return out


class LInfinityBall(pycofn.ShiftLossMixin, pyca.ProxFunc):
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
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.zeros((*arr.shape[:-1], self.codim), dtype=arr.dtype)
        norm = pylinalg.norm(arr, ord=np.inf, axis=-1, keepdims=True)
        out[norm > self._radius] = np.inf
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr.clip(-self._radius, self._radius)
        return y
