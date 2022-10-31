import cupy as cp
import dask.array as ds
import numpy as np
import scipy.optimize as sciop

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.array_module as pycam
import pycsou.util.ptype as pyct

__all__ = ["L1Norm", "L2Norm", "SquaredL2Norm", "SquaredL1Norm", "LInftyNorm", "L1Ball", "L2Ball", "LInftyBall"]


class ShiftLossMixin:
    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        from pycsou.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op


class L1Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_1`-norm, :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)

        Notes
        -----
        The operator's Lipschitz constant is set to :math:`\infty` if domain-agnostic.
        It is recommended to set `dim` explicitly to compute a tight closed-form.
        """
        super().__init__(shape=(1, dim))
        if dim is None:
            self._lipschitz = np.inf
        else:
            self._lipschitz = np.sqrt(dim)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class L2Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N |x_i|^2}`.
    """

    def __init__(self, dim: pyct.Integer = None):
        super().__init__(shape=(1, dim))
        self._lipschitz = 1
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=2, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        scale = 1 - tau / xp.fmax(self.apply(arr), tau)

        y = arr.copy()
        y *= scale.astype(dtype=arr.dtype)
        return y


class SquaredL2Norm(pyca._QuadraticFunc):
    r"""
    :math:`\ell^2_2`-norm, :math:`\Vert\mathbf{x}\Vert^2_2:=\sum_{i=1}^N |x_i|^2`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, axis=-1, keepdims=True)
        y **= 2
        return y

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y

    def _hessian(self) -> pyct.OpT:
        from pycsou.operator.linop import IdentityOp

        if self.dim is None:
            msg = "\n".join(
                [
                    "hessian: domain-agnostic functionals unsupported.",
                    f"Explicitly set `dim` in {self.__class__}.__init__().",
                ]
            )
            raise ValueError(msg)
        return IdentityOp(dim=self.dim).squeeze()


class SquaredL1Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell^2_1`-norm, :math:`\Vert\mathbf{x}\Vert^2_1:=(\sum_{i=1}^N |x_i|)^2`.
    """

    def __init__(self, dim: pyct.Integer = None, prox_computation: str = "sort"):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        prox_computation: str
            Algorithm for computing the proximal operator: 'root' uses [FirstOrd]_ Lemma 6.70, while 'sort' uses [OnKerLearn]_ Algorithm 2 (faster). (Default : 'sort'.)

        Notes
        -----
        If module of input array is dask, algorithm 'root' is used independently of the user choice (sorting is not recommended in dask).
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf
        self.prox_computation = prox_computation

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        y **= 2
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def _prox_root(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if xp.linalg.norm(arr) > 0:
            mu_max = xp.max(xp.fabs(arr) ** 2) / (4 * tau)
            mu_min = 1e-12
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) * xp.sqrt(tau / mu) - 2 * tau, 0)) - 1
            mu_star = sciop.brentq(func, a=mu_min, b=mu_max)
            lambda_ = xp.fmax(xp.abs(arr) * xp.sqrt(tau / mu_star) - 2 * tau, 0)
            return lambda_ * arr / (lambda_ + 2 * tau)
        else:
            return arr

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycam.redirect("arr", DASK=_prox_root)
    def _prox_sort(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        z = xp.sort(xp.abs(arr))[::-1]
        cumsum_z = xp.cumsum(z)
        test_array = z - (2 * tau / (1 + (xp.arange(z.size) + 1) * 2 * tau)) * cumsum_z
        max_nzi = xp.max(xp.nonzero(test_array.reshape(-1) > 0)[0])
        threshold = (2 * tau / (1 + (max_nzi + 1) * 2 * tau)) * cumsum_z[max_nzi]
        y = xp.fmax(0, xp.fabs(arr) - threshold)
        y *= xp.sign(arr)
        return y

    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        arr = arr.ravel()
        if self.prox_computation == "root":
            return self._prox_root(arr, tau)
        elif self.prox_computation == "sort":
            return self._prox_sort(arr, tau)


class LInftyNorm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_{\infty}`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\max_{i=1,.,N} |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)

        Notes
        -----
        The operator's Lipschitz constant is set to :math:`\infty` if domain-agnostic.
        It is recommended to set `dim` explicitly to compute a tight closed-form.
        """
        super().__init__(shape=(1, dim))
        if dim is None:
            self._lipschitz = np.inf
        else:
            self._lipschitz = np.sqrt(dim)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.max(xp.fabs(arr), axis=-1)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        mu_max = self.apply(arr)
        func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) - mu, 0)) - tau
        mu_star = sciop.brentq(func, a=0, b=mu_max)
        y = xp.fmin(xp.fabs(arr), mu_star)
        y *= xp.sign(arr)
        return y


class L1Ball(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_1`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_1\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_1\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Radius of ball. (Default: unit ball.)

        Notes
        -----
        The scale parameter in the proximal operator can be set by the user, but it does not affect the computation. Indeed, the prox only depends on the ball radius.
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.Real:
        xp = pycu.get_array_module(arr)
        condition = pylinalg.norm(arr, ord=1) <= self._radius
        return xp.zeros(1) if condition else np.inf * xp.ones(1)

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if pylinalg.norm(arr, ord=1) <= self._radius:
            return arr
        else:
            mu_max = xp.max(xp.fabs(arr))
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) - mu, 0)) - self._radius
            mu_star = sciop.brentq(func, a=0, b=mu_max)
            y = xp.fmax(0, xp.fabs(arr) - mu_star)
            y *= xp.sign(arr)
        return y


class L2Ball(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_2`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_2\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_2\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Radius of ball. (Default: unit ball.)

        Notes
        -----
        The scale parameter in the proximal operator can be set by the user, but it does not affect the computation. Indeed, the prox only depends on the ball radius.
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        condition = xp.linalg.norm(arr) <= self._radius
        return xp.zeros(1) if condition else np.inf * xp.ones(1)

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr_norm = xp.linalg.norm(arr)
        if arr_norm <= self._radius:
            return arr
        else:
            return self._radius * arr / arr_norm


class LInftyBall(ShiftLossMixin, pyca.ProxFunc):
    r"""
    Indicator function of the :math:`\ell_\infty`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_\infty\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_\infty\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}
    """

    def __init__(self, dim: pyct.Integer = None, radius: pyct.Real = 1):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        radius: pyct.Real
            Radius of ball. (Default: unit ball.)

        Notes
        -----
        The scale parameter in the proximal operator can be set by the user, but it does not affect the computation. Indeed, the prox only depends on the ball radius.
        """
        super().__init__(shape=(1, dim))
        self._radius = radius
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        condition = xp.max(xp.fabs(arr)) <= self._radius
        return xp.zeros(1) if condition else np.inf * xp.ones(1)

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
