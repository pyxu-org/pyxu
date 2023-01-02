import functools
import warnings

import dask
import numpy as np
import scipy.optimize as sopt

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "L1Norm",
    "L2Norm",
    "SquaredL2Norm",
    "SquaredL1Norm",
    "LInfinityNorm",
]


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

    def __init__(self, dim: pyct.Integer = None, prox_algo: str = "sort"):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        prox_algo: str
            Algorithm used for computing the proximal operator:

            * 'root' uses [FirstOrd]_ Lemma 6.70
            * 'sort' uses [OnKerLearn]_ Algorithm 2 (faster).

        Notes
        -----
        :py:meth:`~pycsou.operator.func.norm.SquaredL1Norm.prox` will always use the root method
        when applied on Dask inputs. (Reason: sorting Dask inputs at scale is discouraged.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf

        algo = prox_algo.strip().lower()
        assert algo in ("root", "sort")
        self._algo = algo

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        y **= 2
        return y

    def _prox_root(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        *sh, dim = arr.shape
        arr = arr.reshape(-1, dim)  # (N_stack, N)
        N_stack, N = arr.shape

        # Restrict processing to non-zero inputs
        xp = pycu.get_array_module(arr)
        idx = xp.arange(N_stack)[xp.linalg.norm(arr, axis=-1) > 0]

        def f(i: pyct.Integer, mu: pyct.Real) -> pyct.Real:
            # Proxy function to compute \mu_opt
            #
            # Inplace implementation of
            #     f(i, mu) = clip(|arr[i]| * sqrt(tau / mu) - 2 * tau, 0, None).sum() - 1
            x = xp.fabs(arr[i])
            x *= xp.sqrt(tau / mu)
            x -= 2 * tau
            xp.clip(x, 0, None, out=x)
            y = x.sum()
            y -= 1

            return y

        # Part 1 below is not vectorized, hence the loop.
        out = xp.zeros((N_stack, N), dtype=arr.dtype)
        for i in idx:
            # Part 1: Compute \mu_opt -----------------------------------------
            mu_min = 1e-12
            mu_max = xp.fabs(arr[i]).max() ** 2 / (4 * tau)  # todo: where does this come from?
            mu_opt, res = sopt.brentq(
                f=functools.partial(f, i),
                a=mu_min,
                b=mu_max,
                full_output=True,
                disp=False,
            )
            if not res.converged:
                msg = "Computing mu_opt did not converge."
                raise ValueError(msg)

            # Part 2: Compute \lambda -----------------------------------------
            # Inplace implementation of
            #     lambda_ = clip(|arr[i]| * sqrt(tau / mu_opt) - 2 * tau, 0, None)
            lambda_ = xp.fabs(arr[i])
            lambda_ *= xp.sqrt(tau / mu_opt)
            lambda_ -= 2 * tau
            xp.clip(lambda_, 0, None, out=lambda_)

            # Part 3: Compute \prox -------------------------------------------
            # Inplace implementation of
            #     out[i] = arr[i] * lambda_ / (lambda_ + 2 * tau)
            out[i] = arr[i]
            out[i] *= lambda_
            lambda_ += 2 * tau
            out[i] /= lambda_

        out = out.reshape(*sh, dim)
        return out

    def _prox_sort(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        *sh, dim = arr.shape
        arr = arr.reshape(-1, dim)  # (N_stack, N)
        N_stack, N = arr.shape

        xp = pycu.get_array_module(arr)

        # Inplace implementation of
        #     y = sort(|arr|, axis=-1)[..., ::-1]
        y = xp.fabs(arr)
        y.sort(axis=-1)
        y = y[..., ::-1]

        out = xp.zeros((N_stack, N), dtype=arr.dtype)
        z = y.cumsum(axis=-1)
        z *= tau / (0.5 + tau * xp.arange(1, N + 1, dtype=z.dtype))
        for i in range(N_stack):
            p = max(xp.flatnonzero(y[i] > z[i]), default=0)
            tau2 = z[i, p]
            out[i] = L1Norm().prox(arr[i], tau2)

        out = out.reshape(*sh, N)
        return out

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        f = dict(
            root=self._prox_root,
            sort=self._prox_sort,
        ).get(self._algo)

        N = pycd.NDArrayInfo
        is_dask = N.from_obj(arr) == N.DASK
        if is_dask:
            f = dask.delayed(f, pure=True)
            if self._algo == "sort":
                msg = "\n".join(
                    [
                        "Using prox_algo='sort' on Dask inputs is inefficient.",
                        "Consider using prox_algo='root' instead.",
                    ]
                )
                warnings.warn(msg, pycuw.PerformanceWarning)

        out = f(arr, tau)
        if is_dask:
            xp = N.DASK.module()
            out = xp.from_delayed(
                out,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        return out


class LInfinityNorm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_{\infty}`-norm, :math:`\Vert\mathbf{x}\Vert_\infty:=\max_{i=1,\ldots,N} |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.max(xp.fabs(arr), axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        mu_max = self.apply(arr)
        if mu_max == 0:
            return arr
        else:
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) - mu, 0)) - tau
            mu_star = sopt.brentq(func, a=0, b=mu_max)
            y = xp.fmin(xp.fabs(arr), mu_star)
            y *= xp.sign(arr)
            return y
