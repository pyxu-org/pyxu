import warnings

import numpy as np
import scipy.optimize as sopt

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.math as pxm
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "L1Norm",
    "L2Norm",
    "SquaredL2Norm",
    "SquaredL1Norm",
    "LInfinityNorm",
    "L21Norm",
    "PositiveL1Norm",
]


class _ShiftLossMixin:
    def asloss(self, data: pxt.NDArray = None) -> pxt.OpT:
        from pyxu.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op


class L1Norm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    :math:`\ell_{1}`-norm, :math:`\Vert\mathbf{x}\Vert_{1} := \sum_{i=1}^{N} |x_{i}|`.
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
        self.lipschitz = np.sqrt(dim)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = pxm.norm(arr, ord=1, axis=-1, keepdims=True)
        return y

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class L2Norm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    :math:`\ell_{2}`-norm, :math:`\Vert\mathbf{x}\Vert_{2} := \sqrt{\sum_{i=1}^{N} |x_{i}|^{2}}`.
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
        self.lipschitz = 1
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = pxm.norm(arr, ord=2, axis=-1, keepdims=True)
        return y

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        scale = 1 - tau / xp.fmax(self.apply(arr), tau)

        y = arr.copy()
        y *= scale.astype(dtype=arr.dtype)
        return y


class SquaredL2Norm(pxa.QuadraticFunc):
    r"""
    :math:`\ell^{2}_{2}`-norm, :math:`\Vert\mathbf{x}\Vert^{2}_{2} := \sum_{i=1}^{N} |x_{i}|^{2}`.
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
        self.lipschitz = np.inf
        self.diff_lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = pxm.norm(arr, axis=-1, keepdims=True)
        y **= 2
        return y

    @pxrt.enforce_precision(i="arr")
    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        return 2 * arr

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y

    def _quad_spec(self):
        from pyxu.operator import HomothetyOp, NullFunc

        Q = HomothetyOp(dim=self.dim, cst=2)
        c = NullFunc(dim=self.dim)
        t = 0
        return (Q, c, t)


class SquaredL1Norm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    :math:`\ell^{2}_{1}`-norm, :math:`\Vert\mathbf{x}\Vert^{2}_{1} := (\sum_{i=1}^{N} |x_{i}|)^{2}`.
    """

    def __init__(self, dim: pxt.Integer, prox_algo: str = "sort"):
        r"""
        Parameters
        ----------
        dim: Integer
        prox_algo: "root", "sort"
            Algorithm used for computing the proximal operator:

            * 'root' uses [FirstOrd]_ Lemma 6.70,
            * 'sort' uses [OnKerLearn]_ Algorithm 2 (faster).

        Notes
        -----
        Calling :py:meth:`~pyxu.operator.SquaredL1Norm.prox` with DASK inputs when `algo="sort"` is inefficient at
        scale.  Prefer `algo="root"` in this case.
        """
        super().__init__(shape=(1, dim))
        self.lipschitz = np.inf

        algo = prox_algo.strip().lower()
        assert algo in ("root", "sort")
        self._algo = algo

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = pxm.norm(arr, ord=1, axis=-1, keepdims=True)
        y **= 2
        return y

    def _prox_root(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        norm = xp.linalg.norm(arr, axis=-1)
        if not (norm > 0):
            out = arr
        else:
            # Part 1: Compute \mu_opt -----------------------------------------
            def f(mu: pxt.Real) -> pxt.Real:
                # Proxy function to compute \mu_opt
                #
                # Inplace implementation of
                #     f(mu) = clip(|arr| * sqrt(tau / mu) - 2 * tau, 0, None).sum() - 1
                x = xp.fabs(arr)
                x *= xp.sqrt(tau / mu)
                x -= 2 * tau
                xp.clip(x, 0, None, out=x)
                y = x.sum()
                y -= 1
                return y

            mu_opt, res = sopt.brentq(
                f=f,
                a=1e-12,
                b=xp.fabs(arr).max() ** 2 / (4 * tau),
                full_output=True,
                disp=False,
            )
            if not res.converged:
                msg = "Computing mu_opt did not converge."
                raise ValueError(msg)

            # Part 2: Compute \lambda -----------------------------------------
            # Inplace implementation of
            #     lambda_ = clip(|arr| * sqrt(tau / mu_opt) - 2 * tau, 0, None)
            lambda_ = xp.fabs(arr)
            lambda_ *= xp.sqrt(tau / mu_opt)
            lambda_ -= 2 * tau
            xp.clip(lambda_, 0, None, out=lambda_)

            # Part 3: Compute \prox -------------------------------------------
            # Inplace implementation of
            #     out = arr * lambda_ / (lambda_ + 2 * tau)
            out = arr.copy()
            out *= lambda_
            lambda_ += 2 * tau
            out /= lambda_
        return out

    def _prox_sort(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)

        # Inplace implementation of
        #     y = sort(|arr|, axis=-1)[::-1]
        y = xp.fabs(arr)
        y.sort(axis=-1)
        y = y[::-1]

        z = y.cumsum(axis=-1)
        z *= tau / (0.5 + tau * xp.arange(1, arr.size + 1, dtype=z.dtype))
        tau2 = z[max(xp.flatnonzero(y > z), default=0)]

        out = L1Norm(dim=self.dim).prox(arr, tau2)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        N = pxd.NDArrayInfo
        if (N.from_obj(arr) == N.DASK) and (self._algo == "sort"):
            msg = "\n".join(
                [
                    "Using prox_algo='sort' on DASK inputs is inefficient.",
                    "Consider using prox_algo='root' instead.",
                ]
            )
            warnings.warn(msg, pxw.PerformanceWarning)

        vectorize = pxu.vectorize(
            # DASK backend required since prox_[root|sort]() don't accept DASK inputs.
            i="arr",
            method="parallel",
            codim=arr.shape[-1],
        )
        f = dict(
            root=vectorize(self._prox_root),
            sort=vectorize(self._prox_sort),
        )[self._algo]

        out = f(arr, tau)
        return out


class LInfinityNorm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    :math:`\ell_{\infty}`-norm, :math:`\Vert\mathbf{x}\Vert_{\infty} := \max_{i=1,\ldots,N} |x_{i}|`.
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = pxm.norm(arr, ord=np.inf, axis=-1, keepdims=True)
        return y

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        vectorize = pxu.vectorize(
            i="arr",
            method="parallel",
            codim=arr.shape[-1],
        )
        f = vectorize(self._prox)

        out = f(arr, tau)
        return out

    def _prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.zeros(arr.shape, dtype=arr.dtype)

        def f(mu: pxt.Real) -> pxt.Real:
            # Proxy function to compute \mu_opt
            #
            # Inplace implementation of
            #     f(mu) = clip(xp.fabs(arr) - mu, 0, None).sum() - tau
            x = xp.fabs(arr)
            x -= mu
            xp.clip(x, 0, None, out=x)
            y = x.sum()
            y -= tau
            return y

        mu_max = self.apply(arr)
        if (mu_max > 0).all():
            mu_opt = sopt.brentq(f, a=0, b=mu_max)

            # Inplace implementation of
            #     y = sgn(arr) * fmin(|arr|, mu_opt)
            xp.fabs(arr, out=y)
            xp.fmin(y, mu_opt, out=y)
            y *= xp.sign(arr)

        return y


class L21Norm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    Mixed :math:`\ell_{2}-\ell_{1}` norm, :math:`\Vert\mathbf{x}\Vert_{2, 1} := \sum_{i=1}^{N} \sqrt{\sum_{j=1}^{M}
    x_{i, j}^{2}}`.

    Note
    ----
    The input array need not be 2-dimensional: the :math:`\ell_{2}` norm is applied along a predefined subset of
    dimensions, and the :math:`\ell_{1}` norm on the remaining ones.
    """

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        l2_axis: pxt.NDArrayAxis = (0,),
    ):
        r"""
        Parameters
        ----------
        arg_shape: NDArrayShape
            Shape of the input array.
        l2_axis: NDArrayAxis
            Axis (or axes) along which the :math:`\ell_{2}` norm is applied.
        """
        arg_shape = pxu.as_canonical_shape(arg_shape)
        assert all(a > 0 for a in arg_shape)
        N_dim = len(arg_shape)
        assert N_dim >= 2

        l2_axis = np.unique(pxu.as_canonical_shape(l2_axis))  # drop potential duplicates
        assert np.all((-N_dim <= l2_axis) & (l2_axis < N_dim))  # all axes in valid range
        l2_axis = (l2_axis + N_dim) % N_dim  # get rid of negative axes
        l1_axis = np.setdiff1d(np.arange(N_dim), l2_axis)

        super().__init__(shape=(1, np.prod(arg_shape)))
        self.lipschitz = np.inf
        self._arg_shape = arg_shape
        self._l1_axis = l1_axis
        self._l2_axis = l2_axis

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray):
        sh = arr.shape[:-1]
        arr = arr.reshape(sh + self._arg_shape)

        l2_axis = tuple(len(sh) + self._l2_axis)
        x = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(x, out=x)

        l1_axis = tuple(len(sh) + self._l1_axis)
        out = x.sum(axis=l1_axis, keepdims=True)
        return out.reshape(*sh, -1)

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real):
        sh = arr.shape[:-1]
        arr = arr.reshape(sh + self._arg_shape)

        l2_axis = tuple(len(sh) + self._l2_axis)
        n = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(n, out=n)

        out = arr.copy()
        out *= 1 - tau / xp.fmax(n, tau)
        return out.reshape(*sh, -1)


class PositiveL1Norm(_ShiftLossMixin, pxa.ProxFunc):
    r"""
    :math:`\ell_{1}`-norm, with a positivity constraint.

    .. math::

       f(\mathbf{x})
       :=
       \lVert\mathbf{x}\rVert_{1} + \iota_{+}(\mathbf{x}),

    .. math::

       \textbf{prox}_{\tau f}(\mathbf{z})
       :=
       \max(\mathrm{soft}_\tau(\mathbf{z}), \mathbf{0})

    See Also
    --------
    :py:class:`~pyxu.operator.PositiveOrthant`
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(1, dim))
        from pyxu.operator.func.indicator import PositiveOrthant

        self._indicator = PositiveOrthant(dim=dim)
        self._l1norm = L1Norm(dim=dim)
        self.lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self._indicator(arr) + self._l1norm(arr)

    @pxrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.fmax(0, arr - tau)
