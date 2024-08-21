import numpy as np
import scipy.optimize as sopt

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
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


class L1Norm(pxa.ProxFunc):
    r"""
    :math:`\ell_{1}`-norm, :math:`\Vert\mathbf{x}\Vert_{1} := \sum_{i} |x_{i}|`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.sqrt(self.dim_size)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        axis = tuple(range(-self.dim_rank, 0))
        y = xp.fabs(arr).sum(axis=axis)[..., np.newaxis]
        return y

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class L2Norm(pxa.ProxFunc):
    r"""
    :math:`\ell_{2}`-norm, :math:`\Vert\mathbf{x}\Vert_{2} := \sqrt{\sum_{i} |x_{i}|^{2}}`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = 1
        self.diff_lipschitz = np.inf

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        axis = tuple(range(-self.dim_rank, 0))
        y = xp.sqrt((arr**2).sum(axis=axis))[..., np.newaxis]
        return y

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        scale = 1 - tau / xp.fmax(self.apply(arr), tau)  # (..., 1)

        y = arr.copy()
        expand = (np.newaxis,) * (self.dim_rank - 1)
        y *= scale[..., *expand]
        return y


class SquaredL2Norm(pxa.QuadraticFunc):
    r"""
    :math:`\ell^{2}_{2}`-norm, :math:`\Vert\mathbf{x}\Vert^{2}_{2} := \sum_{i} |x_{i}|^{2}`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = 2

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        axis = tuple(range(-self.dim_rank, 0))
        y = (arr**2).sum(axis=axis)[..., np.newaxis]
        return y

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        return 2 * arr

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y

    def _quad_spec(self):
        from pyxu.operator import HomothetyOp, NullFunc

        Q = HomothetyOp(dim_shape=self.dim_shape, cst=2)
        c = NullFunc(dim_shape=self.dim_shape)
        t = 0
        return (Q, c, t)


class SquaredL1Norm(pxa.ProxFunc):
    r"""
    :math:`\ell^{2}_{1}`-norm, :math:`\Vert\mathbf{x}\Vert^{2}_{1} := (\sum_{i} |x_{i}|)^{2}`.

    Note
    ----
    * Computing :py:meth:`~pyxu.abc.ProxFunc.prox` is unavailable with DASK inputs.
      (Inefficient exact solution at scale.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.inf

        # prox(): vectorize
        vectorize = pxu.vectorize(
            i="arr",
            dim_shape=self.dim_shape,
            codim_shape=self.dim_shape,
        )
        self.prox = vectorize(self.prox)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = L1Norm(dim_shape=self.dim_shape).apply(arr)
        y **= 2
        return y

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        ndi = pxd.NDArrayInfo.from_obj(arr)
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError("Not implemented at scale.")

        norm = self.apply(arr).item()
        if norm > 0:
            xp = ndi.module()

            # Part 1: Compute \mu_opt -----------------------------------------
            mu_opt, res = sopt.brentq(
                f=lambda mu: (xp.fabs(arr) * xp.sqrt(tau / mu) - 2 * tau).clip(0, None).sum() - 1,
                a=1e-12,
                b=(xp.fabs(arr).max() ** 2) / (4 * tau),
                full_output=True,
                disp=False,
            )
            if not res.converged:
                raise ValueError("Computing mu_opt did not converge.")

            # Part 2: Compute \lambda -----------------------------------------
            lambda_ = (xp.fabs(arr) * xp.sqrt(tau / mu_opt) - 2 * tau).clip(0, None)

            # Part 3: Compute \prox -------------------------------------------
            y = arr.copy()
            y *= lambda_ / (lambda_ + 2 * tau)
        else:
            y = pxu.read_only(arr)

        return y


class LInfinityNorm(pxa.ProxFunc):
    r"""
    :math:`\ell_{\infty}`-norm, :math:`\Vert\mathbf{x}\Vert_{\infty} := \max_{i} |x_{i}|`.

    Note
    ----
    * Computing :py:meth:`~pyxu.abc.ProxFunc.prox` is unavailable with DASK inputs.
      (Inefficient exact solution at scale.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = 1

        # prox(): vectorize
        vectorize = pxu.vectorize(
            i="arr",
            dim_shape=self.dim_shape,
            codim_shape=self.dim_shape,
        )
        self.prox = vectorize(self.prox)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        axis = tuple(range(-self.dim_rank, 0))
        y = xp.fabs(arr).max(axis=axis)[..., np.newaxis]
        return y

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        ndi = pxd.NDArrayInfo.from_obj(arr)
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError("Not implemented at scale.")

        mu_max = self.apply(arr).item()
        if mu_max > tau:
            xp = ndi.module()
            mu_opt = sopt.brentq(
                f=lambda mu: (xp.fabs(arr) - mu).clip(0, None).sum() - tau,
                a=0,
                b=mu_max,
            )
            y = xp.sign(arr) * xp.fmin(xp.fabs(arr), mu_opt)
        else:
            y = pxu.read_only(arr)

        return y


class L21Norm(pxa.ProxFunc):
    r"""
    Mixed :math:`\ell_{2}-\ell_{1}` norm, :math:`\Vert\mathbf{x}\Vert_{2, 1} := \sum_{i} \sqrt{\sum_{j} x_{i, j}^{2}}`.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        l2_axis: pxt.NDArrayAxis = (0,),
    ):
        r"""
        Parameters
        ----------
        l2_axis: NDArrayAxis
            Axis (or axes) along which the :math:`\ell_{2}` norm is applied.
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        assert self.dim_rank >= 2

        l2_axis = pxu.as_canonical_axes(l2_axis, rank=self.dim_rank)
        l1_axis = tuple(ax for ax in range(self.dim_rank) if ax not in l2_axis)

        self.lipschitz = np.inf
        self._l1_axis = l1_axis
        self._l2_axis = l2_axis

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]

        l2_axis = tuple(len(sh) + ax for ax in self._l2_axis)
        x = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(x, out=x)

        l1_axis = tuple(len(sh) + ax for ax in self._l1_axis)
        out = x.sum(axis=l1_axis, keepdims=True)

        out = out.squeeze(l1_axis + l2_axis)[..., np.newaxis]
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]

        l2_axis = tuple(len(sh) + ax for ax in self._l2_axis)
        n = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(n, out=n)

        out = arr.copy()
        out *= 1 - tau / xp.fmax(n, tau)
        return out


class PositiveL1Norm(pxa.ProxFunc):
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

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        from pyxu.operator.func.indicator import PositiveOrthant

        self._indicator = PositiveOrthant(dim_shape=dim_shape)
        self._l1norm = L1Norm(dim_shape=dim_shape)
        self.lipschitz = np.inf

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self._indicator(arr) + self._l1norm(arr)

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        y = (arr - tau).clip(0, None)
        return y
