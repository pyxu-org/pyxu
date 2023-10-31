import types

import numpy as np
import scipy.integrate as spi

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "Dirac",
    "FSSPulse",
    "Box",
    "Triangle",
    "TruncatedGaussian",
    "KaiserBessel",
]


def _get_module(arr: pxt.NDArray):
    N = pxd.NDArrayInfo
    ndi = N.from_obj(arr)
    if ndi == N.NUMPY:
        xp = N.NUMPY.module()
        sps = pxu.import_module("scipy.special")
    elif ndi == N.CUPY:
        xp = N.CUPY.module()
        sps = pxu.import_module("cupyx.scipy.special")
    else:
        raise ValueError(f"Unsupported array type {ndi}.")
    return xp, sps


class FSSPulse(pxa.Map):
    r"""
    1D Finite-Support Symmetric function :math:`f: \mathbb{R} \to \mathbb{R}`, element-wise.
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))

    def support(self) -> pxt.Real:
        r"""
        Returns
        -------
        s: Real
            Value :math:`s > 0` such that :math:`f(x) = 0, \; \forall |x| > s`.
        """
        raise NotImplementedError

    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate :math:`f^{\mathcal{F}}(v)`.

        :py:meth:`~pyxu.operator.FSSPulse.applyF` has the same semantics as :py:meth:`~pyxu.abc.Map.apply`.

        The Fourier convention used is

        .. math::

           \mathcal{F}(f)(v) = \int f(x) e^{-j 2\pi v x} dx
        """
        raise NotImplementedError

    def supportF(self, eps: pxt.Real) -> pxt.Real:
        r"""
        Parameters
        ----------
        eps: Real
            Energy cutoff threshold :math:`\epsilon \in [0, 0.05]`.

        Returns
        -------
        sF: Real
            Value such that

            .. math::

               \int_{-s^{\mathcal{F}}}^{s^{\mathcal{F}}} |f^{\mathcal{F}}(v)|^{2} dv
               \approx
               (1 - \epsilon) \|f\|_{2}^{2}
        """
        eps = float(eps)
        assert 0 <= eps <= 0.05
        tol = 1 - eps

        def energy(f: callable, a: float, b: float) -> float:
            # Estimate \int_{a}^{b} f^{2}(x) dx
            E, _ = spi.quadrature(lambda _: f(_) ** 2, a, b, maxiter=200)
            return E

        if np.isclose(eps, 0):
            sF = np.inf
        else:
            s = self.support()
            E_tot = energy(self.apply, -s, s)

            # Coarse-grain search for a max bandwidth in v_step increments.
            tolerance_reached = False
            v_step = 1 / s  # slowest decay signal is sinc() -> steps at its zeros.
            v_max = 0
            while not tolerance_reached:
                v_max += v_step
                E = energy(self.applyF, -v_max, v_max)
                tolerance_reached = E >= tol * E_tot

            # Fine-grained search for a max bandwidth in [v_max - v_step, v_max] region.
            v_fine = np.linspace(v_max - v_step, v_max, 100)
            E = np.array([energy(self.applyF, -v, v) for v in v_fine])

            sF = v_fine[E >= tol * E_tot].min()
        return sF

    def argscale(self, scalar: pxt.Real) -> pxt.OpT:
        scalar = float(scalar)
        assert scalar > 0

        @pxrt.enforce_precision(i="arr")
        def g_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            # :math:`g(x) = f(\alpha x)`
            op, cst = _._op, _._cst
            return op.apply(cst * arr)

        def g_support(_) -> pxt.Real:
            op, cst = _._op, _._cst
            return op.support() / cst

        @pxrt.enforce_precision(i="arr")
        def g_applyF(_, arr: pxt.NDArray) -> pxt.NDArray:
            # :math:`g^{F}(v) = f^{F}(v / \alpha) / \alpha`
            op, cst = _._op, _._cst
            return op.applyF(arr / cst) / cst

        def g_supportF(_, eps: pxt.Real) -> pxt.Real:
            op, cst = _._op, _._cst
            return op.supportF(eps) * cst

        def g_expr(_) -> tuple:
            return ("argscale", _._op, _._cst)

        g = FSSPulse(dim=self.dim)
        g._cst = scalar  # scale factor
        g._op = self  # initial pulse

        g.apply = types.MethodType(g_apply, g)
        g.support = types.MethodType(g_support, g)
        g.applyF = types.MethodType(g_applyF, g)
        g.supportF = types.MethodType(g_supportF, g)
        g._expr = types.MethodType(g_expr, g)
        return g


class Dirac(FSSPulse):
    r"""
    Dirac-delta function.

    Notes
    -----
    * :math:`f(x) = \delta(x)`
    * :math:`f^{\mathcal{F}}(v) = 1`
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(dim=dim)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.zeros_like(arr)
        y[xp.isclose(arr, 0)] = 1
        return y

    def support(self) -> pxt.Real:
        return 1e-6  # small value approximating 0

    @pxrt.enforce_precision(i="arr")
    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        return np.ones_like(arr)

    def supportF(self, eps: pxt.Real) -> pxt.Real:
        return np.inf


class Box(FSSPulse):
    r"""
    Box function.

    Notes
    -----
    * :math:`f(x) = 1_{[-1, 1]}(x)`
    * :math:`f^{\mathcal{F}}(v) = 2 \; \text{sinc}(2 v)`
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(dim=dim)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.zeros_like(arr)
        y[xp.fabs(arr) <= 1] = 1
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = 2 * xp.sinc(2 * arr)
        return y


class Triangle(FSSPulse):
    r"""
    Triangle function.

    Notes
    -----
    * :math:`f(x) = (1 - |x|) 1_{[-1, 1]}(x)`
    * :math:`f^{\mathcal{F}}(v) = \text{sinc}^{2}(v)`
    """

    def __init__(self, dim: pxt.Integer):
        super().__init__(dim=dim)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.clip(1 - xp.fabs(arr), 0, None)
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.sinc(arr)
        y **= 2
        return y


class TruncatedGaussian(FSSPulse):
    r"""
    Truncated Gaussian.

    Notes
    -----
    * :math:`f(x) = \exp\left[-\frac{1}{2} \left(\frac{x}{\sigma}\right)^{2}\right]
      1_{[-1, 1]}(x)`
    * :math:`f^{\mathcal{F}}(v) =
      \sqrt{2 \pi} \sigma \exp\left[-2 (\pi \sigma v)^{2} \right]
      \Re\left\{
      \text{erf}\left(
      \frac{1}{\sqrt{2} \sigma} +
      j \sqrt{2} \pi \sigma v
      \right)
      \right\}`
    """

    def __init__(self, dim: pxt.Integer, sigma: pxt.Real):
        super().__init__(dim=dim)
        self._sigma = float(sigma)
        assert 0 < self._sigma <= 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        w = arr / (np.sqrt(2) * self._sigma)
        out = xp.exp(-(w**2))
        out[xp.fabs(arr) > 1] = 0
        return out

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, sps = _get_module(arr)
        w = np.sqrt(2) * np.pi * self._sigma * arr
        a = np.sqrt(2) * self._sigma
        b = xp.exp(-(w**2))
        c = sps.erf((1 / a) + 1j * w).real
        out = np.sqrt(np.pi) * a * b * c
        return out


class KaiserBessel(FSSPulse):
    r"""
    Kaiser-Bessel pulse.

    Notes
    -----
    * :math:`f(x) = \frac{I_{0}(\beta \sqrt{1 - x^{2}})}{I_{0}(\beta)}
      1_{[-1, 1]}(x)`
    * :math:`f^{\mathcal{F}}(v) =
      \frac{2}{I_{0}(\beta)}
      \frac
      {\sinh\left[\sqrt{\beta^{2} - (2 \pi v)^{2}} \right]}
      {\sqrt{\beta^{2} - (2 \pi v)^{2}}}`
    """

    def __init__(self, dim: pxt.Integer, beta: pxt.Real):
        super().__init__(dim=dim)
        self._beta = float(beta)
        assert self._beta >= 0

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, sps = _get_module(arr)
        a = xp.zeros_like(arr)
        mask = xp.fabs(arr) <= 1
        a[mask] = sps.i0(self._beta * xp.sqrt(1 - (arr[mask] ** 2)))
        y = a / sps.i0(self._beta)
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    def applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        if np.isclose(self._beta, 0):
            y = Box().applyF(arr)
        else:
            xp, sps = _get_module(arr)

            a = self._beta**2 - (2 * np.pi * arr) ** 2
            mask = a > 0
            a = xp.sqrt(xp.fabs(a))

            y = xp.zeros_like(arr)
            y[mask] = xp.sinh(a[mask]) / a[mask]
            y[~mask] = xp.sinc(a[~mask] / np.pi)

            y *= 2 / sps.i0(self._beta)
        return y

    def supportF(self, eps: pxt.Real) -> pxt.Real:
        if np.isclose(self._beta, 0):
            sF = Box().supportF(eps)
        elif np.isclose(eps, 0):
            # use cut-off frequency: corresponds roughly to eps=1e-10
            sF = self._beta / (2 * np.pi)
        else:
            sF = super().supportF(eps)
        return sF
