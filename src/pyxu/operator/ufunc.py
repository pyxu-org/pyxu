import types

import numpy as np
import scipy.integrate as spi

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    # Universal functions -----------------------------------------------------
    "Sin",
    "Cos",
    "Tan",
    "ArcSin",
    "ArcCos",
    "ArcTan",
    "Sinh",
    "Cosh",
    "Tanh",
    "ArcSinh",
    "ArcCosh",
    "ArcTanh",
    "Exp",
    "Log",
    "Clip",
    "Sqrt",
    "Cbrt",
    "Square",
    "Abs",
    "Sign",
    "Gaussian",
    "Sigmoid",
    "SoftPlus",
    "LeakyReLU",
    "ReLU",
    "SiLU",
    # Kernels -----------------------------------------------------------------
    "Dirac",
    "FSSPulse",
    "Box",
    "Triangle",
    "TruncatedGaussian",
    "KaiserBessel",
]


# Trigonometric Functions =====================================================
class Sin(pxa.DiffMap):
    r"""
    Trigonometric sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sin(x)`
    * :math:`f'(x) = \cos(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = k \pi, \, k \in \mathbb{Z}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = (2k + 1) \frac{\pi}{2}, \, k
      \in \mathbb{Z}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sin(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        return pxlb.DiagonalOp(xp.cos(arr))


class Cos(pxa.DiffMap):
    r"""
    Trigonometric cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cos(x)`
    * :math:`f'(x) = -\sin(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = (2k + 1) \frac{\pi}{2}, \, k \in
      \mathbb{Z}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = k \pi, \, k \in \mathbb{Z}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cos(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        return pxlb.DiagonalOp(-xp.sin(arr))


class Tan(pxa.DiffMap):
    r"""
    Trigonometric tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tan(x)`
    * :math:`f'(x) = \cos^{-2}(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-\pi, \pi]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-\pi, \pi]`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.tan(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        v = xp.cos(arr)
        v **= 2
        return pxlb.DiagonalOp(1 / v)


class ArcSin(pxa.DiffMap):
    r"""
    Inverse sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \arcsin(x)`
    * :math:`f'(x) = (1 - x^{2})^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arcsin(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        v = arr**2
        v *= -1
        v += 1
        xp.sqrt(v, out=v)
        return pxlb.DiagonalOp(1 / v)


class ArcCos(pxa.DiffMap):
    r"""
    Inverse cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \arccos(x)`
    * :math:`f'(x) = -(1 - x^{2})^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arccos(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        v = arr**2
        v *= -1
        v += 1
        xp.sqrt(v, out=v)
        return pxlb.DiagonalOp(-1 / v)


class ArcTan(pxa.DiffMap):
    r"""
    Inverse tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \arctan(x)`
    * :math:`f'(x) = (1 + x^{2})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 3 \sqrt{3} / 8`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \pm \frac{1}{\sqrt{3}}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 3 * np.sqrt(3) / 8
        #   max_{x \in R} |arctan''(x)|
        # = max_{x \in R} |2x / (1+x^2)^2|
        # = 3 \sqrt(3) / 8  [at x = +- 1/\sqrt(3)]

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arctan(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = arr**2
        v += 1
        return pxlb.DiagonalOp(1 / v)


# Hyperbolic Functions ========================================================
class Sinh(pxa.DiffMap):
    r"""
    Hyperbolic sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sinh(x)`
    * :math:`f'(x) = \cosh(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sinh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        return pxlb.DiagonalOp(xp.cosh(arr))


class Cosh(pxa.DiffMap):
    r"""
    Hyperbolic cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cosh(x)`
    * :math:`f'(x) = \sinh(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cosh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        return pxlb.DiagonalOp(xp.sinh(arr))


class Tanh(pxa.DiffMap):
    r"""
    Hyperbolic tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tanh(x)`
    * :math:`f'(x) = 1 - \tanh^{2}(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 4 / 3 \sqrt{3}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \frac{1}{2} \ln(2 \pm
      \sqrt{3})`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 4 / (3 * np.sqrt(3))
        #   max_{x \in R} |tanh''(x)|
        # = max_{x \in R} |-2 tanh(x) [1 - tanh(x)^2|
        # = 4 / (3 \sqrt(3))  [at x = ln(2 +- \sqrt(3))]

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.tanh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        v**2
        v *= -1
        v += 1
        return pxlb.DiagonalOp(v)


class ArcSinh(pxa.DiffMap):
    r"""
    Inverse hyperbolic sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sinh^{-1}(x) = \ln(x + \sqrt{x^{2} + 1})`
    * :math:`f'(x) = (x^{2} + 1)^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \frac{2}{3 \sqrt{3}}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \pm \frac{1}{\sqrt{2}}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 2 / (3 * np.sqrt(3))
        #   max_{x \in R} |arcsinh''(x)|
        # = max_{x \in R} |-x (x^2 + 1)^{-3/2}|
        # = 2 / (3 \sqrt(3))  [at x = += 1 / \sqrt(2)]

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arcsinh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        v = arr**2
        v += 1
        xp.sqrt(v, out=v)
        return pxlb.DiagonalOp(1 / v)


class ArcCosh(pxa.DiffMap):
    r"""
    Inverse hyperbolic cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cosh^{-1}(x) = \ln(x + \sqrt{x^{2} - 1})`
    * :math:`f'(x) = (x^{2} - 1)^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [1, \infty[`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [1, \infty[`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arccosh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        v = arr**2
        v -= 1
        xp.sqrt(v, out=v)
        return pxlb.DiagonalOp(1 / v)


class ArcTanh(pxa.DiffMap):
    r"""
    Inverse hyperbolic tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tanh^{-1}(x) = \frac{1}{2}\ln\left(\frac{1+x}{1-x}\right)`
    * :math:`f'(x) = (1 - x^{2})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arctanh(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = arr**2
        v *= -1
        v += 1
        return pxlb.DiagonalOp(1 / v)


# Exponential Functions =======================================================
class Exp(pxa.DiffMap):
    r"""
    Exponential, element-wise. (Default: base-E exponential.)

    Notes
    -----
    * :math:`f_{b}(x) = b^{x}`
    * :math:`f_{b}'(x) = b^{x} \ln(b)`
    * :math:`\vert f_{b}(x) - f_{b}(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f_{b}'(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}`.)
    * :math:`\vert f_{b}'(x) - f_{b}'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant
      :math:`\partial L = \infty`.

      (Reason: :math:`f_{b}''(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape, base: pxt.Real = None):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf
        self._base = base

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr.copy()
        if self._base is not None:
            out *= np.log(float(self._base))

        xp = pxu.get_array_module(arr)
        xp.exp(out, out=out)
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        if self._base is not None:
            v *= np.log(float(self._base))
        return pxlb.DiagonalOp(v)


class Log(pxa.DiffMap):
    r"""
    Logarithm, element-wise. (Default: base-E logarithm.)

    Notes
    -----
    * :math:`f_{b}(x) = \log_{b}(x)`
    * :math:`f_{b}'(x) = x^{-1} / \ln(b)`
    * :math:`\vert f_{b}(x) - f_{b}(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f_{b}'(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}_{+}`.)
    * :math:`\vert f_{b}'(x) - f_{b}'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant
      :math:`\partial L = \infty`.

      (Reason: :math:`f_{b}''(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}_{+}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape, base: pxt.Real = None):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf
        self._base = base

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        out = xp.log(arr)
        if self._base is not None:
            out /= np.log(float(self._base))
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = 1 / arr
        if self._base is not None:
            v /= np.log(float(self._base))
        return pxlb.DiagonalOp(v)


# Miscellaneous ===============================================================
class Clip(pxa.Map):
    r"""
    Clip (limit) values in an array, element-wise.

    Notes
    -----
    * .. math::

         f_{[a,b]}(x) =
         \begin{cases}
             a, & \text{if} \ x \leq a, \\
             x, & a < x < b, \\
             b, & \text{if} \ x \geq b.
         \end{cases}
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        a_min: pxt.Real = None,
        a_max: pxt.Real = None,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        if (a_min is None) and (a_max is None):
            raise ValueError("One of Parameter[a_min, a_max] must be specified.")
        self._llim = a_min
        self._ulim = a_max
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        out = arr.copy()
        xp.clip(
            arr,
            self._llim,
            self._ulim,
            out=out,
        )
        return out


class Sqrt(pxa.DiffMap):
    r"""
    Non-negative square-root, element-wise.

    Notes
    -----
    * :math:`f(x) = \sqrt{x}`
    * :math:`f'(x) = 1 / 2 \sqrt{x}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}_{+}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}_{+}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sqrt(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        v *= 2
        return pxlb.DiagonalOp(1 / v)


class Cbrt(pxa.DiffMap):
    r"""
    Cube-root, element-wise.

    Notes
    -----
    * :math:`f(x) = \sqrt[3]{x}`
    * :math:`f'(x) = 1 / 3 \sqrt[3]{x^{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cbrt(arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        v **= 2
        v *= 3
        return pxlb.DiagonalOp(1 / v)


class Square(pxa.DiffMap):
    r"""
    Square, element-wise.

    Notes
    -----
    * :math:`f(x) = x^{2}`
    * :math:`f'(x) = 2 x`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` everywhere.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr.copy()
        out **= 2
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = arr.copy()
        v *= 2
        return pxlb.DiagonalOp(v)


class Abs(pxa.Map):
    r"""
    Absolute value, element-wise.

    Notes
    -----
    * :math:`f(x) = \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.abs(arr)


class Sign(pxa.Map):
    r"""
    Number sign indicator, element-wise.

    Notes
    -----
    * :math:`f(x) = x / \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 2`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sign(arr)


# Activation Functions ========================================================
class Gaussian(pxa.DiffMap):
    r"""
    Gaussian, element-wise.

    Notes
    -----
    * :math:`f(x) = \exp(-x^{2})`
    * :math:`f'(x) = -2 x \exp(-x^{2})`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \sqrt{2 / e}`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = \pm 1 / \sqrt{2}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = np.sqrt(2 / np.e)
        self.diff_lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        out = arr.copy()
        out **= 2
        out *= -1
        xp.exp(out, out=out)
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        v *= -2
        v *= arr
        return pxlb.DiagonalOp(v)


class Sigmoid(pxa.DiffMap):
    r"""
    Sigmoid, element-wise.

    Notes
    -----
    * :math:`f(x) = (1 + e^{-x})^{-1}`
    * :math:`f'(x) = f(x) [ f(x) - 1 ]`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1 / 4`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1 / 6 \sqrt{3}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \ln(2 \pm \sqrt{3})`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1 / 4
        self.diff_lipschitz = 1 / (6 * np.sqrt(3))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = -arr
        xp.exp(x, out=x)
        x += 1
        return 1 / x

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        x = self.apply(arr)
        v = x.copy()
        x -= 1
        v *= x
        return pxlb.DiagonalOp(v)


class SoftPlus(pxa.DiffMap):
    r"""
    Softplus operator.

    Notes
    -----
    * :math:`f(x) = \ln(1 + e^{x})`
    * :math:`f'(x) = (1 + e^{-x})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1 / 4`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 1 / 4

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.logaddexp(0, arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        f = Sigmoid(dim_shape=self.dim_shape)
        v = f.apply(arr)
        return pxlb.DiagonalOp(v)


class LeakyReLU(pxa.Map):
    r"""
    Leaky rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = x \left[\mathbb{1}_{\ge 0}(x) + \alpha \mathbb{1}_{< 0}(x)\right], \quad \alpha \ge 0`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \max(1, \alpha)`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape, alpha: pxt.Real):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self._alpha = float(alpha)
        assert self._alpha >= 0
        self.lipschitz = float(max(alpha, 1))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.where(arr >= 0, arr, arr * self._alpha)


class ReLU(LeakyReLU):
    r"""
    Rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = \lfloor x \rfloor_{+}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(dim_shape=dim_shape, alpha=0)


class SiLU(pxa.DiffMap):
    r"""
    Sigmoid linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = x / (1 + e^{-x})`
    * :math:`f'(x) = (1 + e^{-x} + x e^{-x}) / (1 + e^{-x})^{2}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1.1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x \approx 2.4`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1 / 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1.1
        self.diff_lipschitz = 1 / 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        f = Sigmoid(dim_shape=self.dim_shape)
        out = f.apply(arr)
        out *= arr
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        f = Sigmoid(dim_shape=self.dim_shape)
        xp = pxu.get_array_module(arr)
        a = xp.exp(-arr)
        a *= 1 + arr
        a += 1
        b = f.apply(arr)
        b **= 2
        return pxlb.DiagonalOp(a * b)


# Kernels ---------------------------------------------------------------------
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
    Finite-Support Symmetric function :math:`f: \mathbb{R} \to \mathbb{R}`, element-wise.
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )

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
        # Redefined to propagate (support[F], applyF,)
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

        g = FSSPulse(dim_shape=self.dim_shape)
        g._cst = scalar  # scale factor
        g._op = self  # initial pulse

        g.apply = types.MethodType(g_apply, g)
        g.support = types.MethodType(g_support, g)
        g.applyF = types.MethodType(g_applyF, g)
        g.supportF = types.MethodType(g_supportF, g)
        g._expr = types.MethodType(g_expr, g)
        return g

    def _blockwize_apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # self.apply, applied blockwize on DASK inputs.
        out = arr.map_blocks(
            func=self.apply,
            dtype=arr.dtype,
            meta=arr._meta,
        )
        return out

    def _blockwize_applyF(self, arr: pxt.NDArray) -> pxt.NDArray:
        # self.applyF, applied blockwize on DASK inputs.
        out = arr.map_blocks(
            func=self.applyF,
            dtype=arr.dtype,
            meta=arr._meta,
        )
        return out


class Dirac(FSSPulse):
    r"""
    Dirac-delta function.

    Notes
    -----
    * :math:`f(x) = \delta(x)`
    * :math:`f^{\mathcal{F}}(v) = 1`
    """

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(dim_shape=dim_shape)

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_apply)
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.zeros_like(arr)
        y[xp.isclose(arr, 0)] = 1
        return y

    def support(self) -> pxt.Real:
        return 1e-6  # small value approximating 0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_applyF)
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

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(dim_shape=dim_shape)

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_apply)
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.zeros_like(arr)
        y[xp.fabs(arr) <= 1] = 1
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_applyF)
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

    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(dim_shape=dim_shape)

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_apply)
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        y = xp.clip(1 - xp.fabs(arr), 0, None)
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_applyF)
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

    def __init__(self, dim_shape: pxt.NDArrayShape, sigma: pxt.Real):
        super().__init__(dim_shape=dim_shape)
        self._sigma = float(sigma)
        assert 0 < self._sigma <= 1

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_apply)
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, _ = _get_module(arr)
        w = arr / (np.sqrt(2) * self._sigma)
        out = xp.exp(-(w**2))
        out[xp.fabs(arr) > 1] = 0
        return out

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_applyF)
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

    def __init__(self, dim_shape: pxt.NDArrayShape, beta: pxt.Real):
        super().__init__(dim_shape=dim_shape)
        self._beta = float(beta)
        assert self._beta >= 0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_apply)
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp, sps = _get_module(arr)
        y = xp.zeros_like(arr)
        mask = xp.fabs(arr) <= 1
        y[mask] = sps.i0(self._beta * xp.sqrt(1 - (arr[mask] ** 2)))
        y /= sps.i0(self._beta)
        return y

    def support(self) -> pxt.Real:
        return 1.0

    @pxrt.enforce_precision(i="arr")
    @pxu.redirect(i="arr", DASK=FSSPulse._blockwize_applyF)
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
