import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sin(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cos(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.tan(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arcsin(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arccos(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arctan(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sinh(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cosh(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.tanh(arr)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        v = v**2
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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arcsinh(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arccosh(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.arctanh(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr.copy()
        if self._base is not None:
            out *= np.log(float(self._base))

        xp = pxu.get_array_module(arr)
        xp.exp(out, out=out)
        return out

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        out = xp.log(arr)
        if self._base is not None:
            out /= np.log(float(self._base))
        return out

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sqrt(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.cbrt(arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr.copy()
        out **= 2
        return out

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        out = arr.copy()
        out **= 2
        out *= -1
        xp.exp(out, out=out)
        return out

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = -arr
        xp.exp(x, out=x)
        x += 1
        return 1 / x

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.logaddexp(0, arr)

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

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        f = Sigmoid(dim_shape=self.dim_shape)
        out = f.apply(arr)
        out *= arr
        return out

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        f = Sigmoid(dim_shape=self.dim_shape)
        xp = pxu.get_array_module(arr)
        a = xp.exp(-arr)
        a *= 1 + arr
        a += 1
        b = f.apply(arr)
        b **= 2
        return pxlb.DiagonalOp(a * b)
