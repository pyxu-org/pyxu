r"""
Universal functions.

Ufuncs have class-oriented and function-oriented interfaces.
Only the functional interface is exposed due to its common use.

Example
-------
.. code-block:: python3

   from pyxu.abc import LinOp
   from pyxu.operator import sin
   from pyxu.operator.map.ufunc import _Sin

   N = 10
   x = np.random.randn(N)
   A = LinOp.from_array(np.random.randn(N, N))

   op1 = _Sin(x.size) * A  # class interface
   op2 = sin(A)           # function interface

   np.allclose(op1.apply(x), np.sin(A.apply(x)))  # True
   np.allclose(op2.apply(x), np.sin(A.apply(x)))  # True
"""

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "log",
    "clip",
    "sqrt",
    "cbrt",
    "square",
    "abs",
    "sign",
    "gaussian",
    "sigmoid",
    "softplus",
    "leakyrelu",
    "relu",
    "silu",
    "softmax",
]


# Trigonometric Functions =====================================================
class _Sin(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def sin(op: pxt.OpT) -> pxt.OpT:
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
    return _Sin(op.codim) * op


class _Cos(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def cos(op: pxt.OpT) -> pxt.OpT:
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
    return _Cos(op.codim) * op


class _Tan(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def tan(op: pxt.OpT) -> pxt.OpT:
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
    return _Tan(op.codim) * op


class _ArcSin(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def arcsin(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcSin(op.codim) * op


class _ArcCos(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))

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


def arccos(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcCos(op.codim) * op


class _ArcTan(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def arctan(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcTan(op.codim) * op


# Hyperbolic Functions ========================================================
class _Sinh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def sinh(op: pxt.OpT) -> pxt.OpT:
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
    return _Sinh(op.codim) * op


class _Cosh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def cosh(op: pxt.OpT) -> pxt.OpT:
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
    return _Cosh(op.codim) * op


class _Tanh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def tanh(op: pxt.OpT) -> pxt.OpT:
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
    return _Tanh(op.codim) * op


class _ArcSinh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def arcsinh(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcSinh(op.codim) * op


class _ArcCosh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def arccosh(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcCosh(op.codim) * op


class _ArcTanh(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def arctanh(op: pxt.OpT) -> pxt.OpT:
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
    return _ArcTanh(op.codim) * op


# Exponential Functions =======================================================
class _Exp(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer, base: pxt.Real = None):
        super().__init__(shape=(dim, dim))
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


def exp(op: pxt.OpT, base: pxt.Real = None) -> pxt.OpT:
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
    return _Exp(op.codim, base) * op


class _Log(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer, base: pxt.Real = None):
        super().__init__(shape=(dim, dim))
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


def log(op: pxt.OpT, base: pxt.Real = None) -> pxt.OpT:
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
    return _Log(op.codim, base) * op


# Miscellaneous ===============================================================
class _Clip(pxa.Map):
    def __init__(self, dim: pxt.Integer, a_min: pxt.Real = None, a_max: pxt.Real = None):
        super().__init__(shape=(dim, dim))
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


def clip(op: pxt.OpT, a_min: pxt.Real = None, a_max: pxt.Real = None):
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
    return _Clip(op.codim, a_min=a_min, a_max=a_max) * op


class _Sqrt(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def sqrt(op: pxt.OpT) -> pxt.OpT:
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
    return _Sqrt(op.codim) * op


class _Cbrt(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def cbrt(op: pxt.OpT) -> pxt.OpT:
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
    return _Cbrt(op.codim) * op


class _Square(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def square(op: pxt.OpT) -> pxt.OpT:
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
    return _Square(op.codim) * op


class _Abs(pxa.Map):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.abs(arr)


def abs(op: pxt.OpT) -> pxt.OpT:
    r"""
    Absolute value, element-wise.

    Notes
    -----
    * :math:`f(x) = \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """
    return _Abs(op.codim) * op


class _Sign(pxa.Map):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.sign(arr)


def sign(op: pxt.OpT) -> pxt.OpT:
    r"""
    Number sign indicator, element-wise.

    Notes
    -----
    * :math:`f(x) = x / \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 2`.
    """
    return _Sign(op.codim) * op


# Activation Functions ========================================================
class _Gaussian(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def gaussian(op: pxt.OpT) -> pxt.OpT:
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
    return _Gaussian(op.codim) * op


class _Sigmoid(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
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


def sigmoid(op: pxt.OpT) -> pxt.OpT:
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
    return _Sigmoid(op.codim) * op


class _SoftPlus(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1
        self.diff_lipschitz = 1 / 4

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.logaddexp(0, arr)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        f = _Sigmoid(dim=self.dim)
        v = f.apply(arr)
        return pxlb.DiagonalOp(v)


def softplus(op: pxt.OpT) -> pxt.OpT:
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
    return _SoftPlus(op.codim) * op


class _LeakyReLU(pxa.Map):
    def __init__(self, dim: pxt.Integer, alpha: pxt.Real):
        super().__init__(shape=(dim, dim))
        self._alpha = float(alpha)
        assert self._alpha >= 0
        self.lipschitz = float(max(alpha, 1))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        return xp.where(arr >= 0, arr, arr * self._alpha)


def leakyrelu(op: pxt.OpT, alpha: pxt.Real) -> pxt.OpT:
    r"""
    Leaky rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = x \left[\mathbb{1}_{\ge 0}(x) + \alpha \mathbb{1}_{< 0}(x)\right], \quad \alpha \ge 0`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \max(1, \alpha)`.
    """
    return _LeakyReLU(dim=op.codim, alpha=alpha) * op


class _ReLU(_LeakyReLU):
    def __init__(self, dim: pxt.Integer):
        super().__init__(dim=dim, alpha=0)


def relu(op: pxt.OpT) -> pxt.OpT:
    r"""
    Rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = \lfloor x \rfloor_{+}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """
    return _ReLU(op.codim) * op


class _SiLU(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1.1
        self.diff_lipschitz = 1 / 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        f = _Sigmoid(dim=self.dim)
        out = f.apply(arr)
        out *= arr
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        f = _Sigmoid(dim=self.dim)
        xp = pxu.get_array_module(arr)
        a = xp.exp(-arr)
        a *= 1 + arr
        a += 1
        b = f.apply(arr)
        b **= 2
        return pxlb.DiagonalOp(a * b)


def silu(op: pxt.OpT) -> pxt.OpT:
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
    return _SiLU(op.codim) * op


class _SoftMax(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1
        self.diff_lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = xp.exp(arr)
        out = x / x.sum(axis=-1, keepdims=True)
        return out

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        v = self.apply(arr)
        lhs = pxlb.DiagonalOp(v)
        rhs = pxa.LinFunc.from_array(v).gram()
        op = lhs - rhs
        return op


def softmax(op: pxt.OpT) -> pxt.OpT:
    r"""
    Softmax operator.

    Notes
    -----
    * :math:`[f(x_{1},\ldots,x_{N})]_{i} = e^{x_{i}} / \sum_{k=1}^{N} e^{x_{k}}`
    * :math:`J_{f}(\mathbf{x}) = \text{diag}(f(\mathbf{x})) - f(\mathbf{x}) f(\mathbf{x})^{T}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz constant :math:`\partial L
      = 1`.
    """
    return _SoftMax(op.codim) * op
