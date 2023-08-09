r"""
Universal functions.

Ufuncs have class-oriented and function-oriented interfaces.

Example
-------
.. code-block:: python3

   from pycsou.operator.map import Sin, sin
   from pycsou.abc import LinOp

   N = 10
   x = np.random.randn(N)
   A = LinOp.from_array(np.random.randn(N, N))

   op1 = Sin(x.size) * A  # class interface
   op2 = sin(A)           # function interface

   np.allclose(op1.apply(x), np.sin(A.apply(x)))  # True
   np.allclose(op2.apply(x), np.sin(A.apply(x)))  # True
"""

import numpy as np

import pycsou.abc as pyca
import pycsou.operator.linop.base as pyclb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


# Trigonometric Functions =====================================================
class Sin(pyca.DiffMap):
    r"""
    Trigonometric sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sin(x)`
    * :math:`f'(x) = \cos(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = k \pi, \, k \in \mathbb{Z}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = (2k + 1)
      \frac{\pi}{2}, \, k \in \mathbb{Z}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cos(arr))


def sin(op: pyct.OpT) -> pyct.OpT:
    return Sin(op.dim) * op


class Cos(pyca.DiffMap):
    r"""
    Trigonometric cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cos(x)`
    * :math:`f'(x) = -\sin(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = (2k + 1)
      \frac{\pi}{2}, \, k \in \mathbb{Z}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = k \pi, \, k
      \in \mathbb{Z}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(-xp.sin(arr))


def cos(op: pyct.OpT) -> pyct.OpT:
    return Cos(op.dim) * op


class Tan(pyca.DiffMap):
    r"""
    Trigonometric tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tan(x)`
    * :math:`f'(x) = \cos^{-2}(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-\pi, \pi]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-\pi, \pi]`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        v = xp.cos(arr)
        v **= 2
        return pyclb.DiagonalOp(1 / v)


def tan(op: pyct.OpT) -> pyct.OpT:
    return Tan(op.dim) * op


class ArcSin(pyca.DiffMap):
    r"""
    Inverse sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \arcsin(x)`
    * :math:`f'(x) = (1 - x^{2})^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        v = arr**2
        v *= -1
        v += 1
        xp.sqrt(v, out=v)
        return pyclb.DiagonalOp(1 / v)


def arcsin(op: pyct.OpT) -> pyct.OpT:
    return ArcSin(op.dim) * op


class ArcCos(pyca.DiffMap):
    r"""
    Inverse cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \arccos(x)`
    * :math:`f'(x) = -(1 - x^{2})^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccos(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        v = arr**2
        v *= -1
        v += 1
        xp.sqrt(v, out=v)
        return pyclb.DiagonalOp(-1 / v)


def arccos(op: pyct.OpT) -> pyct.OpT:
    return ArcCos(op.dim) * op


class ArcTan(pyca.DiffMap):
    r"""
    Inverse tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \arctan(x)`
    * :math:`f'(x) = (1 + x^{2})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 3 \sqrt{3} / 8`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \pm \frac{1}{\sqrt{3}}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 3 * np.sqrt(3) / 8
        #   max_{x \in R} |arctan''(x)|
        # = max_{x \in R} |2x / (1+x^2)^2|
        # = 3 \sqrt(3) / 8  [at x = +- 1/\sqrt(3)]

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = arr**2
        v += 1
        return pyclb.DiagonalOp(1 / v)


def arctan(op: pyct.OpT) -> pyct.OpT:
    return ArcTan(op.dim) * op


# Hyperbolic Functions ========================================================
class Sinh(pyca.DiffMap):
    r"""
    Hyperbolic sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sinh(x)`
    * :math:`f'(x) = \cosh(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sinh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cosh(arr))


def sinh(op: pyct.OpT) -> pyct.OpT:
    return Sinh(op.dim) * op


class Cosh(pyca.DiffMap):
    r"""
    Hyperbolic cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cosh(x)`
    * :math:`f'(x) = \sinh(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cosh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.sinh(arr))


def cosh(op: pyct.OpT) -> pyct.OpT:
    return Cosh(op.dim) * op


class Tanh(pyca.DiffMap):
    r"""
    Hyperbolic tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tanh(x)`
    * :math:`f'(x) = 1 - \tanh^{2}(x)`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 4 / 3 \sqrt{3}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \frac{1}{2}
      \ln(2 \pm \sqrt{3})`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 4 / (3 * np.sqrt(3))
        #   max_{x \in R} |tanh''(x)|
        # = max_{x \in R} |-2 tanh(x) [1 - tanh(x)^2|
        # = 4 / (3 \sqrt(3))  [at x = ln(2 +- \sqrt(3))]

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tanh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        v**2
        v *= -1
        v += 1
        return pyclb.DiagonalOp(v)


def tanh(op: pyct.OpT) -> pyct.OpT:
    return Tanh(op.dim) * op


class ArcSinh(pyca.DiffMap):
    r"""
    Inverse hyperbolic sine, element-wise.

    Notes
    -----
    * :math:`f(x) = \sinh^{-1}(x) = \ln(x + \sqrt{x^{2} + 1})`
    * :math:`f'(x) = (x^{2} + 1)^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \frac{2}{3 \sqrt{3}}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \pm \frac{1}{\sqrt{2}}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 2 / (3 * np.sqrt(3))
        #   max_{x \in R} |arcsinh''(x)|
        # = max_{x \in R} |-x (x^2 + 1)^{-3/2}|
        # = 2 / (3 \sqrt(3))  [at x = += 1 / \sqrt(2)]

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsinh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        v = arr**2
        v += 1
        xp.sqrt(v, out=v)
        return pyclb.DiagonalOp(1 / v)


def arcsinh(op: pyct.OpT) -> pyct.OpT:
    return ArcSinh(op.dim) * op


class ArcCosh(pyca.DiffMap):
    r"""
    Inverse hyperbolic cosine, element-wise.

    Notes
    -----
    * :math:`f(x) = \cosh^{-1}(x) = \ln(x + \sqrt{x^{2} - 1})`
    * :math:`f'(x) = (x^{2} - 1)^{-\frac{1}{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [1, \infty[`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [1, \infty[`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        xp = pycu.get_array_module(arr)
        v = arr**2
        v -= 1
        xp.sqrt(v, out=v)
        return pyclb.DiagonalOp(1 / v)


def arccosh(op: pyct.OpT) -> pyct.OpT:
    return ArcCosh(op.dim) * op


class ArcTanh(pyca.DiffMap):
    r"""
    Inverse hyperbolic tangent, element-wise.

    Notes
    -----
    * :math:`f(x) = \tanh^{-1}(x) = \frac{1}{2}\ln\left(\frac{1+x}{1-x}\right)`
    * :math:`f'(x) = (1 - x^{2})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = [-1, 1]`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = arr**2
        v *= -1
        v += 1
        return pyclb.DiagonalOp(1 / v)


def arctanh(op: pyct.OpT) -> pyct.OpT:
    return ArcTanh(op.dim) * op


# Exponential Functions =======================================================
class Exp(pyca.DiffMap):
    r"""
    Exponential, element-wise. (Default: base-E exponential.)

    Notes
    -----
    * :math:`f_{b}(x) = b^{x}`
    * :math:`f_{b}'(x) = b^{x} \ln(b)`
    * :math:`\vert f_{b}(x) - f_{b}(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f_{b}'(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}`.)
    * :math:`\vert f_{b}'(x) - f_{b}'(y) \vert \le \partial L \vert x - y \vert`, with
      diff-Lipschitz constant :math:`\partial L = \infty`.

      (Reason: :math:`f_{b}''(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}`.)
    """

    def __init__(self, dim: pyct.Integer, base: pyct.Real = None):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf
        self._base = base

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        if self._base is not None:
            out *= np.log(float(self._base))

        xp = pycu.get_array_module(arr)
        xp.exp(out, out=out)
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        if self._base is not None:
            v *= np.log(float(self._base))
        return pyclb.DiagonalOp(v)


def exp(op: pyct.OpT, base: pyct.Real = None) -> pyct.OpT:
    return Exp(op.dim, base) * op


class Log(pyca.DiffMap):
    r"""
    Logarithm, element-wise. (Default: base-E logarithm.)

    Notes
    -----
    * :math:`f_{b}(x) = \log_{b}(x)`
    * :math:`f_{b}'(x) = x^{-1} / \ln(b)`
    * :math:`\vert f_{b}(x) - f_{b}(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f_{b}'(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}_{+}`.)
    * :math:`\vert f_{b}'(x) - f_{b}'(y) \vert \le \partial L \vert x - y \vert`, with
      diff-Lipschitz constant :math:`\partial L = \infty`.

      (Reason: :math:`f_{b}''(x)` is unbounded on :math:`\text{dom}(f_{b}) = \mathbb{R}_{+}`.)
    """

    def __init__(self, dim: pyct.Integer, base: pyct.Real = None):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf
        self._base = base

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.log(arr)
        if self._base is not None:
            out /= np.log(float(self._base))
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = 1 / arr
        if self._base is not None:
            v /= np.log(float(self._base))
        return pyclb.DiagonalOp(v)


def log(op: pyct.OpT, base: pyct.Real = None) -> pyct.OpT:
    return Log(op.dim, base) * op


# Miscellaneous ===============================================================
class Clip(pyca.Map):
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

    def __init__(self, dim: pyct.Integer, a_min: pyct.Real = None, a_max: pyct.Real = None):
        super().__init__(shape=(dim, dim))
        if (a_min is None) and (a_max is None):
            raise ValueError("One of Parameter[a_min, a_max] must be specified.")
        self._llim = a_min
        self._ulim = a_max
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = arr.copy()
        xp.clip(
            arr,
            self._llim,
            self._ulim,
            out=out,
        )
        return out


def clip(op: pyct.OpT, a_min: pyct.Real = None, a_max: pyct.Real = None):
    return Clip(op.dim, a_min=a_min, a_max=a_max) * op


class Sqrt(pyca.DiffMap):
    r"""
    Non-negative square-root, element-wise.

    Notes
    -----
    * :math:`f(x) = \sqrt{x}`
    * :math:`f'(x) = 1 / 2 \sqrt{x}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}_{+}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}_{+}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        v *= 2
        return pyclb.DiagonalOp(1 / v)


def sqrt(op: pyct.OpT) -> pyct.OpT:
    return Sqrt(op.dim) * op


class Cbrt(pyca.DiffMap):
    r"""
    Cube-root, element-wise.

    Notes
    -----
    * :math:`f(x) = \sqrt[3]{x}`
    * :math:`f'(x) = 1 / 3 \sqrt[3]{x^{2}}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = \infty`.

      (Reason: :math:`f''(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        v **= 2
        v *= 3
        return pyclb.DiagonalOp(1 / v)


def cbrt(op: pyct.OpT) -> pyct.OpT:
    return Cbrt(op.dim) * op


class Square(pyca.DiffMap):
    r"""
    Square, element-wise.

    Notes
    -----
    * :math:`f(x) = x^{2}`
    * :math:`f'(x) = 2 x`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \infty`.

      (Reason: :math:`f'(x)` is unbounded on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` everywhere.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        out **= 2
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = arr.copy()
        v *= 2
        return pyclb.DiagonalOp(v)


def square(op: pyct.OpT) -> pyct.OpT:
    return Square(op.dim) * op


class Abs(pyca.Map):
    r"""
    Absolute value, element-wise.

    Notes
    -----
    * :math:`f(x) = \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.abs(arr)


def abs(op: pyct.OpT) -> pyct.OpT:
    return Abs(op.dim) * op


class Sign(pyca.Map):
    r"""
    Number sign indicator, element-wise.

    Notes
    -----
    * :math:`f(x) = x / \vert x \vert`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 2`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)


def sign(op: pyct.OpT) -> pyct.OpT:
    return Sign(op.dim) * op


class CumSum(pyca.SquareOp):
    r"""
    Cumulative sum of elements.

    Notes
    -----
    * :math:`[f(x_{1},\ldots,x_{N})]_{i} = \sum_{k=1}^{i} x_{k}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \sqrt{N (N+1) / 2}`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.sqrt(dim * (dim + 1) / 2)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.cumsum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = arr[..., ::-1].cumsum(axis=-1)[..., ::-1]
        return y


def cumsum(op: pyct.OpT) -> pyct.OpT:
    return CumSum(op.dim) * op


# Activation Functions ========================================================
class Gaussian(pyca.DiffMap):
    r"""
    Gaussian, element-wise.

    Notes
    -----
    * :math:`f(x) = \exp(-x^{2})`
    * :math:`f'(x) = -2 x \exp(-x^{2})`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \sqrt{2 / e}`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = \pm 1 / \sqrt{2}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = np.sqrt(2 / np.e)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = arr.copy()
        out **= 2
        out *= -1
        xp.exp(out, out=out)
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        v *= -2
        v *= arr
        return pyclb.DiagonalOp(v)


def gaussian(op: pyct.OpT) -> pyct.OpT:
    return Gaussian(op.dim) * op


class Sigmoid(pyca.DiffMap):
    r"""
    Sigmoid, element-wise.

    Notes
    -----
    * :math:`f(x) = (1 + e^{-x})^{-1}`
    * :math:`f'(x) = f(x) [ f(x) - 1 ]`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1 / 4`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x = 0`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1 / 6 \sqrt{3}`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = \ln(2 \pm \sqrt{3})`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1 / 4
        self._diff_lipschitz = 1 / (6 * np.sqrt(3))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        x = -arr
        xp.exp(x, out=x)
        x += 1
        return 1 / x

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        x = self.apply(arr)
        v = x.copy()
        x -= 1
        v *= x
        return pyclb.DiagonalOp(v)


def sigmoid(op: pyct.OpT) -> pyct.OpT:
    return Sigmoid(op.dim) * op


class SoftPlus(pyca.DiffMap):
    r"""
    Softplus, element-wise.

    Notes
    -----
    * :math:`f(x) = \ln(1 + e^{x})`
    * :math:`f'(x) = (1 + e^{-x})^{-1}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` on :math:`\text{dom}(f) = \mathbb{R}`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1 / 4`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 1 / 4

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.logaddexp(0, arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        f = Sigmoid(dim=self.dim)
        v = f.apply(arr)
        return pyclb.DiagonalOp(v)


def softplus(op: pyct.OpT) -> pyct.OpT:
    return SoftPlus(op.dim) * op


class LeakyReLU(pyca.Map):
    r"""
    Leaky rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = x \left[\mathbb{1}_{\ge 0}(x) + \alpha \mathbb{1}_{< 0}(x)\right], \quad \alpha \ge 0`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = \max(1, \alpha)`.
    """

    def __init__(self, dim: pyct.Integer, alpha: pyct.Real):
        super().__init__(shape=(dim, dim))
        self._alpha = float(alpha)
        assert self._alpha >= 0
        self._lipschitz = float(max(alpha, 1))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.where(arr >= 0, arr, arr * self._alpha)


def leakyrelu(op: pyct.OpT, alpha: pyct.Real) -> pyct.OpT:
    return LeakyReLU(dim=op.dim, alpha=alpha) * op


class ReLU(LeakyReLU):
    r"""
    Rectified linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = \lfloor x \rfloor_{+}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(dim=dim, alpha=0)


def relu(op: pyct.OpT) -> pyct.OpT:
    return ReLU(op.dim) * op


class SiLU(pyca.DiffMap):
    r"""
    Sigmoid linear unit, element-wise.

    Notes
    -----
    * :math:`f(x) = x / (1 + e^{-x})`
    * :math:`f'(x) = (1 + e^{-x} + x e^{-x}) / (1 + e^{-x})^{2}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1.1`.

      (Reason: :math:`\vert f'(x) \vert` is bounded by :math:`L` at :math:`x \approx 2.4`.)
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1 / 2`.

      (Reason: :math:`\vert f''(x) \vert` is bounded by :math:`\partial L` at :math:`x = 0`.)
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1.1
        self._diff_lipschitz = 1 / 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        f = Sigmoid(dim=self.dim)
        out = f.apply(arr)
        out *= arr
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        f = Sigmoid(dim=self.dim)
        xp = pycu.get_array_module(arr)
        a = xp.exp(-arr)
        a *= 1 + arr
        a += 1
        b = f.apply(arr)
        b **= 2
        return pyclb.DiagonalOp(a * b)


def silu(op: pyct.OpT) -> pyct.OpT:
    return SiLU(op.dim) * op


class SoftMax(pyca.DiffMap):
    r"""
    Softmax, element-wise.

    Notes
    -----
    * :math:`[f(x_{1},\ldots,x_{N})]_{i} = e^{x_{i}} / \sum_{k=1}^{N} e^{x_{k}}`
    * :math:`J_{f}(\mathbf{x}) = \text{diag}(f(\mathbf{x})) - f(\mathbf{x}) f(\mathbf{x})^{T}`
    * :math:`\vert f(x) - f(y) \vert \le L \vert x - y \vert`, with Lipschitz constant :math:`L = 1`.
    * :math:`\vert f'(x) - f'(y) \vert \le \partial L \vert x - y \vert`, with diff-Lipschitz
      constant :math:`\partial L = 1`.
    """

    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))
        self._lipschitz = 1
        self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        x = xp.exp(arr)
        out = x / x.sum(axis=-1, keepdims=True)
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        v = self.apply(arr)
        lhs = pyclb.DiagonalOp(v)
        rhs = pyca.LinFunc.from_array(v).gram()
        op = lhs - rhs
        return op


def softmax(op: pyct.OpT) -> pyct.OpT:
    return SoftMax(op.dim) * op
