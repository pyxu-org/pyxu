# #############################################################################
# ufunc.py
# ========
# Author : Kaan Okumus [okukaan@gmail.com]
# #############################################################################

r"""
Universal functions.

This module provides various universal functions.

Notes
------

* Implementation:

Example
-------
.. testsetup::

    import numpy as np

.. doctest::

    >>> from pycsou.operator.map import Sin
    >>> x = np.random.randn(10)
    >>> sin_x = Sin(x.shape)
    >>> res = sin_x.apply(x)
    >>> np_res = np.sin(x)
    >>> np.allclose(np_res, res)
    True
    >>> jacob = sin_x.jacobian(x)
    >>> jacob_res = jacob.apply(np.ones(10))
    >>> np.allclose(jacob_res, np.cos(x))
    True

* Every classes have its own functional interface in order to be able to combine with different maps:

Example
-------
.. testsetup::

    import numpy as np

.. doctest::

    >>> from pycsou.operator.map import sin
    >>> from pycsou.operator.linop.base import ExplicitLinOp
    >>> x = np.random.randn(10)
    >>> A = ExplicitLinOp(np.random.randn(10, 10))
    >>> sin_A = sin(A)
    >>> res = sin_A.apply(x)
    >>> np.allclose(res, np.sin(A.apply(x)))
    True
"""

import numpy as np

import pycsou.abc as pyca
import pycsou.operator.linop.base as pyclb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

# Trigonometric Functions


class Sin(pyca.DiffMap):
    r"""
    Trigonometric sine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\sin(x)}{dx} = \cos(x)}

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|\cos (x)|\} = 1` at :math:`x=\pi k` for :math:`k\in \mathbb{Z}`.

    * Differential Lipschitz constant: :math:`1`

    since :math:`\max\{|-\sin (x)|\} = 1` at :math:`x=\pi k + \frac{\pi}{2}` for :math:`k\in \mathbb{Z}`.

    See Also
    --------
    Cos, Tan, Arcsin, Arccos, Arctan
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cos(arr))


def sin(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sin`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sin(op.shape) * op


class Cos(pyca.DiffMap):
    r"""
    Trigonometric cosine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\cos(x)}{dx} = -\sin(x)}

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|-\sin (x)|\} = 1` at :math:`x=\pi k + \frac{\pi}{2}` for :math:`k\in \mathbb{Z}`.

    * Differential Lipschitz constant: :math:`1`

    since :math:`\max\{|-\cos (x)|\} = 1` at :math:`x=\pi k` for :math:`k\in \mathbb{Z}`.

    See Also
    --------
    Sin, Tan, Arcsin, Arccos, Arctan
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(-xp.sin(arr))


def cos(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Cos`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Cos(op.shape) * op


class Tan(pyca.DiffMap):
    r"""
    Trigonometric tangent, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\tan(x)}{dx} = \frac{1}{\cos^2(x)}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{1}{\cos^2(x)}|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{2\tan(x)}{\cos^2(x)}|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sin, Cos, Arcsin, Arccos, Arctan
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.cos(arr) ** 2)


def tan(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Tan`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Tan(op.shape) * op


class Arcsin(pyca.DiffMap):
    r"""
    Inverse sine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\arcsin(x)}{dx} = \frac{1}{\sqrt{1-x^2}}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{1}{\sqrt{1-x^2}}|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{x}{(1 - x)^{3/2}}|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sin, Cos, Tan, Arccos, Arctan
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.sqrt(1 - arr**2))


def arcsin(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arcsin`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arcsin(op.shape) * op


class Arccos(pyca.DiffMap):
    r"""
    Inverse cosine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\arccos(x)}{dx} = -\frac{1}{\sqrt{1-x^2}}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|-\frac{1}{\sqrt{1-x^2}}|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|-\frac{x}{(1 - x)^{3/2}}|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sin, Cos, Tan, Arcsin, Arctan
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccos(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(-1 / xp.sqrt(1 - arr**2))


def arccos(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arccos`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arccos(op.shape) * op


class Arctan(pyca.DiffMap):
    r"""
    Inverse tangent, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\arctan(x)}{dx} = \frac{1}{1+x^2}}

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|\frac{1}{1+x^2}|\} = 1` at :math:`x=0`.

    * Differential Lipschitz constant: :math:`\frac{3\sqrt{3}}{8}`

    since :math:`\max\{|-\frac{2x}{(1+x^2)^2}|\} = \frac{3\sqrt{3}}{8}` at :math:`x=\pm \frac{1}{\sqrt{3}}`.

    See Also
    --------
    Sin, Cos, Tan, Arcsin, Arccos
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 3 * np.sqrt(3) / 8  # Max of |arctan''(x)|=|2x/(1+x^2)^2| is 3sqrt(3)/8 at x=+-1/sqrt(3)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(1 / (1 + arr**2))


def arctan(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arctan`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arctan(op.shape) * op


# Hyperbolic Functions


class Sinh(pyca.DiffMap):
    r"""
    Hyperbolic sine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\sinh(x)}{dx} = \cosh(x)}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\cosh(x)|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\sinh(x)|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Cosh, Tanh, Arcsinh, Arccosh, Arctanh
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sinh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cosh(arr))


def sinh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sinh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sinh(op.shape) * op


class Cosh(pyca.DiffMap):
    r"""
    Hyperbolic cosine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\cosh(x)}{dx} = \sinh(x)}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\sinh(x)|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\cosh(x)|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sinh, Tanh, Arcsinh, Arccosh, Arctanh
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cosh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.sinh(arr))


def cosh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Cosh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Cosh(op.shape) * op


class Tanh(pyca.DiffMap):
    r"""
    Hyperbolic tangent, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\tanh(x)}{dx} = \frac{1}{\cosh^2(x)}}

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|\frac{1}{\cosh^2(x)}|\} = 1` at :math:`x=0`.

    * Differential Lipschitz constant: :math:`\frac{4}{3\sqrt{3}}`

    since :math:`\max\{|-\frac{2\tanh(x)}{\cosh^2(x)}|\} = \frac{4}{3\sqrt{3}}` at :math:`x= \frac{1}{2}\log(2 \pm \sqrt{3})`.

    See Also
    --------
    Sinh, Cosh, Arcsinh, Arccosh, Arctanh, Sigmoid, ReLU, GELU, Softplus, ELU, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 4 / (
            3 * np.sqrt(3)
        )  # Max of |tanh''(x)|=|2sech^2(x)tanh(x)| is 4/3sqrt(3) at x=+-log(2-sqrt(3))/2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tanh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.cosh(arr) ** 2)


def tanh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Tanh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Tanh(op.shape) * op


class Arcsinh(pyca.DiffMap):
    r"""
    Inverse hyperbolic sine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\sinh^{-1}(x)}{dx} = \frac{1}{\sqrt{1 + x^2}}}

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|\frac{1}{\sqrt{1 + x^2}}|\} = 1` at :math:`x=0`.

    * Differential Lipschitz constant: :math:`\frac{2}{3\sqrt{3}}`

    since :math:`\max\{|-\frac{x}{(1 + x^2)^{3/2}}|\} = \frac{2}{3\sqrt{3}}` at :math:`x=\pm \frac{1}{\sqrt{2}}`.

    See Also
    --------
    Sinh, Cosh, Tanh, Arccosh, Arctanh
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1  # Max of |arcsinh'(x)|=|1/sqrt(1+x^2)| is 1 at x=0
        self._diff_lipschitz = 2 / (
            3 * np.sqrt(3)
        )  # Max of |arcsinh''(x)|=|x/(1+x^2)^(3/2)| is 2/3sqrt(3) at x=+-1/sqrt(2)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsinh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / (xp.sqrt(1 + arr**2)))


def arcsinh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arcsinh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arcsinh(op.shape) * op


class Arccosh(pyca.DiffMap):
    r"""
    Inverse hyperbolic cosine, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\cosh^{-1}(x)}{dx} = \frac{1}{\sqrt{-1 + x}\sqrt{1 + x}}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|\frac{1}{\sqrt{-1 + x}\sqrt{1 + x}}|\}` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|-\frac{x}{(-1 + x)^{3/2}(1 + x)^{3/2}}|\}` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sinh, Cosh, Tanh, Arcsinh, Arctanh
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.sqrt(-1 + arr**2))


def arccosh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arccosh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arccosh(op.shape) * op


class Arctanh(pyca.DiffMap):
    r"""
    Inverse hyperbolic tangent, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d\tanh^{-1}(x)}{dx} = \frac{1}{1-x^2}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|\frac{1}{1-x^2}|\}` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|\frac{2x}{(1-x^2)^2}|\}` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sinh, Cosh, Tanh, Arcsinh, Arccosh
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(1 / (1 - arr**2))


def arctanh(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Arctanh`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Arctanh(op.shape) * op


# Exponentials and logarithms


class Exp(pyca.DiffMap):
    r"""
    Exponential, element-wise. (Default: base-E exponential.)

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d (base)^{x}}{dx} = (base)^{x}\log(base)}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|(base)^{x}\log(base)|\}` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|(base)^{x}\log^2(base)|\}` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Log
    """

    def __init__(self, shape: pyct.Shape, base: pyct.Real = None):
        r"""
        Parameters
        ----------
        shape: :py:class:`pyct.Shape`
            Shape of input array.
        base: :py:class:`pyct.Real`
            Base parameter. Default is `None`, which results in base-E exponential.
        """
        super().__init__(shape)
        self._base = pycrt.coerce(base)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = arr.copy()
        if self._base is not None:
            print(type(self._base))
            out *= xp.log(self._base)
        return xp.exp(out)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        y = self.apply(arr)
        if self._base is not None:
            y *= xp.log(self._base)
        return pyclb.DiagonalOp(y)


def exp(op: pyca.Map, base: pyct.Real = None) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Exp`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.
    base: :py:class:`pyct.Real`
        Base parameter. Default is `None`, which results in base-E exponential.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Exp(op.shape, base) * op


class Log(pyca.DiffMap):
    r"""
    Logarithm, element-wise. (Default: base-E logarithm.)

    Notes
    -----
    * Derivative:

    .. math::
        {\frac{d \log_{base}(x)}{dx} = \frac{1}{x\log(base)}}.

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|\frac{1}{x\log(base)}|\}` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`\max\{|-\frac{1}{x^2\log(base)}|\}` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Exp
    """

    def __init__(self, shape: pyct.Shape, base: pyct.Real = None):
        r"""
        Parameters
        ----------
        shape: :py:class:`pyct.Shape`
            Shape of input array.
        base: :py:class:`pyct.Real`
            Base parameter. Default is `None`, which results in base-E logarithm.
        """
        super().__init__(shape)
        self._base = pycrt.coerce(base)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.log(arr)
        if self._base is not None:
            out /= xp.log(self._base)
        return out

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        y = 1 / arr
        if self._base is not None:
            y /= xp.log(self._base)
        return pyclb.DiagonalOp(y)


def log(op: pyca.Map, base: pyct.Real = None) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Log`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.
    base: :py:class:`pyct.Real`
        Base parameter. Default is`None`, which results in base-E logarithm.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Log(op.shape, base) * op


# Sums and Products


class Prod(pyca.DiffFunc):
    r"""
    Product of array elements.

    Notes
    -----
    * Function:

    .. math::
        {f(\mathbf{x}) = \prod_{i=1}^N x_i}

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N`.

    * Gradient:

    .. math::
        \nabla f(\mathbf{x}) = \left[ \prod_{i \in \{2,3,...,N\}} x_i, \prod_{i \in \{1,3,...,N\}} x_i, \cdots, \prod_{i \in \{1,2,...,N-1\}} x_i\right]^T

    since :math:`{\frac{\partial f(\mathbf{x})}{\partial x_k} = \prod_{1 \leq i \leq N, i \neq k} x_i}`.

    * Lipschitz constant: :math:`\begin{cases}1, & \text{if} \ N=1 \\ \infty, & \text{otherwise}\end{cases}`.

    * Differential Lipschitz constant: :math:`\begin{cases}0, & \text{if} \ N=1 \\ \sqrt{2} & \text{if} \ N=2 \\ \infty, & \text{otherwise}\end{cases}`.

    See Also
    --------
    Sum, Cumprod, Cumsum
    """

    def __init__(self, dim: int):
        super().__init__(shape=(1, dim))
        self._lipschitz = 1 if (dim == 1) else np.inf
        self._diff_lipschitz = 0 if (dim == 1) else np.sqrt(2) if (dim == 2) else np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.prod(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        g = xp.repeat(arr[..., None, :], arr.shape[-1], axis=-2)
        e = xp.broadcast_to(xp.eye(*g.shape[-2:]), g.shape).astype(bool)
        g[e] = 1.0
        return xp.prod(g, axis=-1)


def prod(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Prod`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Prod(op.shape[0]) * op


class Sum(pyca.DiffFunc):
    r"""
    Sum of array elements.

    Notes
    -----
    * Function:

    .. math::
        {f(\mathbf{x}) = \sum_{i=1}^N x_i}

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N`.

    * Gradient:

    .. math::
        \nabla f(\mathbf{x}) = \left[ 1, 1, \cdots, 1\right]^T

    since :math:`\frac{\partial f(\mathbf{x})}{\partial x_k} = 1`.

    * Lipschitz constant: :math:`\sqrt{N}`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Prod, Cumprod, Cumsum
    """

    def __init__(self, dim: int):
        super().__init__(shape=(1, dim))
        self._lipschitz = np.sqrt(dim) if (dim is not None) else np.inf
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.sum(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.ones_like(arr)


def sum(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sum`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sum(op.shape[0]) * op


class Cumprod(pyca.DiffMap):
    r"""
    Cumulative product of elements.

    Notes
    -----
    * Function:

    .. math::
        f(\mathbf{x}) = \left[ x_1, x_1x_2, \cdots, \prod_{i=1}^N x_i\right]^T

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N`.

    * Jacobian:

    .. math::
        \begin{bmatrix}
            1 & 0 & \cdots & 0 \\
            x_2 & x_1 & \cdots & 0 \\
            x_2x_3 & x_1x_3 & \cdots & 0 \\
            \vdots & \vdots & & \vdots \\
            \prod_{i\in\{2,3,...,N\}} x_i & \prod_{i\in\{1,3,...,N\}} x_i & \cdots & \prod_{i\in\{1,2,...,N-1\}} x_i \\
        \end{bmatrix} \in \mathbb{R}^{N\times N}

    * Lipschitz constant: :math:`\begin{cases}1, & \text{if} \ N=1 \\ \infty, & \text{otherwise}\end{cases}`.

    * Differential Lipschitz constant: :math:`\begin{cases}0, & \text{if} \ N=1 \\ \sqrt{2} & \text{if} \ N=2 \\ \infty, & \text{otherwise}\end{cases}`.

    See Also
    --------
    Prod, Sum, Cumsum
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1 if (shape[-1] == 1) else np.inf
        self._diff_lipschitz = 0 if (shape[-1] == 1) else np.sqrt(2) if (shape[-1] == 2) else np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumprod(arr, axis=-1)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.ExplicitLinOp:
        assert arr.ndim == 1  # Jacobian matrix is only valid for vectors.
        xp = pycu.get_array_module(arr)
        g = xp.repeat(arr[:, None], arr.shape[-1], axis=-1)
        e = xp.eye(*g.shape[-2:]).astype(bool)
        g[e] = 1.0
        return pyclb.ExplicitLinOp(xp.tri(*g.shape[-2:]) * xp.cumprod(g, axis=1))


def cumprod(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Cumprod`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Cumprod(op.shape) * op


class Cumsum(pyca.DiffMap):
    r"""
    Cumulative sum of elements.

    Notes
    -----
    * Function:

    .. math::
        f(\mathbf{x}) = \left[ x_1, x_1+x_2, \cdots, \sum_{i=1}^N x_i\right]^T

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N`.

    * Jacobian matrix:

    .. math::
        \begin{bmatrix}
            1 & 0 & \cdots & 0 \\
            1 & 1 & \cdots & 0 \\
            \vdots & \vdots & & \vdots \\
            1 & 1 & \cdots & 1 \\
        \end{bmatrix} \in \mathbb{R}^{N\times N}

    * Lipschitz constant: :math:`\frac{N(N+1)}{2}`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Prod, Sum, Cumprod
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = np.sqrt(shape[-1] * (shape[-1] + 1) / 2)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumsum(arr, axis=-1)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.ExplicitLinOp:
        assert arr.ndim == 1  # Jacobian matrix is only valid for vectors.
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.tri(self.shape[0]))


def cumsum(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Cumsum`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Cumsum(op.shape) * op


# Miscellaneous


class Clip(pyca.DiffMap):
    r"""
    Clip (limit) values in an array, element-wise.

    Notes
    -----
    * Function:

    .. math::
        f(x, a_{min}, a_{max}) =
        \begin{cases}
            a_{min}, & \text{if} \ x \leq a_{min} \\
            a_{max}, & \text{if} \ x \geq a_{max} \\
            x, & \text{otherwise}
        \end{cases}

    where :math:`a_{min}` and :math:`a_{max}` are minimum and maximum values, respectively.

    * Derivative:

    .. math::
        \frac{\partial f(x, a_{min}, a_{max})}{\partial x} =
        \begin{cases}
            1, & \text{if} \ a_{min} \leq x \leq a_{max} \\
            0, & \text{otherwise}
        \end{cases}

    * Lipschitz constant: :math:`1`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Sqrt, Cbrt, Square, Abs, Sign
    """

    def __init__(self, shape: pyct.Shape, a_min: pyct.Real = None, a_max: pyct.Real = None):
        r"""
        Parameters
        ----------
        shape: :py:class:`pyct.Shape`
            Shape of input array.
        a_min: :py:class:`pyct.Real`
            Minimum value. Default is `None`.
        a_max: :py:class:`pyct.Real`
            Maximum value. Default is `None`.
        """

        super().__init__(shape)
        if (a_min is None) and (a_max is None):
            raise ValueError("One of Parameter[a_min, a_max] must be specified.")
        else:
            self._llim = pycrt.coerce(a_min)
            self._ulim = pycrt.coerce(a_max)
        self._lipschitz = 1
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.clip(self._llim, self._ulim)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.where(xp.logical_and(arr <= self._ulim, arr >= self._llim), 1.0, 0.0))


def clip(op: pyca.Map, a_min: pyct.Real = None, a_max: pyct.Real = None):
    r"""
    Functional interface of :py:class:`Clip`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.
    a_min: :py:class:`pyct.Real`
        Minimum value. Default is None.
    a_max: :py:class:`pyct.Real`
        Maximum value. Default is None.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Clip(op.shape, a_min, a_max) * op


class Sqrt(pyca.DiffMap):
    r"""
    Non-negative square-root, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        \frac{d \sqrt{x}}{d x} = \frac{1}{2\sqrt{x}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{1}{2\sqrt{x}}|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|-\frac{1}{4x^{3/2}}|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Clip, Cbrt, Square, Abs, Sign
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(1.0 / (2.0 * self.apply(arr)))


def sqrt(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sqrt`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sqrt(op.shape) * op


class Cbrt(pyca.DiffMap):
    r"""
    Cube-root, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        \frac{d \sqrt[3]{x}}{d x} = \frac{1}{3\sqrt[3]{x^2}}

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\frac{1}{3\sqrt[3]{x^2}}|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|-\frac{2}{9x^{5/3}}|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Clip, Sqrt, Square, Abs, Sign
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(1.0 / (3.0 * (self.apply(arr) ** 2.0)))


def cbrt(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Cbrt`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Cbrt(op.shape) * op


class Square(pyca.DiffMap):
    r"""
    Square, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        \frac{d x^2}{d x} = 2x

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|2x|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`2`.

    See Also
    --------
    Clip, Sqrt, Cbrt, Abs, Sign
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.square(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(2.0 * arr)


def square(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Square`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Square(op.shape) * op


class Abs(pyca.DiffMap):
    r"""
    Absolute value, element-wise.

    Notes
    -----
    * Derivative:

    .. math::
        \frac{d |x|}{d x} =
        \begin{cases}
            1, & \text{if} \ x \geq 0 \\
            -1, & \text{otherwise}
        \end{cases}

    * Lipschitz constant: :math:`1`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Clip, Sqrt, Cbrt, Square, Sign
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.absolute(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.sign(arr))


def abs(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Abs`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Abs(op.shape) * op


class Sign(pyca.DiffMap):
    r"""
    Sign, element-wise.

    Notes
    -----
    * Function:

    .. math::
        f(x) =
        \begin{cases}
            1, & \text{if} \ x > 0 \\
            0, & \text{if} \ x = 0 \\
            -1, & \text{otherwise}
        \end{cases}

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = 0

    for :math:`x \in \mathbb{R}-\{0\}`.

    * Lipschitz constant: :math:`0`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Clip, Sqrt, Cbrt, Square, Abs
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.zeros_like(arr))


def sign(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sign`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sign(op.shape) * op


# Activation Functions (Tanh already implemented)


class Sigmoid(pyca.DiffMap):
    r"""
    Sigmoid, element-wise.

    Notes
    -----
    * Function:

    .. math::
       \sigma (x) = \frac{1}{1 + e^{-x}}

    * Derivative:

    .. math::
        \frac{d \sigma(x)}{d x} = \sigma(x)(\sigma(x)-1)=\frac{e^{-x}}{(e^{-x} + 1)^2}

    * Lipschitz constant: :math:`0.25`

    since :math:`\max\{|\frac{e^{-x}}{(e^{-x} + 1)^2}|\} = 0.25` at :math:`x=0`.

    * Differential Lipschitz constant: :math:`\frac{1}{6\sqrt{3}}`

    since :math:`\max\{|-\frac{e^x(-1 + e^x}{(1 + e^x)^3}|\} = \frac{1}{6\sqrt{3}}` at :math:`x = \log (2 \pm \sqrt{3})`.

    See Also
    --------
    Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 0.25  # Max of |\sigma'(x)|=|\sigma(x)(1-\sigma(x))| is 0.25 at x=0
        self._diff_lipschitz = 1.0 / (
            6.0 * np.sqrt(3)
        )  # Max of |\sigma''(x)|=|(\exp^x (-1 + \exp^x))/(1 + \exp^x)^3| is 1/(6*\sqrt(3)) at x=log(2+-\sqrt(3))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return 1.0 / (1.0 + xp.exp(-arr))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(self.apply(arr) * (1.0 - self.apply(arr)))


def sigmoid(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Sigmoid`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Sigmoid(op.shape) * op


class ReLU(Clip):
    r"""
    Rectified linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = x^+ = \max (0,x)

    * Derivative:

    .. math::
        \frac{d x^+}{d x} =
        \begin{cases}
            1, & \text{if} \ x \geq 0 \\
            0, & \text{otherwise}
        \end{cases}

    * Lipschitz constant: :math:`1`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Sigmoid, Tanh, GELU, Softplus, ELU, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape, a_min=0, a_max=None)


def relu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`ReLU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return ReLU(op.shape) * op


class GELU(pyca.DiffMap):
    r"""
    Gaussian error linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = \frac{x}{2} \left( 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right)

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \frac{e^{-x^2/2} x}{\sqrt{2\pi}}
        + \frac{1}{2}\left( 1 + \text{erf} \left( \frac{x}{\sqrt{2}} \right)\right)

    * Lipschitz constant: :math:`\frac{1}{2}(\text{erf}(1)+1) + \frac{1}{e \sqrt{\pi}}`

    since :math:`\max\{|f'(x)|\} = \frac{1}{2}(\text{erf}(1)+1) + \frac{1}{e \sqrt{\pi}}` at :math:`x=\sqrt{2}`.

    * Differential Lipschitz constant: :math:`\sqrt{\frac{2}{\pi}}`

    since :math:`\max\{|f''(x)|\} = \sqrt{\frac{2}{\pi}}` at :math:`x=0`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, Softplus, ELU, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        from scipy.special import erf

        self._lipschitz = (np.sqrt(2) * erf(1) + 2) / 4 + 1 / (np.exp(1) * np.sqrt(2 * np.pi))
        self._diff_lipschitz = np.sqrt(2 / np.pi)

    def _get_erf_function(self, arr: pyct.NDArray):
        # Update erf function according to input array
        try:
            from scipy.special import erf

            erf(arr)
            return erf
        except:
            from cupyx.scipy.special import erf

            return erf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        erf = self._get_erf_function(arr)
        arr_dtype = arr.dtype
        if arr_dtype == "float128":
            arr = arr.astype("float64")  # scipy erf function does not support float128
        return arr * (1.0 + xp.array(erf(arr).astype(arr_dtype)) / xp.sqrt(2.0)) / 2.0

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        arr_dtype = arr.dtype
        arr_erf = (
            arr.astype("float64") if arr_dtype == "float128" else arr
        )  # scipy erf function does not support float128
        xp = pycu.get_array_module(arr)
        erf = self._get_erf_function(arr)
        op1 = (xp.sqrt(2.0) * xp.array(erf(arr_erf).astype(arr_dtype)) + 1.0) / 4.0
        op2 = xp.exp(-(arr**2.0)) * arr / xp.sqrt(2.0 * np.pi)
        return pyclb.DiagonalOp(op1 + op2)


def gelu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`GELU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return GELU(op.shape) * op


class Softplus(pyca.DiffMap):
    r"""
    Softplus, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f(x) = \ln (1 + e^x)

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \frac{1}{1 + e^{-x}} = \sigma(x)

    * Lipschitz constant: :math:`1`

    since :math:`\max\{|\sigma(x)|\} = 1` as :math:`x\rightarrow \pm \infty`.

    * Differential Lipschitz constant: :math:`\frac{1}{6\sqrt{3}}`

    since :math:`\max\{|\sigma(x)(\sigma(x)-1)|\} = 0.25` at :math:`x = 0`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, ELU, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.25

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log(xp.exp(arr) + 1.0)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1.0 / (1.0 + xp.exp(-arr)))


def softplus(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Softplus`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Softplus(op.shape) * op


class ELU(pyca.DiffMap):
    r"""
    Exponential linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) =
       \begin{cases}
            \alpha (e^x - 1) & \text{if} \ x \leq 0 \\
            x & \text{otherwise} \\
       \end{cases}

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} =
        \begin{cases}
            \alpha e^x & \text{if} \ x \leq 0 \\
            1 & \text{otherwise} \\
        \end{cases}

    * Lipschitz constant: :math:`\max(1, |\alpha|)`.

    * Differential Lipschitz constant: :math:`\max(0, |\alpha|)`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, SELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape, alpha: pyct.Real = None):
        r"""
        Parameters
        ----------
        shape: :py:class:`pyct.Shape`
            Shape of input array.
        alpha: :py:class:`pyct.Real`
            ELU parameter. Default is None, which results in error.
        """
        super().__init__(shape)
        if alpha is None:
            raise ValueError("Parameter[alpha] must be specified.")
        self._alpha = alpha
        self._lipschitz = max(1, np.abs(alpha))
        self._diff_lipschitz = max(0, np.abs(alpha))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.where(arr >= 0, arr, self._alpha * (xp.exp(arr) - 1.0))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        elu = self.apply(arr)
        return pyclb.DiagonalOp(xp.where(elu >= 0, 1.0, elu + self._alpha))


def elu(op: pyca.Map, alpha: pyct.Real) -> pyca.Map:
    r"""
    Functional interface of :py:class:`ELU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.
    alpha: :py:class`pyct.Real`
        ELU parameter. Default is None, which results in error.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return ELU(op.shape, alpha) * op


class SELU(pyca.DiffMap):
    r"""
    Scaled exponential linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = \lambda
       \begin{cases}
            \alpha (e^x - 1) & \text{if} \ x \leq 0 \\
            x & \text{otherwise} \\
       \end{cases}

    with parameters :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \lambda
        \begin{cases}
            \alpha e^x & \text{if} \ x \leq 0 \\
            1 & \text{otherwise} \\
        \end{cases}

    * Lipschitz constant: :math:`\lambda \max(1, |\alpha|)`.

    * Differential Lipschitz constant: :math:`\lambda \max(0, |\alpha|)`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, LeakyReLU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._alpha = 1.67326
        self._lambda = 1.0507
        self._lipschitz = self._lambda * max(1, self._alpha)
        self._diff_lipschitz = self._lambda * max(0, self._alpha)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return self._lambda * xp.where(arr >= 0, arr, self._alpha * (xp.exp(arr) - 1.0))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        elu = self.apply(arr)
        return pyclb.DiagonalOp(self._lambda * xp.where(elu >= 0, 1.0, elu + self._alpha))


def selu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`SELU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return SELU(op.shape) * op


class LeakyReLU(pyca.DiffMap):
    r"""
    Leaky rectified linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) =
       \begin{cases}
            \alpha x & \text{if} \ x < 0 \\
            x & \text{otherwise} \\
       \end{cases}

    with leaky parameters :math:`\alpha`.

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \lambda
        \begin{cases}
            \alpha & \text{if} \ x < 0 \\
            1 & \text{otherwise} \\
       \end{cases}

    * Lipschitz constant: :math:`\max(1, |\alpha|)`.

    * Differential Lipschitz constant: :math:`0`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, SiLU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape, alpha: pyct.Real = None):
        r"""
        Parameters
        ----------
        shape: :py:class:`pyct.Shape`
            Shape of input array.
        alpha: :py:class:`pyct.Real`
            Leaky parameter. Default is `None`, which results in error.
        """
        super().__init__(shape)
        if alpha is None:
            raise ValueError("Parameter[alpha] must be specified.")
        self._alpha = alpha
        self._lipschitz = max(np.abs(alpha), 1)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.where(arr >= 0, arr, arr * self._alpha)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.where(arr >= 0, 1.0, self._alpha))


def leakyrelu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`LeakyReLU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.
    alpha: :py:class:`pyct.Real`
        Leaky parameter. Default is `None`, which results in error.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return LeakyReLU(op.shape) * op


class SiLU(pyca.DiffMap):
    r"""
    Sigmoid linear unit, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = \frac{x}{1 + e^{-x}}

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \frac{1 + e^{-x} + xe^{-x}}{(1 + e^{-x})^2}

    * Lipschitz constant: :math:`1.0999`

    since :math:`\max\{| \frac{1 + e^{-x} + xe^{-x}}{(1 + e^{-x})^2} |\} \approx 1.0998` at :math:`x\approx 2.3994`.

    * Differential Lipschitz constant: :math:`0`.

    since :math:`\max\{| f''(x) |\} = 0.5` at :math:`x = 0`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyRELU, Gaussian, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1.0999  # Max of |silu'(x)| = |(e^x(-e^x(x-2)+x-2))/(1+e^x)^3| is 1.0999 at x=2.3994
        self._diff_lipschitz = 0.5  # Max of |silu''(x)| = |(e^x(-e^x(x-2)+x+2))/((1+e^x)^3)| is 0.5 at x=0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return arr / (1.0 + xp.exp(-arr))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        exp_arr = xp.exp(arr)
        return pyclb.DiagonalOp(((1 + exp_arr + arr) * exp_arr) / ((1 + exp_arr) ** 2))


def silu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`SiLU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return SiLU(op.shape) * op


class Gaussian(pyca.DiffMap):
    r"""
    Gaussian (\exp^(-x^2)), element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = e^{-x^2}

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = -2xe^{-x^2}

    * Lipschitz constant: :math:`\sqrt{\frac{2}{e}}`

    since :math:`\max\{| -2xe^{-x^2} |\} = \sqrt{\frac{2}{e}}` at :math:`x = \pm \frac{1}{\sqrt{2}}`.

    * Differential Lipschitz constant: :math:`2`

    since :math:`\max\{| e^{-x^2} (-2 + 4x^2) |\} = 0.5` at :math:`x = 0`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyRELU, SiLU, GCU, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = np.sqrt(2 / np.exp(1))  # Max of |f'(x)|=|-2xe^(-x^2)| is sqrt(2/e) at x = +-1/sqrt(2)
        self._diff_lipschitz = 2  # Max of |f''(x)|=|(4x^2-2)*e^(-x^2)| is 2 at x=0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.exp(-(arr**2.0))

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        return pyclb.DiagonalOp(-2.0 * arr * self.apply(arr))


def gaussian(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Gaussian`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Gaussian(op.shape) * op


class GCU(pyca.DiffMap):
    r"""
    Growing cosine, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (x) = x \cos(x)

    * Derivative:

    .. math::
        \frac{d f(x)}{d x} = \cos (x) - x \sin (x)

    * Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|\cos (x) - x \sin (x)|` is :math:`\mathbb{R}^+`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined, since range of :math:`|-2\sin (x) - x \cos (x)|` is :math:`\mathbb{R}^+`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyRELU, SiLU, Gaussian, Softmax, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return arr * xp.cos(arr)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.DiagonalOp:
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cos(arr) - arr * xp.sin(arr))


def gcu(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`GCU`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return GCU(op.shape) * op


class Softmax(pyca.DiffMap):
    r"""
    Softmax, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (\mathbf{x}) = \left[ \frac{e^{x_1}}{\sum_{j=1}^N e^{x_j}}, \cdots, \frac{e^{x_N}}{\sum_{j=1}^N e^{x_j}} \right]^T

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N` and the
    i-th element of :math:`f (\mathbf{x})` can be denoted as :math:`f_i(\mathbf{x})`.

    * Jacobian matrix:

    .. math::
        \begin{bmatrix}
            f_1(\mathbf{x})(1 - f_1(\mathbf{x})) & -f_1(\mathbf{x})f_2(\mathbf{x}) & \cdots & -f_1(\mathbf{x})f_N(\mathbf{x}) \\
            -f_1(\mathbf{x})f_2(\mathbf{x}) & f_2(\mathbf{x})(1 - f_2(\mathbf{x})) & \cdots & -f_2(\mathbf{x})f_N(\mathbf{x}) \\
            \vdots & \vdots & & \vdots \\
            -f_1(\mathbf{x})f_N(\mathbf{x}) & -f_2(\mathbf{x})f_N(\mathbf{x}) & \cdots & f_N(\mathbf{x})(1 - f_N(\mathbf{x})) \\
        \end{bmatrix}

    * Lipschitz constant: :math:`1`.

    * Differential Lipschitz constant: :math:`1`.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyRELU, SiLU, Gaussian, GCU, Maxout
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        exp_arr = xp.exp(arr)
        return exp_arr / xp.sum(exp_arr, axis=-1)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyclb.ExplicitLinOp:
        xp = pycu.get_array_module(arr)
        S = self.apply(arr)
        mtx = -S[:, None] * S[None, :]
        mtx = mtx + xp.eye(mtx.shape[0])
        return pyclb.ExplicitLinOp(mtx)


def softmax(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Softmax`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Softmax(op.shape) * op


class Maxout(pyca.DiffFunc):
    r"""
    Maxout class, element-wise.

    Notes
    -----
    * Function:

    .. math::
       f (\mathbf{x}) = \max_{i} x_i

    where :math:`\mathbf{x} = [x_1,x_2,\cdots,x_N]^T \in \mathbb{R}^N` for any positive integer :math:`N`.

    * Gradient:

    .. math::
        \nabla f(\mathbf{x}) = \left[ 0, \cdots, 1, \cdots, 0\right]^T

    where :math:`1` is in the i-th element of :math:`\nabla f(\mathbf{x})` where :math:`\mathbf{x}` has its maximum element.

    * Lipschitz constant: :math:`1`.

    * Differential Lipschitz constant: :math:`\infty`, i.e. undefined.

    See Also
    --------
    Sigmoid, Tanh, ReLU, GELU, Softplus, ELU, SELU, LeakyRELU, SiLU, Gaussian, GCU, Maxout
    """

    def __init__(self, dim: int):
        super().__init__(shape=(1, dim))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.array([xp.max(arr, axis=-1)])

    @pycrt.enforce_precision(i="arr")
    def _generate_grad(self, arr):
        xp = pycu.get_array_module(arr)
        arr_dtype = arr.dtype
        return xp.where(arr == self.apply(arr), 1.0, 0.0).astype(arr_dtype)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.apply_along_axis(self._generate_grad, axis=-1, arr=arr)


def maxout(op: pyca.Map) -> pyca.Map:
    r"""
    Functional interface of :py:class:`Maxout`.

    Parameters
    ----------
    op: :py:class:`pycsou.abc.operator.Map`
        Input map.

    Returns
    -------
    :py:class:`pycsou.abc.operator.Map`
        Output map.
    """
    return Maxout(op.shape) * op
