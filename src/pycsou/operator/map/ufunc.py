import numpy as np

import pycsou.abc as pyca
import pycsou.operator.linop.base as pyclb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

# Trigonometric Functions


class Sin(pyca.DiffMap):
    """
    Trigonometric sine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cos(arr))


def sin(op: pyca.Map) -> pyca.Map:
    return Sin(op.shape)


class Cos(pyca.DiffMap):
    """
    Trigonometric cosine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(-xp.sin(arr))


def cos(op: pyca.Map) -> pyca.Map:
    return Cos(op.shape)


class Tan(pyca.Map):
    """
    Trigonometric tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)


def tan(op: pyca.Map) -> pyca.Map:
    return Tan(op.shape)


class Arcsin(pyca.Map):
    """
    Inverse sine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)


def arcsin(op: pyca.Map) -> pyca.Map:
    return Arcsin(op.shape)


class Arccos(pyca.Map):
    """
    Inverse cosine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccos(arr)


def arccos(op: pyca.Map) -> pyca.Map:
    return Arccos(op.shape)


class Arctan(pyca.DiffMap):
    """
    Inverse tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.65  # Sepand: how is this computed?

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / (1 + arr**2))


def arctan(op: pyca.Map) -> pyca.Map:
    return Arctan(op.shape)


# Hyperbolic Functions


class Sinh(pyca.DiffMap):
    """
    Hyperbolic sine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.cosh(arr))


def sinh(op: pyca.Map) -> pyca.Map:
    return Sinh(op.shape)


class Cosh(pyca.DiffMap):
    """
    Hyperbolic cosine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cosh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.sinh(arr))


def cosh(op: pyca.Map) -> pyca.Map:
    return Cosh(op.shape)


class Tanh(pyca.DiffMap):
    """
    Hyperbolic tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.77  # Sepand: how is this computed?

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tanh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 - xp.tanh(arr) ** 2)


def tanh(op: pyca.Map) -> pyca.Map:
    return Tanh(op.shape)


class Arcsinh(pyca.DiffMap):
    """
    Inverse hyperbolic sine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1  # Sepand: how is this computed?
        self._diff_lipschitz = 0.39  # Sepand: how is this computed?

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / (xp.sqrt(1 + arr**2)))


def arcsinh(op: pyca.Map) -> pyca.Map:
    return Arcsinh(op.shape)


class Arccosh(pyca.Map):
    """
    Inverse hyperbolic cosine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)


def arccosh(op: pyca.Map) -> pyca.Map:
    return Arccosh(op.shape)


class Arctanh(pyca.Map):
    """
    Inverse hyperbolic tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)


def arctanh(op: pyca.Map) -> pyca.Map:
    return Arctanh(op.shape)


# Exponentials and logarithms


class Exp(pyca.DiffMap):
    """
    Exponential function, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.exp(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.exp(arr))


def exp(op: pyca.Map) -> pyca.Map:
    return Exp(op.shape)


class Log(pyca.DiffMap):
    """
    Logarithm, element-wise. (Default: base-E logarithm.)
    """

    def __init__(self, shape: pyct.Shape, base: pyct.Real = None):
        super().__init__(shape)
        self._base = base

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.log(arr)
        if self._base is not None:
            out /= np.log(self._base)
        return out

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        y = 1 / arr
        if self._base is not None:
            y /= np.log(self._base)
        return pyclb.DiagonalOp(y)


def log(op: pyca.Map, base: pyct.Real = None) -> pyca.Map:
    return Log(op.shape, base)


# Sums and Products


class Prod(pyca.DiffFunc):
    """
    Product of array elements.
    """

    def __init__(self):
        super().__init__(shape=(1, None))
        # Sepand: lipschitz/diff_lipschitz?

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.prod(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        # Sepand: to implement safely. (Beware of 0s.)
        # f'(x)[q] = \prod_{k \ne q} x_{k}
        pass


class Sum(pyca.DiffFunc):
    r"""
    Sum of array elements.
    """

    def __init__(self):
        super().__init__(shape=(1, None))
        # Sepand: lipschitz/diff_lipschitz?

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.sum(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.ones_like(arr)


class Cumprod(pyca.DiffMap):
    """
    Cumulative product of elements.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumprod(arr, axis=-1)

    def jacobian(self, arr: pyct.NDArray):
        # Sepand: to implement
        pass


class Cumsum(pyca.DiffMap):
    """
    Cumulative sum of elements.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumsum(arr, axis=-1)

    def jacobian(self, arr: pyct.NDArray):
        # Sepand: to implement
        pass


# Miscellaneous


class Clip(pyca.Map):
    r"""
    Clip (limit) the values in an array. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray, a_min=0.0, a_max=1.0) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        a_min: Float
            Minimum value. Default is 0.0.
        a_max: Float
            Maximum value. Default is 1.0.

        Returns
        -------
        NDArray
            Array with elements of arr but where values < a_{min} are replaced with a_{min} and
            those > a_{max} with a_{max}
        """
        return arr.clip(a_min, a_max)


class Sqrt(pyca.Map):
    """
    Non-negative square-root, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)


def sqrt(op: pyca.Map) -> pyca.Map:
    return Sqrt(op.shape)


class Cbrt(pyca.Map):
    """
    Cube-root, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)


def cbrt(op: pyca.Map) -> pyca.Map:
    return Cbrt(op.shape)


class Square(pyca.DiffMap):
    """
    Square function, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.square(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(2 * arr)


def square(op: pyca.Map) -> pyca.Map:
    return Square(op.shape)


class Abs(pyca.DiffMap):
    """
    Absolute value, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.absolute(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(xp.sign(arr))


def abs(op: pyca.Map) -> pyca.Map:
    return Abs(op.shape)


class Sign(pyca.Map):
    """
    Sign function, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)


def sign(op: pyca.Map) -> pyca.Map:
    return Sign(op.shape)


class Heaviside(pyca.Map):
    r"""
    Heaviside function

    Any heaviside function is not differentiable function, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray, x2=0) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if "heaviside" in dir(xp):
            res = xp.heaviside(arr, x2)
        else:
            res = xp.sign(arr)
            res[res == 0] = x2
            res[res < 0] = 0
        return res
