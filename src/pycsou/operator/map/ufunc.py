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
    return Sin(op.shape) * op


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
    return Cos(op.shape) * op


class Tan(pyca.DiffMap):
    """
    Trigonometric tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.cos(arr) ** 2)


def tan(op: pyca.Map) -> pyca.Map:
    return Tan(op.shape) * op


class Arcsin(pyca.DiffMap):
    """
    Inverse sine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.sqrt(1 - arr**2))


def arcsin(op: pyca.Map) -> pyca.Map:
    return Arcsin(op.shape) * op


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

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(-1 / xp.sqrt(1 - arr**2))


def arccos(op: pyca.Map) -> pyca.Map:
    return Arccos(op.shape) * op


class Arctan(pyca.DiffMap):
    """
    Inverse tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 3 * np.sqrt(3) / 8  # Max of |arctan''(x)|=|2x/(1+x^2)^2| is 3sqrt(3)/8 at x=+-1/sqrt(3)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / (1 + arr**2))


def arctan(op: pyca.Map) -> pyca.Map:
    return Arctan(op.shape) * op


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
    return Sinh(op.shape) * op


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
    return Cosh(op.shape) * op


class Tanh(pyca.DiffMap):
    """
    Hyperbolic tangent, element-wise.
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

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.cosh(arr) ** 2)


def tanh(op: pyca.Map) -> pyca.Map:
    return Tanh(op.shape) * op


class Arcsinh(pyca.DiffMap):
    """
    Inverse hyperbolic sine, element-wise.
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

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / (xp.sqrt(1 + arr**2)))


def arcsinh(op: pyca.Map) -> pyca.Map:
    return Arcsinh(op.shape) * op


class Arccosh(pyca.DiffMap):
    """
    Inverse hyperbolic cosine, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.DiagonalOp(1 / xp.sqrt(-1 + arr**2))


def arccosh(op: pyca.Map) -> pyca.Map:
    return Arccosh(op.shape) * op


class Arctanh(pyca.DiffMap):
    """
    Inverse hyperbolic tangent, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)

    def jacobian(self, arr: pyct.NDArray):
        return pyclb.DiagonalOp(1 / (1 - arr**2))


def arctanh(op: pyca.Map) -> pyca.Map:
    return Arctanh(op.shape) * op


# Exponentials and logarithms


class Exp(pyca.DiffMap):
    """
    Exponential function, element-wise. (Default: base-E exponential.)
    """

    def __init__(self, shape: pyct.Shape, base: pyct.Real = None):
        super().__init__(shape)
        self._base = base

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = arr.copy()
        if self._base is not None:
            out *= np.log(self._base)
        return xp.exp(out)

    def jacobian(self, arr: pyct.NDArray):
        y = self.apply(arr)
        if self._base is not None:
            y *= np.log(self._base)
        return pyclb.DiagonalOp(y)


def exp(op: pyca.Map, base: pyct.Real = None) -> pyca.Map:
    return Exp(op.shape, base) * op


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
        y = 1 / arr
        if self._base is not None:
            y /= np.log(self._base)
        return pyclb.DiagonalOp(y)


def log(op: pyca.Map, base: pyct.Real = None) -> pyca.Map:
    return Log(op.shape, base) * op


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

    def __init__(self, shape: pyct.Shape, axis: pyct.Real = -1):
        super().__init__(shape)
        self._axis = axis

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumprod(arr, axis=self._axis)

    def jacobian(self, arr: pyct.NDArray):
        assert arr.ndim == 1 and (self._axis == -1 or self._axis == 0)  # Jacobian matrix is only valid for vectors
        xp = pycu.get_array_module(arr)
        temp = xp.expand_dims(self.apply(), axis=0)
        num_mtx = temp.transpose() * np.tri(self.shape[0])
        denum_mtx = xp.tile(arr, (self.shape[0], 1))
        return pyclb.ExplicitLinOp(num_mtx / denum_mtx)


def cumprod(op: pyca.Map, axis: pyct.Real = -1) -> pyca.Map:
    return Cumprod(op.shape, axis) * op


class Cumsum(pyca.DiffMap):
    """
    Cumulative sum of elements.
    """

    def __init__(self, shape: pyct.Shape, axis: pyct.Real = -1):
        super().__init__(shape)
        self._axis = axis

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumsum(arr, axis=self._axis)

    def jacobian(self, arr: pyct.NDArray):
        assert arr.ndim == 1 and (self._axis == -1 or self._axis == 0)  # Jacobian matrix is only valid for vectors
        return pyclb.ExplicitLinOp(np.tri(self.shape[0]))


def cumsum(op: pyca.Map, axis: pyct.Real = -1) -> pyca.Map:
    return Cumsum(op.shape, axis) * op


# Miscellaneous


class Clip(pyca.Map):
    """
    Clip (limit) values in an array, element-wise.
    """

    def __init__(self, shape: pyct.Shape, a_min: pyct.Real = None, a_max: pyct.Real = None):
        super().__init__(shape)
        if (a_min is None) and (a_max is None):
            raise ValueError("One of Parameter[a_min, a_max] must be specified.")
        else:
            self._llim = a_min
            self._ulim = a_max

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.clip(self._llim, self._ulim)


def clip(op: pyca.Map, a_min: pyct.Real = None, a_max: pyct.Real = None):
    return Clip(op.shape, a_min, a_max) * op


class Sqrt(pyca.DiffMap):
    """
    Non-negative square-root, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)

    def jacobian(self, arr: pyct.NDArray):
        return pyclb.DiagonalOp(1 / (2 * self.apply(arr)))


def sqrt(op: pyca.Map) -> pyca.Map:
    return Sqrt(op.shape) * op


class Cbrt(pyca.DiffMap):
    """
    Cube-root, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)

    def jacobian(self, arr: pyct.NDArray):
        return pyclb.DiagonalOp(1 / (3 * (arr ** (2 / 3))))


def cbrt(op: pyca.Map) -> pyca.Map:
    return Cbrt(op.shape) * op


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
        return pyclb.DiagonalOp(2 * arr)


def square(op: pyca.Map) -> pyca.Map:
    return Square(op.shape) * op


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
    return Abs(op.shape) * op


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
    return Sign(op.shape) * op
