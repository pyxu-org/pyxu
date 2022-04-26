import pycsou.abc as pyca
import pycsou.operator.linop.base as pyclb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

# Trigonometric Functions


class Sin(pyca.DiffMap):
    r"""
    Sine function

    Any sine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Sin, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Angle in radians

        Returns
        -------
        NDArray
            Sine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Angle in radians

        Returns
        -------
        NDArray
            Jacobian matrix of sine function of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.cos(arr)))


class Cos(pyca.DiffMap):
    r"""
    Cosine function

    Any cosine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Cos, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Angle in radians

        Returns
        -------
        NDArray
            Cosine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Angle in radians

        Returns
        -------
        NDArray
            Jacobian matrix of cosine function of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(-xp.sin(arr)))


class Tan(pyca.Map):
    r"""
    Tangent function

    Any tangent function is non-differentiable function, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Tan, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Angle, in radians (2\pi rad equals 360 degrees)

        Returns
        -------
        NDArray
            Tangent of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)


class Arcsin(pyca.Map):
    r"""
    Inverse sine function

    Any inverse sine function is not differentiable function at {-1, +1}, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arcsin, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            y-coordinate on the unit circle, defined over [-1,1]

        Returns
        -------
        NDArray
            Inverse sine of each element of arr in radians
        """
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)


class Arccos(pyca.Map):
    r"""
    Inverse cosine function

    Any inverse cosine function is not differentiable function at {-1, +1}, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arccos, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            y-coordinate on the unit circle, defined over [-1,1]

        Returns
        -------
        NDArray
            Inverse cosine of each element of arr in radians
        """
        xp = pycu.get_array_module(arr)
        return xp.arccos(arr)


class Arctan(pyca.DiffMap):
    r"""
    Inverse tangent function

    Inverse tangent function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arctan, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.65

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray

        Returns
        -------
        NDArray
            Inverse tangent of each element of arr in radians
        """
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Jacobian matrix of inverse tangent of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (1 + xp.power(arr, 2))))


# Hyperbolic Functions


class Sinh(pyca.DiffMap):
    r"""
    Hyperbolic sine function

    Any hyperbolic sine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Sinh, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Hyperbolic sine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.sinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Jacobian matrix of hyperbolic sine of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.cosh(arr)))


class Cosh(pyca.DiffMap):
    r"""
    Hyperbolic cosine function

    Any hyperbolic cosine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Cosh, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Hyperbolic cosine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.cosh(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Jacobian matrix of hyperbolic cosine of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.sinh(arr)))


class Tanh(pyca.DiffMap):
    r"""
    Hyperbolic tangent function

    Any hyperbolic tangent function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Tanh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.77

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Hyperbolic tangent of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.tanh(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Jacobian matrix of hyperbolic tangent of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.power(xp.cosh(arr), 2))))


class Arcsinh(pyca.DiffMap):
    r"""
    Inverse hyperbolic sine function

    Any inverse hyperbolic sine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arcsinh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.39

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Inverse hyperbolic sine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.arcsinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Jacobian matrix of inverse hyperbolic sine of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.sqrt(xp.power(arr, 2) + 1))))


class Arccosh(pyca.Map):
    r"""
    Inverse hyperbolic cosine function

    Any inverse hyperbolic cosine function is not differentiable function at {-1, +1}, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arccosh, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array defined on positive real numbers

        Returns
        -------
        NDArray
            Inverse hyperbolic cosine of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)


class Arctanh(pyca.Map):
    r"""
    Inverse hyperbolic tangent function

    Any inverse hyperbolic tangent function is not differentiable function at {-1, +1}, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Arctanh, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array defined on [-1,1]

        Returns
        -------
        NDArray
            Inverse hyperbolic tangent of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)


# Exponentials and logarithms


class Exp(pyca.DiffMap):
    r"""
    Exponential function

    Any exponential function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Exp, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Exponential of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.exp(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of exponential of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.exp(arr)))


class Log(pyca.DiffMap):
    r"""
    Natural logarithm function

    Any natural logarithm function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Log, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Natural logarithm of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.log(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of natural logarithm of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / arr))


class Log10(pyca.DiffMap):
    r"""
    Base 10 logarithm function

    Any base 10 logarithm function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Log10, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Base 10 logarithm of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.log10(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of base 10 logarithm of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.log(10) * arr)))


class Log2(pyca.DiffMap):
    r"""
    Base 2 logarithm function

    Any base 2 logarithm function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Log2, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Base 2 logarithm of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.log2(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of base 2 logarithm of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.log(2) * arr)))


# Sums and Products


class Prod(pyca.Map):
    r"""
    Product of array elements over a given axis. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Prod, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the product is performed. Default is -1.

        Returns
        -------
        NDArray
            Product of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.array([xp.prod(arr, axis=axis)])


class Sum(pyca.Map):
    r"""
    Sum of array elements over a given axis. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Sum, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the sum is performed. Default is -1.

        Returns
        -------
        NDArray
            Sum of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.array([xp.sum(arr, axis=axis)])


class Cumprod(pyca.Map):
    r"""
    Cumulative product of elements along a given axis. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Cumprod, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the cumulative product is performed. Default is -1.

        Returns
        -------
        NDArray
            Cumulative product of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.cumprod(arr, axis=axis)


class Cumsum(pyca.Map):
    r"""
    Cumulative sum of elements along a given axis. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Cumsum, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the cumulative sum is performed. Default is -1.

        Returns
        -------
        NDArray
            Cumulative sum of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.cumsum(arr, axis=axis)


# Miscellaneous


class Clip(pyca.Map):
    r"""
    Clip (limit) the values in an array. Its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Clip, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
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
    r"""
    Nonnegative square-root function

    The sqrt function is nondifferentiable function at 0, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Sqrt, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array defined on nonnegative real numbers

        Returns
        -------
        NDArray
            Nonnegative square root of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)


class Cbrt(pyca.Map):
    r"""
    Cube-root function

    The cbrt function is nondifferentiable function at 0, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Cbrt, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Cube root of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)


class Square(pyca.DiffMap):
    r"""
    Square function

    Any square function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Square, self).__init__(shape)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Square of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.square(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of square of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(2 * arr))


class Abs(pyca.DiffMap):
    r"""
    Absolute function

    Any absolute function is differentiable function (except just one point), therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        super(Abs, self).__init__(shape)
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Absolute of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.absolute(arr)

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of absolute of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.sign(arr)))


class Sign(pyca.Map):
    r"""
    Sign function

    Any sign function is not differentiable function, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Sign, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
        NDArray
            Sign of each element of arr
        """
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)


class Heaviside(pyca.Map):
    r"""
    Heaviside function

    Any heaviside function is not differentiable function, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super(Heaviside, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray, x2=0) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        x2: Int
            Value of function when arr is 0

        Returns
        -------
        NDArray
            Heaviside of each element of arr
        """
        xp = pycu.get_array_module(arr)
        if "heaviside" in dir(xp):
            res = xp.heaviside(arr, x2)
        else:
            res = xp.sign(arr)
            res[res == 0] = x2
            res[res < 0] = 0
        return res
