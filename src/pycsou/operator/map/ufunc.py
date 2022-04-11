import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.operator.linop.base as pyclb


# Trigonometric Functions

class Sin(pyca.DiffMap):
    r"""
    Sine function

    Any sine function is differentiable function, therefore its base class is DiffMap.
    """

    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sin, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Cos, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arcsin, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arccos, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arctan, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.65

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/(1 + xp.power(arr, 2))))


# Hyperbolic Functions

class Sinh(pyca.DiffMap):
    r"""
    Hyperbolic sine function

    Any hyperbolic sine function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sinh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Cosh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Tanh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = -0.65

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.power(xp.cosh(arr), 2))))

class Arcsinh(pyca.DiffMap):
    r"""
    Inverse hyperbolic sine function

    Any inverse hyperbolic sine function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arcsinh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.39

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.sqrt(xp.power(arr, 2) + 1))))

class Arccosh(pyca.Map):
    r"""
    Inverse hyperbolic cosine function

    Any inverse hyperbolic cosine function is not differentiable function at {-1, +1}, therefore its base class is Map.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arccosh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Arctanh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Exp, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Log, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/arr))

class Log10(pyca.DiffMap):
    r"""
    Base 10 logarithm function

    Any base 10 logarithm function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Log10, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.log(10)*arr)))

class Log2(pyca.DiffMap):
    r"""
    Base 2 logarithm function

    Any base 2 logarithm function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Log10, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.log(2)*arr)))


# Sums and Products

class Prod(pyca.Map):
    r"""
    Product of array elements over a given axis. Its base class is Map.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Prod, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis: int) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the product is performed

        Returns
        -------
        NDArray
            Product of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.prod(arr, axis=axis)

class Sum(pyca.Map):
    r"""
    Sum of array elements over a given axis. Its base class is Map.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sum, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis: int) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the sum is performed

        Returns
        -------
        NDArray
            Sum of elements of arr over a given axis
        """
        xp = pycu.get_array_module(arr)
        return xp.sum(arr, axis=axis)

class Cumprod(pyca.Map):
    r"""
    Cumulative product of elements along a given axis. Its base class is Map.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Cumprod, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis: int) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the cumulative product is performed

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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Cumsum, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis: int) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array
        axis: Int
            Axis which the cumulative sum is performed

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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Clip, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, a_min: float, a_max: float) -> pyct.NDArray:
        r"""

        Parameters
        ----------
        arr: NDArray
            Input array
        a_min: Float
            Minimum value
        a_max: Float
            Maximum value

        Returns
        -------
        NDArray
            Array with elements of arr but where values < a_{min} are replaced with a_{min} and
            those > a_{max} with a_{max}
        """
        return arr.clip(min=a_min, max=a_max)

class Sqrt(pyca.DiffMap):
    r"""
    Nonnegative square-root function

    Any nonnegative sqrt function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sqrt, self).__init__(shape)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i='arr', o=True)
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

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of nonnegative square root of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(2*xp.sqrt(arr))))

class Cbrt(pyca.DiffMap):
    r"""
    Cube-root function

    Any cbrt function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Cbrt, self).__init__(shape)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i='arr', o=True)
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

    def jacobian(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: NDArray
            Input array

        Returns
        -------
            Jacobian matrix of cube root of arr
        """
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(3*xp.power(arr, 2/3))))

class Square(pyca.DiffMap):
    r"""
    Square function

    Any square function is differentiable function, therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sqrt, self).__init__(shape)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i='arr', o=True)
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
        return pyclb.ExplicitLinOp(xp.diag(2*arr))

class Abs(pyca.DiffMap):
    r"""
    Absolute function

    Any absolute function is differentiable function (except just one point), therefore its base class is DiffMap.
    """
    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Abs, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Sign, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
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
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(Heaviside, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr1: pyct.NDArray, arr2: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr1: NDArrray
            Input array
        arr2: NDArray
            Value of the function when arr1 is 0

        Returns
        -------
        NDArray
            Heaviside of each element of arr1
        """
        xp = pycu.get_array_module(arr1)
        return xp.heaviside(arr1, arr2)

