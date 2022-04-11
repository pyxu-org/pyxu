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

