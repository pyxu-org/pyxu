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
        return pyclb.ExplicitLinOp(xp.diag(xp.cos(arr)))


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
        return pyclb.ExplicitLinOp(xp.diag(-xp.sin(arr)))


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
        return pyclb.ExplicitLinOp(xp.diag(1 / (1 + arr**2)))


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
        return pyclb.ExplicitLinOp(xp.diag(xp.cosh(arr)))


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
        return pyclb.ExplicitLinOp(xp.diag(xp.sinh(arr)))


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
        return pyclb.ExplicitLinOp(xp.diag(1 - xp.tanh(arr) ** 2))


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
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.sqrt(1 + arr**2))))


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
        return pyclb.ExplicitLinOp(xp.diag(xp.exp(arr)))


class Log(pyca.DiffMap):
    """
    Natural logarithm, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / arr))


class Log10(pyca.DiffMap):
    """
    Base-10 logarithm, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log10(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.log(10) * arr)))


class Log2(pyca.DiffMap):
    """
    Base-2 logarithm, element-wise.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log2(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1 / (xp.log(2) * arr)))


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
    r"""
    Nonnegative square-root function

    The sqrt function is nondifferentiable function at 0, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)


class Cbrt(pyca.Map):
    r"""
    Cube-root function

    The cbrt function is nondifferentiable function at 0, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)


class Square(pyca.DiffMap):
    r"""
    Square function

    Any square function is differentiable function, therefore its base class is DiffMap.
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
        return pyclb.ExplicitLinOp(xp.diag(2 * arr))


class Abs(pyca.DiffMap):
    r"""
    Absolute function

    Any absolute function is differentiable function (except just one point), therefore its base class is DiffMap.
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
        return pyclb.ExplicitLinOp(xp.diag(xp.sign(arr)))


class Sign(pyca.Map):
    r"""
    Sign function

    Any sign function is not differentiable function, therefore its base class is Map.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)


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
