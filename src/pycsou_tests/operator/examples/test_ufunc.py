import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.operator.linop.base as pyclb
import pycsou_tests.operator.conftest as conftest


# Trigonometric Functions

class Sin(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Sin, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.cos(arr)))

class TestSin(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Sin(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2*np.pi, data_shape[0])
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Cos(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Cos, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(-xp.sin(arr)))

class TestCos(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cos(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2*np.pi, data_shape[0])
        B = np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Tan(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Tan, self).__init__(shape)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tan(arr)

class TestTan(conftest.MapT):
    """
    Tangent function diverges for :math:`\mp(2k+1)k\pi/2`, with
    :math:`k \in \mathbb{N}`. Testing is done on :math`[-3*\pi/2+0.2,
    -\pi/2-0.2] \cup [-\pi/2+0.2, \pi/2-0.2] \cup [\pi/2+0.2,
    3*\pi/2-0.2]`.
    """
    @pytest.fixture
    def dim(self):
        return 7

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Tan(shape=data_shape)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-3*np.pi / 2 + 0.2, -np.pi / 2 - 0.2, 5)),
                out=np.tan(np.linspace(-3*np.pi / 2 + 0.2, -np.pi / 2 - 0.2, 5))
            ),
            dict(
                in_=dict(arr=np.linspace(-np.pi/2+0.1, np.pi/2-0.1, 5)),
                out=np.tan(np.linspace(-np.pi/2+0.1, np.pi/2-0.1, 5))
            ),
            dict(
                in_=dict(arr=np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, 5)),
                out=np.tan(np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, 5))
            )
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Arcsin(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Arcsin, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsin(arr)

class TestArcsin(conftest.MapT):
    """
    Inverse sine function defined for :math:`[-1,1]`.
    """
    @pytest.fixture
    def dim(self):
        return 7

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arcsin(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-1, 1, data_shape[0])
        B = np.arcsin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), -1, 1)


class Arccos(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Arccos, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccos(arr)

class TestArccos(conftest.MapT):
    """
    Inverse cosine function defined for :math:`[-1,1]`.
    """
    @pytest.fixture
    def dim(self):
        return 7

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arccos(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-1, 1, data_shape[0])
        B = np.arccos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), -1, 1)


class Arctan(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Arctan, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.65

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctan(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(1 + xp.power(arr, 2))))

class TestArctan(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arctan(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-100, 100, data_shape[0])
        B = np.arctan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


# Hyperbolic Functions

class Sinh(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Sinh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.cosh(arr)))

class TestSinh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Sinh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-3, 3, data_shape[0])
        B = np.sinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Cosh(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Cosh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cosh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.sinh(arr)))

class TestCosh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cosh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-3, 3, data_shape[0])
        B = np.cosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Tanh(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Tanh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.77

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.tanh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.power(xp.cosh(arr), 2))))
        
class TestTanh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Tanh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-3, 3, data_shape[0])
        B = np.tanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Arcsinh(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Arcsinh, self).__init__(shape)
        self._lipschitz = 1
        self._diff_lipschitz = 0.39

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arcsinh(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.sqrt(xp.power(arr, 2) + 1))))

class TestArcsinh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arcsinh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-10, 10, data_shape[0])
        B = np.arcsinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Arccosh(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Arccosh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arccosh(arr)

class TestArccosh(conftest.MapT):
    """
    Inverse hyperbolic cosine function defined for :math:`[1,\infty)`.
    """
    @pytest.fixture
    def dim(self):
        return 7

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arccosh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(1, 5, data_shape[0])
        B = np.arccosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0]))+3, a_min=1, a_max=4)


class Arctanh(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Arctanh, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.arctanh(arr)

class TestArctanh(conftest.MapT):
    """
    Inverse hyperbolic tangent function defined for :math:`(-1,1)`.
    """
    @pytest.fixture
    def dim(self):
        return 7

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Arctanh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-1+0.01, 1-0.01, data_shape[0])
        B = np.arctanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), a_min=-1+0.01, a_max=1-0.01)


# Exponentials and logarithms

class Exp(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Exp, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.exp(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.exp(arr)))

class TestExp(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Exp(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.exp(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Log(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Log, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/arr))

class TestLog(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Log(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0.1, 10, data_shape[0])
        print("A: ", A)
        B = np.log(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))


class Log10(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Log10, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log10(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.log(10)*arr)))

class TestLog10(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Log10(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0.1, 10, data_shape[0])
        B = np.log10(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))


class Log2(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Log2, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.log2(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(1/(xp.log(2)*arr)))

class TestLog2(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Log2(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0.1, 10, data_shape[0])
        B = np.log2(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))


# Sums, Products and Differences

class Prod(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Prod, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.array([xp.prod(arr, axis=axis)])

class TestProd(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim,0)

    @pytest.fixture
    def op(self, data_shape):
        return Prod(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(1, 5, data_shape[0])
        B = np.prod(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Sum(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Sum, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.array([xp.sum(arr, axis=axis)])

class TestSum(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def data_shape(self, dim):
        return (dim,0)

    @pytest.fixture
    def op(self, data_shape):
        return Sum(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(5, 10, data_shape[0])
        B = np.sum(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Cumprod(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Cumprod, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumprod(arr, axis=axis)

class TestCumprod(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cumprod(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(1.0, 5.0, data_shape[0])
        B = np.cumprod(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Cumsum(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Cumsum, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, axis=-1) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cumsum(arr, axis=axis)

class TestCumsum(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cumsum(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(1.0, 5.0, data_shape[0])
        B = np.cumsum(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


# Miscellaneous

class Clip(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Clip, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, a_min=0.0, a_max=1.0) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.clip(arr, a_min, a_max)

class TestClip(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Clip(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-100, 100, data_shape[0])
        B = np.clip(A, a_min=0.0, a_max=1.0)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Sqrt(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Sqrt, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sqrt(arr)

class TestSqrt(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Sqrt(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0.1, 100, data_shape[0])
        B = np.sqrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.abs(self._random_array((N_test, data_shape[0])))


class Cbrt(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Cbrt, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cbrt(arr)

class TestCbrt(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cbrt(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-10, 10, data_shape[0])
        B = np.cbrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


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
        super(Square, self).__init__(shape)
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.square(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(2*arr))

class TestSquare(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Square(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-2, 2, data_shape[0])
        B = np.square(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Abs(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Abs, self).__init__(shape)
        self._lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.absolute(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.sign(arr)))

class TestAbs(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Abs(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.abs(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        temp = self._random_array((N_test, data_shape[0]))
        return temp

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Sign(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Sign, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sign(arr)

class TestSign(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Sign(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-100, 100, data_shape[0])
        B = np.sign(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Heaviside(pyca.Map):
    def __init__(self, shape: pyct.Shape):
        super(Heaviside, self).__init__(shape)

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray, x2=0) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if 'heaviside' in dir(xp):
            res = xp.heaviside(arr, x2)
        else:
            res = xp.sign(arr)
            res[res == 0] = x2
            res[res < 0] = 0
        return res

class TestHeaviside(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 10000

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Heaviside(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-100, 100, data_shape[0])
        B = np.heaviside(A, 0)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))
