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

