import numpy as np
import pytest
from scipy.special import erf

import pycsou.operator.map.ufunc as pycmu
import pycsou_tests.operator.conftest as conftest

# Trigonometric Functions


class TestSin(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Sin(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2 * np.pi, data_shape[0])
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


class TestCos(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Cos(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2 * np.pi, data_shape[0])
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


class TestTan(conftest.DiffMapT):
    r"""
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
        return pycmu.Tan(shape=data_shape)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-3 * np.pi / 2 + 0.2, -np.pi / 2 - 0.2, 5)),
                out=np.tan(np.linspace(-3 * np.pi / 2 + 0.2, -np.pi / 2 - 0.2, 5)),
            ),
            dict(
                in_=dict(arr=np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 5)),
                out=np.tan(np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 5)),
            ),
            dict(
                in_=dict(arr=np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, 5)),
                out=np.tan(np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, 5)),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class TestArcsin(conftest.DiffMapT):
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
        return pycmu.Arcsin(shape=data_shape)

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

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), -0.9, 0.9)


class TestArccos(conftest.DiffMapT):
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
        return pycmu.Arccos(shape=data_shape)

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
        N_test = 15
        return np.clip(self._random_array((N_test, data_shape[0])), -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 15
        return np.clip(self._random_array((N_test, data_shape[0])), -0.9, 0.9)


class TestArctan(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Arctan(shape=data_shape)

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


class TestSinh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Sinh(shape=data_shape)

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


class TestCosh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Cosh(shape=data_shape)

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


class TestTanh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Tanh(shape=data_shape)

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


class TestArcsinh(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Arcsinh(shape=data_shape)

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


class TestArccosh(conftest.DiffMapT):
    r"""
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
        return pycmu.Arccosh(shape=data_shape)

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
        return np.clip(self._random_array((N_test, data_shape[0])) + 3, a_min=1, a_max=4)

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])) + 3, a_min=1.1, a_max=4)


class TestArctanh(conftest.DiffMapT):
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
        return pycmu.Arctanh(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-1 + 0.01, 1 - 0.01, data_shape[0])
        B = np.arctanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), a_min=-1 + 0.01, a_max=1 - 0.01)

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return np.clip(self._random_array((N_test, data_shape[0])), a_min=-0.9, a_max=0.9)


# Exponentials and logarithms


class TestExp(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Exp(shape=data_shape)

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


class TestLog(conftest.DiffMapT):
    r"""
    Natural logarithm function defined for :math:`(0,\infty)`.
    """

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Log(shape=data_shape)

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


# Sums, Products and Differences


class TestProd(conftest.DiffFuncT):
    @pytest.fixture(params=[5, None])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def op(self, dim):
        return pycmu.Prod(dim)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 5)),
                out=np.prod(np.linspace(-1, 3, 5), axis=-1, keepdims=True),
            ),
            dict(
                in_=dict(arr=np.linspace(1, 3, 5)),
                out=np.prod(np.linspace(1, 3, 5), axis=-1, keepdims=True),
            ),
            dict(
                in_=dict(arr=np.array([1, 2, 0, 3, 0])),
                out=np.prod(np.array([1, 2, 0, 3, 0]), axis=-1, keepdims=True),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 6, dim if (dim is not None) else 3
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test, dim = 6, dim if (dim is not None) else 3
        return self._random_array((N_test, dim))

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 5)),
                out=np.array([0.0, -6.0, 0.0, 0.0, 0.0]),
            ),
            dict(
                in_=dict(arr=np.linspace(1, 5, 5)),
                out=np.array([120.0, 60.0, 40.0, 30.0, 24.0]),
            ),
            dict(
                in_=dict(arr=np.array([1, 2, 0, 3, 0])),
                out=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            ),
        ]
    )
    def data_grad(self, request):
        return request.param


class TestSum(conftest.DiffFuncT):
    @pytest.fixture(params=[15, None])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def op(self, dim):
        return pycmu.Sum(dim)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 15)),
                out=np.sum(np.linspace(-1, 3, 15), axis=-1, keepdims=True),
            ),
            dict(
                in_=dict(arr=np.linspace(1, 3, 15)),
                out=np.sum(np.linspace(1, 3, 15), axis=-1, keepdims=True),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 15, dim if (dim is not None) else 15
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test, dim = 15, dim if (dim is not None) else 13
        return self._random_array((N_test, dim))

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 15)),
                out=np.ones(15),
            ),
            dict(
                in_=dict(arr=np.linspace(1, 5, 5)),
                out=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            ),
        ]
    )
    def data_grad(self, request):
        return request.param


class TestCumprod(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Cumprod(shape=data_shape)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 5)),
                out=np.cumprod(np.linspace(-1, 3, 5), axis=-1),
            ),
            dict(
                in_=dict(arr=np.linspace(1, 3, 5)),
                out=np.cumprod(np.linspace(1, 3, 5), axis=-1),
            ),
            dict(
                in_=dict(arr=np.array([1, 2, 0, 3, 0])),
                out=np.cumprod(np.array([1, 2, 0, 3, 0]), axis=-1),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class TestCumsum(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Cumsum(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-25.0, 125.0, data_shape[0])
        B = np.cumsum(A, axis=-1)
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


# Miscellaneous


class TestClip(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Clip(shape=data_shape, a_min=0, a_max=1)

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

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class TestSqrt(conftest.DiffMapT):
    r"""
    Square root function defined for :math:`[0,\infty)`.
    """

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Sqrt(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 100, data_shape[0])
        B = np.sqrt(A)
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


class TestCbrt(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Cbrt(shape=data_shape)

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

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class TestSquare(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Square(shape=data_shape)

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


class TestAbs(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Abs(shape=data_shape)

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


class TestSign(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Sign(shape=data_shape)

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

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


# Activation Functions (TestTanh already implemented)


class TestSigmoid(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Sigmoid(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = 1 / (1 + np.exp(-A))
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


class TestReLU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.ReLU(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = A.clip(min=0)
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


class TestGELU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.GELU(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = A * (1 + erf(A) / np.sqrt(2)) / 2
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


class TestSoftplus(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Softplus(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.log(np.exp(A) + 1)
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


class TestELU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.ELU(shape=data_shape, alpha=10.0)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.where(A >= 0, A, 10.0 * (np.exp(A) - 1))
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


class TestSELU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.SELU(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = 1.0507 * np.where(A >= 0, A, 1.67326 * (np.exp(A) - 1))
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


class TestLeakyReLU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.LeakyReLU(shape=data_shape, alpha=0.01)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.where(A >= 0, A, A * 0.01)
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


class TestSiLU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.SiLU(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = A / (1 + np.exp(-A))
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


class TestGaussian(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Gaussian(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = np.exp(-(A**2))
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


class TestGCU(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.GCU(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        B = A * np.cos(A)
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


class TestSoftmax(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return pycmu.Softmax(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(-4, 4, data_shape[0])
        exp_A = np.exp(A)
        B = exp_A / np.sum(exp_A, axis=-1)
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


class TestMaxout(conftest.DiffFuncT):
    @pytest.fixture(params=[5, None])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def op(self, dim):
        return pycmu.Maxout(dim)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 5)),
                out=np.array([np.max(np.linspace(-1, 3, 5), axis=-1)]),
            ),
            dict(
                in_=dict(arr=np.array([0.1, 0.6, 0.5, 0.3, 0.1])),
                out=np.array([np.max(np.array([0.1, 0.6, 0.5, 0.3, 0.1]), axis=-1)]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 6, dim if (dim is not None) else 3
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test, dim = 6, dim if (dim is not None) else 3
        return self._random_array((N_test, dim))

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.linspace(-1, 3, 5)),
                out=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            ),
            dict(
                in_=dict(arr=np.array([0.1, 0.6, 0.5, 0.3, 0.1])),
                out=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            ),
        ]
    )
    def data_grad(self, request):
        return request.param
