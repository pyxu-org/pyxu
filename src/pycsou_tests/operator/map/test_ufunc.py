import itertools

import numpy as np
import pytest

import pycsou.operator.map.ufunc as pycmu
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


# Trigonometric Functions
class MixinUFunc(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return dim, dim

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return self._random_array((N_test, dim))


class TestSin(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Sin(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, 2 * np.pi, dim)
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCos(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Cos(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, 2 * np.pi, dim)
        B = np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTan(MixinUFunc):
    r"""
    Tangent function diverges for :math:`\mp(2k+1)k\pi/2`, with
    :math:`k \in \mathbb{N}`. Testing is done on :math`[-3*\pi/2+0.2,
    -\pi/2-0.2] \cup [-\pi/2+0.2, \pi/2-0.2] \cup [\pi/2+0.2,
    3*\pi/2-0.2]`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Tan(dim), _spec[0], _spec[1]

    @pytest.fixture(params=["subdomain-1", "subdomain-2", "subdomain-3"])
    def data_apply(self, request, dim):
        return {
            "subdomain-1": dict(
                in_=dict(arr=np.linspace(-3 * np.pi / 2 + 0.2, -np.pi / 2 - 0.2, dim)),
                out=np.tan(np.linspace(-3 * np.pi / 2 + 0.2, -np.pi / 2 - 0.2, dim)),
            ),
            "subdomain-2": dict(
                in_=dict(arr=np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, dim)),
                out=np.tan(np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, dim)),
            ),
            "subdomain-3": dict(
                in_=dict(arr=np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, dim)),
                out=np.tan(np.linspace(np.pi / 2 + 0.2, 3 * np.pi / 2 - 0.2, dim)),
            ),
        }[request.param]


class TestArcsin(MixinUFunc):
    """
    Inverse sine function defined for :math:`[-1,1]`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arcsin(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-1, 1, dim)
        B = np.arcsin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), -0.9, 0.9)


class TestArccos(MixinUFunc):
    """
    Inverse cosine function defined for :math:`[-1,1]`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arccos(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-1, 1, dim)
        B = np.arccos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), -0.9, 0.9)


class TestArctan(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arctan(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-100, 100, dim)
        B = np.arctan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Hyperbolic Functions


class TestSinh(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Sinh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.sinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCosh(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Cosh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.cosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTanh(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Tanh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.tanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcsinh(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arcsinh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-10, 10, dim)
        B = np.arcsinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArccosh(MixinUFunc):
    r"""
    Inverse hyperbolic cosine function defined for :math:`[1,\infty)`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arccosh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(1, 5, dim)
        B = np.arccosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)) + 3, a_min=1, a_max=4)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)) + 3, a_min=1.1, a_max=4)


class TestArctanh(MixinUFunc):
    """
    Inverse hyperbolic tangent function defined for :math:`(-1,1)`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Arctanh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-1 + 0.01, 1 - 0.01, dim)
        B = np.arctanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), a_min=-1 + 0.01, a_max=1 - 0.01)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.clip(self._random_array((N_test, dim)), a_min=-0.9, a_max=0.9)


# Exponentials and logarithms


class TestExp(MixinUFunc):
    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, base, _spec):
        return pycmu.Exp(dim, base), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, base):
        A = np.linspace(-4, 4, dim)
        if base is not None:
            A *= np.log(base)
        B = np.exp(A)

        return dict(
            in_=dict(arr=np.linspace(-4, 4, dim)),
            out=B,
        )


class TestLog(MixinUFunc):
    r"""
    Natural logarithm function defined for :math:`(0,\infty)`.
    """

    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, base, _spec):
        return pycmu.Log(dim, base), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, base):
        A = np.linspace(0.1, 10, dim)
        B = np.log(A)
        if base is not None:
            B /= np.log(base)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.abs(self._random_array((N_test, dim)))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.abs(self._random_array((N_test, dim)))


# Sums, Products and Differences


class TestCumprod(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Cumprod(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0.1, 10, dim)
        B = np.cumprod(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCumsum(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Cumsum(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-25.0, 125.0, dim)
        B = np.cumsum(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Miscellaneous
class TestClip(MixinUFunc):
    @pytest.fixture(params=([0, None], [None, 1], [0, 1]))
    def lims(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, lims, _spec):
        return pycmu.Clip(dim, a_min=lims[0], a_max=lims[1]), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, lims):
        A = np.linspace(-100, 100, dim)
        B = np.clip(A, a_min=lims[0], a_max=lims[1])
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSqrt(MixinUFunc):
    r"""
    Square root function defined for :math:`[0,\infty)`.
    """

    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Sqrt(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, 100, dim)
        B = np.sqrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return np.abs(self._random_array((N_test, dim)))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return np.abs(self._random_array((N_test, dim)))


class TestCbrt(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Cbrt(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-10, 10, dim)
        B = np.cbrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSquare(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Square(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-2, 2, dim)
        B = np.square(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestAbs(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Abs(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.abs(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSign(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Sign(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-100, 100, dim)
        B = np.sign(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Activation Functions (TestTanh already implemented)


class TestSigmoid(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Sigmoid(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = 1 / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestReLU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.ReLU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = A.clip(min=0)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestGELU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.GELU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        from scipy.special import erf

        A = np.linspace(-4, 4, dim)
        B = A * (1 + erf(A) / np.sqrt(2)) / 2
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSoftplus(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Softplus(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.log(np.exp(A) + 1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestELU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.ELU(dim, alpha=10.0), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.where(A >= 0, A, 10.0 * (np.exp(A) - 1))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSELU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.SELU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = 1.0507 * np.where(A >= 0, A, 1.67326 * (np.exp(A) - 1))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestLeakyReLU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.LeakyReLU(dim, alpha=0.01), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.where(A >= 0, A, A * 0.01)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSiLU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.SiLU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = A / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestGaussian(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Gaussian(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.exp(-(A**2))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestGCU(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.GCU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = A * np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSoftmax(MixinUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pycmu.Softmax(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        exp_A = np.exp(A)
        B = exp_A / np.sum(exp_A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )
