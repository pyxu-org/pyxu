import itertools

import numpy as np
import pytest

import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class MixinUFunc:
    N_test_lipschitz = 10

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
        # override in tests if dom(f) != R
        return self._random_array((self.N_test_lipschitz, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        # override in tests if dom(f) != R
        return self._random_array((self.N_test_lipschitz, dim))


class MixinMapUFunc(MixinUFunc, conftest.MapT):
    pass


class MixinDiffMapUFunc(MixinUFunc, conftest.DiffMapT):
    pass


class MixinSquareOpUFunc(MixinDiffMapUFunc, conftest.SquareOpT):
    pass


# Trigonometric Functions =====================================================
class TestSin(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Sin(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, 2 * np.pi, dim)
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCos(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Cos(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, 2 * np.pi, dim)
        B = np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTan(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Tan(dim), _spec[0], _spec[1]

    @pytest.fixture(params=["subdomain-1", "subdomain-2", "subdomain-3"])
    def data_apply(self, request, dim):
        A = np.linspace(-np.pi, np.pi, dim)
        B = np.tan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcSin(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcSin(dim), _spec[0], _spec[1]

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
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), -1, 1)


class TestArcCos(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcCos(dim), _spec[0], _spec[1]

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
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), -1, 1)


class TestArcTan(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcTan(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-100, 100, dim)
        B = np.arctan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Hyperbolic Functions ========================================================
class TestSinh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Sinh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.sinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCosh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Cosh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.cosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTanh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Tanh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-3, 3, dim)
        B = np.tanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcSinh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcSinh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-10, 10, dim)
        B = np.arcsinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcCosh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcCosh(dim), _spec[0], _spec[1]

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
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), a_min=1, a_max=None)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), a_min=1, a_max=None)


class TestArcTanh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ArcTanh(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-1, 1, dim)
        B = np.arctanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), a_min=-1, a_max=1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.clip(self._random_array((self.N_test_lipschitz, dim)), a_min=-1, a_max=1)


# Exponential Functions =======================================================
class TestExp(MixinDiffMapUFunc):
    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, base, _spec):
        return pyco.Exp(dim, base), _spec[0], _spec[1]

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


class TestLog(MixinDiffMapUFunc):
    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, base, _spec):
        return pyco.Log(dim, base), _spec[0], _spec[1]

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
        return np.abs(self._random_array((self.N_test_lipschitz, dim)))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.abs(self._random_array((self.N_test_lipschitz, dim)))


# Miscellaneous Functions =====================================================
class TestClip(MixinMapUFunc):
    @pytest.fixture(params=([0, None], [None, 1], [0, 1]))
    def lims(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim, lims, _spec):
        return pyco.Clip(dim, a_min=lims[0], a_max=lims[1]), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, lims):
        A = np.linspace(-100, 100, dim)
        B = np.clip(A, a_min=lims[0], a_max=lims[1])
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSqrt(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Sqrt(dim), _spec[0], _spec[1]

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
        return np.abs(self._random_array((self.N_test_lipschitz, dim)))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        return np.abs(self._random_array((self.N_test_lipschitz, dim)))


class TestCbrt(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Cbrt(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-10, 10, dim)
        B = np.cbrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSquare(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Square(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-2, 2, dim)
        B = np.square(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestAbs(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Abs(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.abs(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSign(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Sign(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-100, 100, dim)
        B = np.sign(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCumSum(MixinSquareOpUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.CumSum(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-25.0, 125.0, dim)
        B = np.cumsum(A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Activation Functions ========================================================
class TestGaussian(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Gaussian(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.exp(-(A**2))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSigmoid(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Sigmoid(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = 1 / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSoftPlus(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.SoftPlus(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.log1p(np.exp(A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestLeakyReLU(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.LeakyReLU(dim, alpha=0.01), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = np.where(A >= 0, A, A * 0.01)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestReLU(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.ReLU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = A.clip(min=0)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSiLU(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.SiLU(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        B = A / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSoftmax(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim, _spec):
        return pyco.Softmax(dim), _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(-4, 4, dim)
        exp_A = np.exp(A)
        B = exp_A / np.sum(exp_A, axis=-1)
        return dict(
            in_=dict(arr=A),
            out=B,
        )
