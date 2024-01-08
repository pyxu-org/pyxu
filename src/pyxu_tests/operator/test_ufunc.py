import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class MixinUFunc:
    N_test_lipschitz = 20

    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        # override in tests if dom(f) != R
        return self._random_array((self.N_test_lipschitz, *dim_shape))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        # override in tests if dom(f) != R
        return self._random_array((self.N_test_lipschitz, *dim_shape))


class MixinMapUFunc(MixinUFunc, conftest.MapT):
    pass


class MixinDiffMapUFunc(MixinUFunc, conftest.DiffMapT):
    pass


# Trigonometric Functions =====================================================
class TestSin(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Sin(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = self._random_array(dim_shape)
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCos(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Cos(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = self._random_array(dim_shape)
        B = np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTan(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Tan(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = self._random_array(dim_shape).clip(-np.pi, np.pi)
        B = np.tan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcSin(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcSin(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = self._random_array(dim_shape).clip(-1, 1)
        B = np.arcsin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, -1, 1)


class TestArcCos(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcCos(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = self._random_array(dim_shape).clip(-1, 1)
        B = np.arccos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, -1, 1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, -1, 1)


class TestArcTan(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcTan(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-100, 100, np.prod(dim_shape)).reshape(dim_shape)
        B = np.arctan(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Hyperbolic Functions ========================================================
class TestSinh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Sinh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-3, 3, np.prod(dim_shape)).reshape(dim_shape)
        B = np.sinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestCosh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Cosh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-3, 3, np.prod(dim_shape)).reshape(dim_shape)
        B = np.cosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestTanh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Tanh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-3, 3, np.prod(dim_shape)).reshape(dim_shape)
        B = np.tanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcSinh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcSinh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-10, 10, np.prod(dim_shape)).reshape(dim_shape)
        B = np.arcsinh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestArcCosh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcCosh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(1, 5, np.prod(dim_shape)).reshape(dim_shape)
        B = np.arccosh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, a_min=1, a_max=None)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, a_min=1, a_max=None)


class TestArcTanh(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ArcTanh(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-1, 1, np.prod(dim_shape)).reshape(dim_shape)
        B = np.arctanh(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, a_min=-1, a_max=1)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.clip(x, a_min=-1, a_max=1)


# Exponential Functions =======================================================
class TestExp(MixinDiffMapUFunc):
    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim_shape, base, _spec):
        op = pxo.Exp(dim_shape, base)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape, base):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        if base is not None:
            A *= np.log(base)
        B = np.exp(A)

        return dict(
            in_=dict(arr=np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)),
            out=B,
        )


class TestLog(MixinDiffMapUFunc):
    @pytest.fixture(params=[None, 2, 10])
    def base(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim_shape, base, _spec):
        op = pxo.Log(dim_shape, base)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape, base):
        A = np.linspace(0.1, 10, np.prod(dim_shape)).reshape(dim_shape)
        B = np.log(A)
        if base is not None:
            B /= np.log(base)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.abs(x)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.abs(x)


# Miscellaneous Functions =====================================================
class TestClip(MixinMapUFunc):
    @pytest.fixture(params=([0, None], [None, 1], [0, 1]))
    def lims(self, request):
        return request.param

    @pytest.fixture
    def spec(self, dim_shape, lims, _spec):
        op = pxo.Clip(dim_shape, a_min=lims[0], a_max=lims[1])
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape, lims):
        A = np.linspace(-100, 100, np.prod(dim_shape)).reshape(dim_shape)
        B = np.clip(A, a_min=lims[0], a_max=lims[1])
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSqrt(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Sqrt(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(0, 100, np.prod(dim_shape)).reshape(dim_shape)
        B = np.sqrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.abs(x)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape):
        x = self._random_array((self.N_test_lipschitz, *dim_shape))
        return np.abs(x)


class TestCbrt(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Cbrt(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-10, 10, np.prod(dim_shape)).reshape(dim_shape)
        B = np.cbrt(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSquare(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Square(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-2, 2, np.prod(dim_shape)).reshape(dim_shape)
        B = np.square(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestAbs(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Abs(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = np.abs(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSign(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Sign(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-100, 100, np.prod(dim_shape)).reshape(dim_shape)
        B = np.sign(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Activation Functions ========================================================
class TestGaussian(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Gaussian(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = np.exp(-(A**2))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSigmoid(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Sigmoid(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = 1 / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSoftPlus(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.SoftPlus(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = np.log1p(np.exp(A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestLeakyReLU(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.LeakyReLU(dim_shape, alpha=0.01)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = np.where(A >= 0, A, A * 0.01)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestReLU(MixinMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.ReLU(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = A.clip(min=0)
        return dict(
            in_=dict(arr=A),
            out=B,
        )


class TestSiLU(MixinDiffMapUFunc):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.SiLU(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        A = np.linspace(-4, 4, np.prod(dim_shape)).reshape(dim_shape)
        B = A / (1 + np.exp(-A))
        return dict(
            in_=dict(arr=A),
            out=B,
        )


# Kernels =====================================================================
# We do not test FSSPulse.[support,supportF]() which are assumed correct from manual tests.
class MixinFSSPulse(MixinMapUFunc):
    @pytest.fixture
    def data_math_lipschitz(self, dim_shape, op):
        rng = np.random.default_rng()
        x = rng.uniform(
            -op.support(),
            op.support(),
            size=(self.N_test_lipschitz, *dim_shape),
        )
        return x


class TestDirac(MixinFSSPulse):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Dirac(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = np.zeros(dim_size)
        x[1:] = rng.choice([-1, 1], size=dim_size - 1)
        x[1:] *= rng.uniform(1e-2, 1.1, size=dim_size - 1)

        y = np.zeros_like(x)
        y[0] = 1

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestBox(MixinFSSPulse):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Box(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.zeros_like(x)
        y[np.fabs(x) <= 1] = 1

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestTriangle(MixinFSSPulse):
    @pytest.fixture
    def spec(self, dim_shape, _spec):
        op = pxo.Triangle(dim_shape=dim_shape)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.clip(1 - np.fabs(x), 0, None)

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestTruncatedGaussian(MixinFSSPulse):
    @pytest.fixture(params=[0.3, 1])
    def gaussian_sigma(self, request) -> float:
        return request.param

    @pytest.fixture
    def spec(self, gaussian_sigma, dim_shape, _spec):
        op = pxo.TruncatedGaussian(
            dim_shape=dim_shape,
            sigma=gaussian_sigma,
        )
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, gaussian_sigma, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.exp(-0.5 * (x / gaussian_sigma) ** 2)
        y[np.fabs(x) > 1] = 0

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestKaiserBessel(MixinFSSPulse):
    @pytest.fixture(params=[0.3, 1])
    def kb_beta(self, request) -> float:
        return request.param

    @pytest.fixture
    def spec(self, kb_beta, dim_shape, _spec):
        op = pxo.KaiserBessel(
            dim_shape=dim_shape,
            beta=kb_beta,
        )
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, kb_beta, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.i0(kb_beta * np.sqrt(np.clip(1 - x**2, 0, None)))
        y /= np.i0(kb_beta)
        y[np.fabs(x) > 1] = 0

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )
