# We do not test FSSPulse.[support,supportF]() which are assumed correct from manual tests.

import copy
import itertools

import numpy as np
import pytest
import scipy.special as sp

import pyxu.info.deps as pxd
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class FSSPulseT(conftest.MapT):
    @pytest.fixture(params=[0, 34, 119])
    def _seed(self, request) -> int:
        # Seed used for randomized inputs.
        return request.param

    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return dim, dim

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # pxd.NDArrayInfo.DASK,  # currently unsupported
            ],
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        # override in tests if dom(f) != [-1, 1]
        N_test = 10
        return np.clip(self._random_array((N_test, dim)), -1.1, 1.1)

    # Extra tests/fixtures for applyF() =======================================
    @pytest.fixture
    def data_applyF(self) -> conftest.DataLike:
        # override in subclass with 1D input/outputs of op.applyF().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    @pytest.fixture
    def _data_applyF(self, data_applyF, xp, width) -> conftest.DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.applyF()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_applyF["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
        data = dict(
            in_=in_,
            out=data_applyF["out"],
        )
        return data

    def test_value1D_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_value1D(op.applyF, _data_applyF)

    def test_valueND_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_valueND(op.applyF, _data_applyF)

    def test_backend_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_backend(op.applyF, _data_applyF)

    def test_prec_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_prec(op.applyF, _data_applyF)

    def test_precCM_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_precCM(op.applyF, _data_applyF)

    def test_transparent_applyF(self, op, _data_applyF):
        self._skip_if_disabled()
        self._check_no_side_effect(op.applyF, _data_applyF)


class TestDirac(FSSPulseT):
    @pytest.fixture
    def spec(self, dim, _spec):
        op = pxo.Dirac(dim)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = np.zeros(dim)
        x[1:] = rng.choice([-1, 1], size=dim - 1)
        x[1:] *= rng.uniform(1e-2, 1.1, size=dim - 1)

        y = np.zeros_like(x)
        y[0] = 1
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_applyF(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.ones_like(x)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestBox(FSSPulseT):
    @pytest.fixture
    def spec(self, dim, _spec):
        op = pxo.Box(dim)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.zeros_like(x)
        y[np.fabs(x) <= 1] = 1

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_applyF(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = 2 * np.sinc(2 * x)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestTriangle(FSSPulseT):
    @pytest.fixture
    def spec(self, dim, _spec):
        op = pxo.Triangle(dim)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.clip(1 - np.fabs(x), 0, None)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_applyF(self, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.sinc(x) ** 2

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestTruncatedGaussian(FSSPulseT):
    @pytest.fixture(params=[0.3, 1])
    def gaussian_sigma(self, request) -> float:
        return request.param

    @pytest.fixture
    def spec(self, gaussian_sigma, dim, _spec):
        op = pxo.TruncatedGaussian(dim, sigma=gaussian_sigma)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, gaussian_sigma, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.exp(-0.5 * (x / gaussian_sigma) ** 2)
        y[np.fabs(x) > 1] = 0

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_applyF(self, gaussian_sigma, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.exp(-2 * (np.pi * gaussian_sigma * x) ** 2)
        y *= np.sqrt(2 * np.pi) * gaussian_sigma
        y *= sp.erf((1 / (np.sqrt(2) * gaussian_sigma)) + 1j * (np.sqrt(2) * np.pi * gaussian_sigma * x)).real

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestKaiserBessel(FSSPulseT):
    @pytest.fixture(params=[0.3, 1])
    def kb_beta(self, request) -> float:
        return request.param

    @pytest.fixture
    def spec(self, kb_beta, dim, _spec):
        op = pxo.KaiserBessel(dim, beta=kb_beta)
        return op, _spec[0], _spec[1]

    @pytest.fixture
    def data_apply(self, kb_beta, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        y = np.i0(kb_beta * np.sqrt(np.clip(1 - x**2, 0, None)))
        y /= np.i0(kb_beta)
        y[np.fabs(x) > 1] = 0

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_applyF(self, kb_beta, dim, _seed):
        rng = np.random.default_rng(_seed)

        x = rng.uniform(-1.5, 1.5, dim)
        v = np.emath.sqrt(kb_beta**2 - (2 * np.pi * x) ** 2)
        y = np.real(np.sinh(v) / v)
        y *= 2 / np.i0(kb_beta)

        return dict(
            in_=dict(arr=x),
            out=y,
        )
