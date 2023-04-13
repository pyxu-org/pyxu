import itertools

import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest
import pycsou_tests.operator.linop.nufft.conftest as conftest_nufft


class TestNUFFT3(conftest_nufft.NUFFT_Mixin, conftest.LinOpT):
    # (Extra) Fixtures which parametrize operator -----------------------------
    @pytest.fixture(params=[10, 13])
    def transform_x(self, transform_dimension, request) -> np.ndarray:
        # (M, D) D-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{D}`.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(request.param, transform_dimension))
        return x

    @pytest.fixture(params=[1, 22])
    def transform_z(self, transform_dimension, request) -> np.ndarray:
        # (N, D) D-dimensional query points :math:`\mathbf{z}_{k} \in \mathbb{R}^{D}`.
        rng = np.random.default_rng(1)
        x = rng.normal(size=(request.param, transform_dimension))
        return x

    @pytest.fixture
    def _transform_cArray(
        self,
        transform_x,
        transform_z,
        transform_sign,
    ) -> np.ndarray:
        # Ground-truth LinOp A: \bC^{M} -> \bC^{N} which encodes the type-3 transform.
        A = np.exp(1j * transform_sign * transform_z @ transform_x.T)  # (N, M)
        return A

    # Fixtures from conftest.LinOpT -------------------------------------------
    @pytest.fixture(
        params=itertools.product(
            [
                pycd.NDArrayInfo.NUMPY,
                pycd.NDArrayInfo.DASK,
            ],
            pycrt.Width,
        )
    )
    def spec(
        self,
        transform_x,
        transform_z,
        transform_sign,
        transform_eps,
        transform_real,
        transform_ntrans,
        transform_nthreads,
        transform_modeord,
        request,
    ):
        ndi, width = request.param
        with pycrt.Precision(width):
            op = pycl.NUFFT.type3(
                x=transform_x,
                z=transform_z,
                isign=transform_sign,
                eps=transform_eps,
                real=transform_real,
                enable_warnings=False,
                n_trans=transform_ntrans,
                nthreads=transform_nthreads,
                modeord=transform_modeord,
            )
        return op, ndi, width

    @pytest.fixture
    def data_shape(
        self,
        transform_x,
        transform_z,
        transform_real,
    ):
        dim = len(transform_x) * (1 if transform_real else 2)
        codim = 2 * len(transform_z)
        return (codim, dim)

    @pytest.fixture
    def data_apply(
        self,
        _transform_cArray,
        transform_real,
    ):
        N, M = _transform_cArray.shape

        rng = np.random.default_rng(3)
        w = rng.normal(size=(M,)) + 1j * rng.normal(size=(M,))
        if transform_real:
            w = w.real
        v = _transform_cArray @ w

        return dict(
            in_=dict(arr=w if transform_real else pycu.view_as_real(w)),
            out=pycu.view_as_real(v),
        )


class TestNUFFT3_chunked(TestNUFFT3):
    # Fixtures from conftest.LinOpT -------------------------------------------
    @pytest.fixture(
        params=itertools.product(
            [
                pycd.NDArrayInfo.NUMPY,
                pycd.NDArrayInfo.DASK,
            ],
            pycrt.Width,
            [True, False],
        )
    )
    def spec(
        self,
        transform_x,
        transform_z,
        transform_sign,
        transform_eps,
        transform_real,
        transform_ntrans,
        transform_nthreads,
        request,
    ):
        ndi, width, parallel = request.param
        with pycrt.Precision(width):
            op = pycl.NUFFT.type3(
                x=transform_x,
                z=transform_z,
                isign=transform_sign,
                eps=transform_eps,
                real=transform_real,
                n_trans=transform_ntrans,
                nthreads=transform_nthreads,
                chunked=True,
                parallel=parallel,
            )
        return op, ndi, width
