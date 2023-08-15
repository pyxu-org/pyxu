import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest
import pyxu_tests.operator.linop.fft.conftest_nufft as conftest_nufft


class TestNUFFT3(conftest_nufft.NUFFT_Mixin, conftest.LinOpT):
    # (Extra) Fixtures which parametrize operator -----------------------------
    @pytest.fixture(params=[10, 13])
    def transform_x(self, transform_dimension, request) -> np.ndarray:
        # (M, D) D-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{D}`.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(request.param, transform_dimension))
        return x

    @pytest.fixture(params=[11, 22])
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
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.DASK,
            ],
            pxrt.Width,
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
        xp, dtype = ndi.module(), width.value
        with pxrt.Precision(width):
            op = pxl.NUFFT.type3(
                x=xp.array(transform_x, dtype=dtype),
                z=xp.array(transform_z, dtype=dtype),
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
            in_=dict(arr=w if transform_real else pxu.view_as_real(w)),
            out=pxu.view_as_real(v),
        )


class TestNUFFT3Chunked(TestNUFFT3):
    # (Extra) Fixtures which parametrize operator -----------------------------
    @pytest.fixture(params=[True, False])
    def transform_parallel(self, request) -> bool:
        return request.param

    # Fixtures from conftest.LinOpT -------------------------------------------
    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.DASK,
            ],
            pxrt.Width,
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
        transform_parallel,
        request,
    ):
        ndi, width = request.param
        xp, dtype = ndi.module(), width.value
        with pxrt.Precision(width):
            op = pxl.NUFFT.type3(
                x=xp.array(transform_x, dtype=dtype),
                z=xp.array(transform_z, dtype=dtype),
                isign=transform_sign,
                eps=transform_eps,
                real=transform_real,
                enable_warnings=False,
                n_trans=transform_ntrans,
                nthreads=transform_nthreads,
                modeord=transform_modeord,
                chunked=True,
                parallel=transform_parallel,
            )

            # Extra initialization steps for chunked transforms
            x_chunks, z_chunks = op.auto_chunk()
            op.allocate(x_chunks, z_chunks)
        return op, ndi, width
