import itertools

import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest
import pycsou_tests.operator.linop.nufft.conftest as conftest_nufft


class TestNUFFT2(conftest_nufft.NUFFT_Mixin, conftest.LinOpT):
    # (Extra) Fixtures which parametrize operator -----------------------------
    @pytest.fixture(params=[1, 13])
    def transform_x(self, transform_dimension, request) -> np.ndarray:
        # (M, D) D-dimensional sample points :math:`\mathbf{x}_{j} \in [-\pi, \pi)^{D}`.
        rng = np.random.default_rng(0)
        _x = rng.normal(size=(request.param, transform_dimension))
        x = np.fmod(_x, 2 * np.pi)
        return x

    @pytest.fixture(params=[2, 10])
    def transform_N(self, transform_dimension, request) -> tuple[int]:
        # (D,) mesh size in each dimension :math:`(N_1, \ldots, N_{D})`.
        rng = np.random.default_rng(1)
        N = rng.integers(
            low=1,
            high=request.param,
            size=(transform_dimension,),
            endpoint=True,
        )
        return tuple(N)

    @pytest.fixture
    def _transform_cArray(
        self,
        transform_x,
        transform_N,
        transform_dimension,
        transform_sign,
    ) -> np.ndarray:
        # Ground-truth LinOp A: \bC^{N.prod()} -> \bC^{M} which encodes the type-2 transform.
        mesh = np.meshgrid(
            *[np.arange(-(n // 2), (n - 1) // 2 + 1) for n in transform_N],
            indexing="ij",
        )
        mesh = np.stack(mesh, axis=0).reshape((transform_dimension, -1)).T
        A = np.exp(1j * np.sign(transform_sign) * mesh @ transform_x.T).T  # (M, N.prod())
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
        transform_N,
        transform_sign,
        transform_eps,
        transform_real,
        transform_ntrans,
        transform_nthreads,
        request,
    ):
        ndi, width = request.param
        with pycrt.Precision(width):
            op = pycl.NUFFT.type2(
                x=transform_x,
                N=transform_N,
                isign=transform_sign,
                eps=transform_eps,
                real=transform_real,
                n_trans=transform_ntrans,
                nthreads=transform_nthreads,
            )
        return op, ndi, width

    @pytest.fixture
    def data_shape(
        self,
        transform_x,
        transform_N,
        transform_real,
    ):
        dim = np.prod(transform_N) * (1 if transform_real else 2)
        codim = 2 * len(transform_x)
        return (codim, dim)

    @pytest.fixture
    def data_apply(
        self,
        _transform_cArray,
        transform_real,
    ):
        M, N = _transform_cArray.shape

        rng = np.random.default_rng(3)
        w = rng.normal(size=(N,)) + 1j * rng.normal(size=(N,))
        if transform_real:
            w = w.real
        v = _transform_cArray @ w

        return dict(
            in_=dict(arr=w if transform_real else pycu.view_as_real(w)),
            out=pycu.view_as_real(v),
        )
