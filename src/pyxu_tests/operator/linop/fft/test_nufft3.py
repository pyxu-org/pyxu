import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest


class TestNUFFT3(conftest.LinOpT):
    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # NUFFT is an approximate transform.
        # Based on [FINUFFT], results hold up to a small relative error.
        #
        # We choose a conservative threshold, irrespective of the `eps` parameter chosen by the
        # user. Additional tests below test explicitly if computed values correctly obey `eps`.
        eps_default = 1e-3

        cast = lambda x: pxu.compute(x)
        lhs = np.linalg.norm(pxu.to_NUMPY(cast(a) - cast(b)).ravel())
        rhs = np.linalg.norm(pxu.to_NUMPY(cast(b)).ravel())
        return ct.less_equal(lhs, eps_default * rhs, as_dtype=as_dtype).all()

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.DASK,
            ],
            pxrt.CWidth,
        )
    )
    def spec(
        self,
        x_spec,
        v_spec,
        isign,
        upsampfac,
        chunked,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        x_spec = ct.chunk_array(
            xp.array(
                x_spec,
                dtype=width.real.value,
            ),
            # `x` is not a complex view, but its last axis cannot be chunked.
            # [See UniformSpread() as to why.]
            # We emulate this by setting `complex_view=True`.
            complex_view=True,
        )
        v_spec = ct.chunk_array(
            xp.array(
                v_spec,
                dtype=width.real.value,
            ),
            # `v` is not a complex view, but its last axis cannot be chunked.
            # [See UniformSpread() as to why.]
            # We emulate this by setting `complex_view=True`.
            complex_view=True,
        )

        with pxrt.Precision(width.real):
            op = pxo.NUFFT3(
                x=x_spec,
                v=v_spec,
                isign=isign,
                eps=1e-9,  # tested manually -> works
                spp=None,  # tested manually -> works
                upsampfac=upsampfac,
                chunked=chunked,
                domain="xv",  # if 'xv' works, then so should ('x', 'v')
                max_fft_mem=1e-3,  # to force chunking when enabled
                enable_warnings=False,
            )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, x_spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        return (len(x_spec), 2)

    @pytest.fixture
    def codim_shape(self, v_spec) -> pxt.NDArrayShape:
        return (len(v_spec), 2)

    @pytest.fixture
    def data_apply(
        self,
        x_spec,
        v_spec,
        isign,
    ) -> conftest.DataLike:
        M = len(x_spec)
        w = self._random_array((M,)) + 1j * self._random_array((M,))

        A = np.exp((isign * 2j * np.pi) * (v_spec @ x_spec.T))  # (N, M)
        z = A @ w  # (N,)

        return dict(
            in_=dict(arr=pxu.view_as_real(w)),
            out=pxu.view_as_real(z),
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture(params=[1, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> np.ndarray:
        # (M, D) canonical point cloud [NUMPY]
        M = 150
        rng = np.random.default_rng()

        x = rng.uniform(-1, 1, size=(M, space_dim))
        return x

    @pytest.fixture
    def v_spec(self, space_dim) -> np.ndarray:
        # (N, D) canonical point cloud [NUMPY]
        N = 151
        rng = np.random.default_rng()

        v = rng.uniform(-1, 1, size=(N, space_dim))
        return v

    @pytest.fixture(params=[1, -1])
    def isign(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1.25, 2])
    def upsampfac(self, space_dim, request) -> np.ndarray:
        # We force upsampfac to be slightly different per dimension
        base = request.param

        rng = np.random.default_rng()
        extra = rng.uniform(0.1, 0.5, size=space_dim)

        upsampfac = base + extra
        return upsampfac

    @pytest.fixture(params=[True, False])
    def chunked(self, request) -> bool:
        # Perform chunked evaluation
        return request.param

    # Tests -------------------------------------------------------------------
