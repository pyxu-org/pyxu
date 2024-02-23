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


class TestNUFFT1(conftest.LinOpT):
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
        N,
        isign,
        upsampfac,
        T,
        Tc,
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

        with pxrt.Precision(width.real):
            op = pxo.NUFFT1(
                x=x_spec,
                N=N,
                isign=isign,
                eps=1e-9,  # tested manually -> works
                spp=None,  # tested manually -> works
                upsampfac=upsampfac,
                T=T,
                Tc=Tc,
                enable_warnings=False,
            )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, x_spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        return (len(x_spec), 2)

    @pytest.fixture
    def codim_shape(self, N) -> pxt.NDArrayShape:
        return (*(2 * N + 1), 2)

    @pytest.fixture
    def data_apply(
        self,
        x_spec,
        N,
        isign,
        T,
    ) -> conftest.DataLike:
        M = len(x_spec)
        x = self._random_array((M,)) + 1j * self._random_array((M,))

        A = np.stack(  # (L1,...,LD, D)
            np.meshgrid(
                *[np.arange(-n, n + 1) for n in N],
                indexing="ij",
            ),
            axis=-1,
        )
        B = np.exp(  # (L1,...,LD, M)
            (2j * isign * np.pi)
            * np.tensordot(
                A,
                x_spec / T,
                axes=[[-1], [-1]],
            )
        )
        y = np.tensordot(B, x, axes=1)  # (L1,...,LD)

        return dict(
            in_=dict(arr=pxu.view_as_real(x)),
            out=pxu.view_as_real(y),
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture(params=[1, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def N(self, space_dim) -> np.ndarray:
        # guarantees having different modes/dim
        N = 27 + np.arange(space_dim) * 3
        return N

    @pytest.fixture
    def x_spec(self, space_dim, T, Tc) -> np.ndarray:
        # (M, D) canonical point cloud [NUMPY]
        M = 150
        rng = np.random.default_rng()

        x = np.zeros((M, space_dim))
        for d in range(space_dim):
            x[:, d] = rng.uniform(Tc[d] - T[d] / 2, Tc[d] + T[d] / 2, size=M)
        return x

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

    @pytest.fixture
    def T(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        T = rng.uniform(0.1, 10, size=space_dim)
        return T

    @pytest.fixture
    def Tc(self, space_dim) -> np.ndarray:
        rng = np.random.default_rng()
        Tc = rng.standard_normal(size=space_dim)
        return Tc

    # Tests -------------------------------------------------------------------
