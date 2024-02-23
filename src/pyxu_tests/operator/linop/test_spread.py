import functools
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest
from pyxu_tests.conftest import chunk_array, flaky


class TestUniformSpread(conftest.LinOpT):
    # Idea - test the following setting:
    #     y(z) = \sum_{m=1}^{M} w_{m} \phi(z - x_{m})
    #     x_{m} \in [-1, 1]^{D}
    #     z     \in [-1, 1]^{D}
    #     \phi: truncated N(0, s^{2}) PDF, with \phi(|x| > s) = 0. [seperable]

    # Fixtures ----------------------------------------------------------------
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
        x_spec,
        z_spec,
        kernel_spec,
        spreader_config,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        x_spec = chunk_array(
            xp.array(
                x_spec,
                dtype=width.value,
            ),
            # `x` is not a complex view, but its last axis cannot be chunked.
            # [See UniformSpread() as to why.]
            # We emulate this by setting `complex_view=True`.
            complex_view=True,
        )

        with pxrt.Precision(width):
            op = pxo.UniformSpread(
                x=x_spec,
                z=z_spec,
                kernel=kernel_spec,
                enable_warnings=False,
                **spreader_config,
            )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, x_spec) -> pxt.NDArrayShape:
        M, _ = x_spec.shape
        return (M,)

    @pytest.fixture
    def codim_shape(self, z_spec) -> pxt.NDArrayShape:
        return z_spec["num"]

    @pytest.fixture
    def data_apply(
        self,
        x_spec,
        z_spec,
        kernel_spec,
    ) -> conftest.DataLike:
        # Values are purposefully evaluated the most inefficient way possible. The goal is to be
        # sure about computed outputs, without performing vectorization/etc optimizations.
        M, D = x_spec.shape

        w = self._random_array((M,))

        N, lattice = z_spec["num"], []
        for d in range(D):
            alpha = z_spec["start"][d]
            beta = z_spec["stop"][d]
            _lattice = alpha + (beta - alpha) / (N[d] - 1) * np.arange(N[d])
            lattice.append(_lattice)

        y = np.zeros(N, dtype=w.dtype)
        with np.nditer(y, flags=["multi_index"], op_flags=["writeonly"]) as it:
            for _y in it:
                idx = it.multi_index  # shorthand
                z = np.array([lattice[d][idx[d]] for d in range(D)])  # (D,)

                val = 0
                for m in range(M):
                    basis = 1
                    for d in range(D):
                        func = kernel_spec[d]
                        basis *= func(z[[d]] - x_spec[m, [d]])
                    val += w[m] * basis
                _y[...] = val

        return dict(
            in_=dict(arr=w),
            out=y,
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture(params=[1, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> np.ndarray:
        # (M, D) canonical point cloud [NUMPY]
        # We purposefully choose a range larger than [-1, 1] to ensure out-of-lattice points are handled correctly.
        M = 15
        rng = np.random.default_rng()
        x = rng.uniform(-1.5, 1.5, size=(M, space_dim))
        return x

    @pytest.fixture(params=[10, 15])
    def z_spec(self, space_dim, request) -> dict:
        # canonical lattice spec
        alpha, beta, num = -1, 1, request.param
        return dict(
            start=(alpha,) * space_dim,
            stop=(beta,) * space_dim,
            num=(num,) * space_dim,
        )

    @pytest.fixture
    def kernel_spec(self, space_dim) -> list[callable]:
        # canonical kernel spec.
        # We purposefully do not choose a Map() sub-class to verify it works.
        def gauss(x: pxt.NDArray, sigma: float) -> pxt.NDArray:  # ufunc
            xp = pxu.get_array_module(x)
            y = xp.exp(-((x / sigma) ** 2) / 2)
            y[xp.fabs(x) > sigma] = 0
            return y

        s = 0.15
        func = functools.partial(gauss, sigma=s)
        func.support = lambda: s
        return (func,) * space_dim

    # Keyword parameters to configure the spreader ----------------------------
    @pytest.fixture
    def spreader_config(self, x_spec) -> dict:
        # keyword parameters to configure the spreadder.
        M = len(x_spec)
        kwargs = dict(
            max_window_ratio=3,
            max_cluster_size=M // 2,  # so that we test with multiple sub-grids
        )
        return kwargs

    # Tests -------------------------------------------------------------------
    def test_math_adjoint(self, op, ndi, width):
        flaky(
            func=super().test_math_adjoint,
            args=dict(
                op=op,
                xp=ndi.module(),
                width=width,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="Strict chunk size rules for DASK inputs.",
        )

    def test_math_gram(self, op, ndi, width):
        flaky(
            func=super().test_math_gram,
            args=dict(
                op=op,
                xp=ndi.module(),
                width=width,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="Strict chunk size rules for DASK inputs.",
        )

    def test_math_lipschitz(
        self,
        op,
        ndi,
        width,
        data_math_lipschitz,
        _data_estimate_lipschitz,
    ):
        flaky(
            func=super().test_math_lipschitz,
            args=dict(
                op=op,
                xp=ndi.module(),
                width=width,
                data_math_lipschitz=data_math_lipschitz,
                _data_estimate_lipschitz=_data_estimate_lipschitz,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="Strict chunk size rules for DASK inputs.",
        )

    def test_math2_lipschitz(self, op, _data_estimate_lipschitz, _gpu):
        ndi = pxd.NDArrayInfo.from_obj(op._x)
        flaky(
            func=super().test_math2_lipschitz,
            args=dict(
                op=op,
                _data_estimate_lipschitz=_data_estimate_lipschitz,
                _gpu=_gpu,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="Strict chunk size rules for DASK inputs.",
        )

    def test_math3_lipschitz(self, op, _data_estimate_lipschitz, _op_svd, _gpu):
        ndi = pxd.NDArrayInfo.from_obj(op._x)
        flaky(
            func=super().test_math3_lipschitz,
            args=dict(
                op=op,
                _data_estimate_lipschitz=_data_estimate_lipschitz,
                _op_svd=_op_svd,
                _gpu=_gpu,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="Strict chunk size rules for DASK inputs.",
        )
