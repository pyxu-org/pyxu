import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class L1Norm(pxa.ProxFunc):
    # f: \bR^{M1,...,MD} -> \bR
    #      x             -> \norm{x}{1}
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.sqrt(self.dim_size)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.sum(
            xp.fabs(arr),
            axis=tuple(range(-self.dim_rank, 0)),
        )[..., np.newaxis]
        return y

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class TestL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = L1Norm(dim_shape)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = np.array([np.fabs(x).sum()])

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 17, 93])
    def data_prox(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        t = np.fabs(x).min() + 1e-2  # some small positive offset

        positive = x >= 0
        y = np.zeros_like(x)
        y[positive] = np.clip(x[positive] - t, 0, None)
        y[~positive] = np.clip(x[~positive] + t, None, 0)

        return dict(
            in_=dict(
                arr=x,
                tau=t,
            ),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x


class TestL1NormMoreau(conftest.DiffFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, mu, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = L1Norm(dim_shape).moreau_envelope(mu)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[1, 2])
    def mu(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture
    def op_orig(self, dim_shape):
        return L1Norm(dim_shape)

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, op_orig, mu, request):
        seed = request.param

        x = self._random_array(op_orig.dim_shape, seed=seed)
        y = op_orig.prox(x, mu)
        z = op_orig.apply(y) + (0.5 / mu) * np.sum((y - x) ** 2)
        return dict(
            in_=dict(arr=x),
            out=z,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture(params=[0, 17, 93])
    def data_grad(self, op_orig, mu, request):
        seed = request.param

        x = self._random_array(op_orig.dim_shape, seed=seed)
        y = (x - op_orig.prox(x, mu)) / mu
        return dict(
            in_=dict(arr=x),
            out=y,
        )
