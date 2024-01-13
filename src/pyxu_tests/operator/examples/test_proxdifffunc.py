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


class SquaredL2Norm(pxa.ProxDiffFunc):
    # f: \bR^{M1,...,MD} -> \bR
    #      x             -> \norm{x}{2}^{2}
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.inf
        self.diff_lipschitz = 2

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        axis = tuple(range(-self.dim_rank, 0))
        y = xp.sum(arr**2, axis=axis)[..., np.newaxis]
        return y

    @pxrt.enforce_precision(i="arr")
    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        return 2 * arr

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y


class TestSquaredL2Norm(conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = SquaredL2Norm(dim_shape)
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
        y = np.array([np.sum(x**2)])

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture(params=[0, 17, 93])
    def data_grad(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = 2 * x

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture(params=[0, 17, 93])
    def data_prox(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = x / (2 * t + 1)

        return dict(
            in_=dict(
                arr=x,
                tau=t,
            ),
            out=y,
        )
