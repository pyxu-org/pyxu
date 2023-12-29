import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.L1Norm(dim_shape=dim_shape)
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


class TestL2Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.L2Norm(dim_shape=dim_shape)
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
        y = np.array([np.sqrt((x**2).sum())])

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 17, 93])
    def data_prox(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        t = np.fabs(x).min() + 1e-2  # some small positive offset

        x_norm = np.linalg.norm(x.reshape(-1), ord=2)
        scale = 1 - (t / max(x_norm, t))
        y = x * scale

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


class TestSquaredL2Norm(conftest.QuadraticFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request):
        ndi, width = request.param
        op = pxo.SquaredL2Norm(dim_shape=dim_shape)
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
        y = np.array([(x**2).sum()])

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 17, 93])
    def data_grad(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = 2 * x

        return dict(
            in_=dict(arr=x),
            out=y,
        )

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


class TestSquaredL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.SquaredL1Norm(dim_shape=dim_shape)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        return (1, 5, 1)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((1, 5, 1))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2).reshape(1, 5, 1)),
                out=np.array([49]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((1, 5, 1)),
                    tau=1,
                ),
                out=np.zeros((1, 5, 1)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2).reshape(1, 5, 1),
                    tau=1,
                ),
                out=np.array([-1, 0, 0, 0, 0]).reshape(1, 5, 1),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x


class TestLInfinityNorm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.LInfinityNorm(dim_shape=dim_shape)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        return (1, 5, 1)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((1, 5, 1))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-5, 0).reshape(1, 5, 1)),
                out=np.array([5]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((1, 5, 1)),
                    tau=1,
                ),
                out=np.zeros((1, 5, 1)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2).reshape((1, 5, 1)),
                    tau=1,
                ),
                out=np.r_[-2, -2, -1, 0, 1].reshape(1, 5, 1),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x


class TestL21Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, l2_axis, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.L21Norm(
            dim_shape=dim_shape,
            l2_axis=l2_axis,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        return (2, 3)

    @pytest.fixture
    def l2_axis(self) -> pxt.NDArrayAxis:
        return (0,)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((2, 3))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(
                    arr=np.array(
                        [
                            [1, 2, -3],
                            [0, -2, -4],
                        ]
                    )
                ),
                out=np.array([6 + 2 * np.sqrt(2)]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((2, 3)),
                    tau=1,
                ),
                out=np.zeros((2, 3)),
            ),
            dict(
                in_=dict(
                    arr=np.array(
                        [
                            [1, 2, -3],
                            [0, -2, -4],
                        ]
                    ),
                    tau=4,
                ),
                out=np.array(
                    [
                        [0, 0, -3 / 5],
                        [0, 0, -4 / 5],
                    ]
                ),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x


class TestPositiveL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.PositiveL1Norm(dim_shape=dim_shape)
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

    @pytest.fixture(params=["negative", "positive-only"])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        mode = request.param

        if mode == "negative":
            x = self._random_array(dim_shape)
            x -= x.min() - 1  # at least 1 negative element guaranteed
        elif mode == "positive-only":
            x = self._random_array(dim_shape)
            x -= x.min() + 1e-3  # all >= 0 guaranteed
        else:
            raise NotImplementedError

        if np.any(x < 0):
            y = np.array([np.inf])
        else:
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
        y[~positive] = 0

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
