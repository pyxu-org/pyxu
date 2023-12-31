import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestL1Ball(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, radius, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.L1Ball(dim_shape=dim_shape, radius=radius)
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

    @pytest.fixture(params=[1, 1.1])
    def radius(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture
    def data_apply(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.where(
            np.linalg.norm(x.reshape(-1), ord=1) <= radius,
            0,
            np.inf,
        )[np.newaxis]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = x - pxo.LInfinityNorm(dim_shape).prox(x, tau=radius)

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


class TestL2Ball(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, radius, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.L2Ball(dim_shape=dim_shape, radius=radius)
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

    @pytest.fixture(params=[1, 1.1])
    def radius(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture
    def data_apply(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.where(
            np.linalg.norm(x.reshape(-1), ord=2) <= radius,
            0,
            np.inf,
        )[np.newaxis]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = x - pxo.L2Norm(dim_shape).prox(x, tau=radius)

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


class TestLInfinityBall(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, radius, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.LInfinityBall(dim_shape=dim_shape, radius=radius)
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

    @pytest.fixture(params=[1, 1.1])
    def radius(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture
    def data_apply(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.where(
            np.linalg.norm(x.reshape(-1), ord=np.inf) <= radius,
            0,
            np.inf,
        )[np.newaxis]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, dim_shape, radius) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = x - pxo.L1Norm(dim_shape).prox(x, tau=radius)

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


class TestPositiveOrthant(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.PositiveOrthant(dim_shape=dim_shape)
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
            y = np.array([0])

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 17, 93])
    def data_prox(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = np.where(x >= 0, x, 0)

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


class TestHyperSlab(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        from pyxu_tests.operator.examples.test_linfunc import Sum

        ndi, width = request.param
        op = pxo.HyperSlab(
            a=Sum(dim_shape=dim_shape),
            lb=-1,
            ub=2,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        return (1, 2, 1, 1)

    @pytest.fixture(
        params=[
            # provided as dim_shape=(2,); reshape in body to true dim-shape.
            (np.r_[0, 0], np.r_[0]),
            (np.r_[2, 0], np.r_[0]),
            (np.r_[-1, 0], np.r_[0]),
            (np.r_[0, 2], np.r_[0]),
            (np.r_[0, -1], np.r_[0]),
            (np.r_[1.05, 1.05], np.r_[np.inf]),
            (np.r_[-0.5, -0.6], np.r_[np.inf]),
        ]
    )
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        x, out = request.param
        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=out,
        )

    @pytest.fixture(
        params=[
            # provided as dim_shape=(2,); reshape in body to true dim-shape.
            (np.r_[0, 0], np.r_[0, 0]),
            (np.r_[2.1, 3], np.r_[0.55, 1.45]),
            (np.r_[-3, -2], np.r_[-1, 0]),
            (np.r_[1, 0.5], np.r_[1, 0.5]),
        ]
    )
    def data_prox(self, dim_shape, request) -> conftest.DataLike:
        x, y = request.param
        return dict(
            in_=dict(
                arr=x.reshape(dim_shape),
                tau=0.75,  # some random value; doesn't affect prox outcome
            ),
            out=y.reshape(dim_shape),
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x


class TestRangeSet(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, A, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.RangeSet(A)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, A) -> pxt.OpShape:
        return A.codim_shape

    @pytest.fixture(
        params=[
            "tall",
            "square",
            "fat",
        ]
    )
    def A(self, request) -> pxa.LinOp:
        # linear operator describing the span
        variant = request.param
        if variant == "tall":
            from pyxu.operator import BroadcastAxes

            return BroadcastAxes(
                dim_shape=(4, 5),
                codim_shape=(3, 4, 5),
            )
        elif variant == "square":
            from pyxu_tests.operator.examples.test_squareop import CumSum

            return CumSum(dim_shape=(5, 3, 4))
        elif variant == "fat":
            from pyxu_tests.operator.examples.test_linop import Sum

            return Sum(dim_shape=(5, 3, 4))
        else:
            raise NotImplementedError

    @pytest.fixture(
        params=[  # (seed, in_set)
            (1, True),
            (1, False),
            (2, True),
            (2, False),
            (6, True),
            (6, False),
        ]
    )
    def data_apply(self, A, request):
        seed, in_set = request.param

        M, N = A.codim_size, A.dim_size
        B = A.asarray().reshape(M, N)

        u, s, vh = np.linalg.svd(B)
        Q = u[(slice(None), *s.nonzero())]  # orth basis of range(A)
        Qp = np.eye(M) - Q @ Q.T  # orth basis of range(A)^\perp

        # override `in_set` if A is full rank.
        if (s > 0).sum() == min(M, N):
            in_set = True

        if in_set:
            # generate data point in range(A)
            x = (Q @ Q.T) @ self._random_array((M,), seed=seed)
            out = np.r_[0]
        else:
            # generate data point in range(A)^\perp
            seek = True
            while seek:
                x = Qp @ self._random_array((M,), seed=seed)
                seek = np.allclose(x, 0)
            out = np.r_[np.inf]

        return dict(
            in_=dict(arr=x.reshape(A.codim_shape)),
            out=out,
        )

    @pytest.fixture(params=[0, 3, 7])
    def data_prox(self, A, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(A.codim_size, seed=seed)
        B = A.asarray().reshape(A.codim_size, A.dim_size)

        Q, _ = np.linalg.qr(B)
        y = Q @ Q.T @ x

        return dict(
            in_=dict(
                arr=x.reshape(A.codim_shape),
                tau=1,  # some random value; doesn't affect prox outcome
            ),
            out=y.reshape(A.codim_shape),
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        N_test = 10
        return self._random_array((N_test, *dim_shape))
