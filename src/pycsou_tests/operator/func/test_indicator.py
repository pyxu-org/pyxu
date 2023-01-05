import itertools

import numpy as np
import pytest

import pycsou.operator.func as pycof
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class TestL1Ball(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (5, pycof.L1Ball(dim=5, radius=1)),
                (None, pycof.L1Ball(dim=None, radius=1)),
            ),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([np.inf]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.zeros((5,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.array([-1, 0, 0, 0, 0]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestL2Ball(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (5, pycof.L2Ball(dim=5, radius=1)),
                (None, pycof.L2Ball(dim=None, radius=1)),
            ),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([np.inf]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.zeros((5,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.arange(-3, 2) / np.linalg.norm(np.arange(-3, 2)),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestLInfinityBall(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (5, pycof.LInfinityBall(dim=5, radius=1)),
                (None, pycof.LInfinityBall(dim=None, radius=1)),
            ),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([np.inf]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.zeros((5,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.array([-1, -1, -1, 0, 1]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestPositiveOrthant(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (5, pycof.PositiveOrthant(dim=5)),
                (None, pycof.PositiveOrthant(dim=None)),
            ),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([np.inf]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.zeros((5,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.array([0, 0, 0, 0, 1]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestHyperSlab(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, request) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        if (xp := ndi.module()) is None:
            pytest.skip(f"{ndi} unsupported on this machine.")

        v = xp.array([1, 1], dtype=width.value)
        a = pyca.LinFunc.from_array(v, enable_warnings=False)
        op = pycof.HyperSlab(a, l=-1, u=2)

        return op, ndi, width

    @pytest.fixture
    def data_shape(self) -> pyct.OpShape:
        return (1, 2)

    @pytest.fixture(
        params=[
            (np.r_[0, 0], np.r_[0]),
            (np.r_[2, 0], np.r_[0]),
            (np.r_[-1, 0], np.r_[0]),
            (np.r_[0, 2], np.r_[0]),
            (np.r_[0, -1], np.r_[0]),
            (np.r_[1.05, 1.05], np.r_[np.inf]),
            (np.r_[-0.5, -0.6], np.r_[np.inf]),
        ]
    )
    def data_apply(self, request) -> conftest.DataLike:
        arr, out = request.param
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture(
        params=[
            (np.r_[0, 0], np.r_[0, 0]),
            (np.r_[2.1, 3], np.r_[0.55, 1.45]),
            (np.r_[-3, -2], np.r_[-1, 0]),
            (np.r_[1, 0.5], np.r_[1, 0.5]),
        ]
    )
    def data_prox(self, request) -> conftest.DataLike:
        arr, out = request.param
        return dict(
            in_=dict(
                arr=arr,
                tau=0.75,  # some random value; doesn't affect prox outcome
            ),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self):
        N_test, dim = 10, 2
        return self._random_array((N_test, dim))


class TestRangeSet(conftest.ProxFuncT):
    @pytest.fixture(  # test matrices
        params=[
            np.array(  # tall matrix
                [
                    [1, 0],
                    [0, 2],
                    [3, 0],
                    [0, 4],
                ]
            ),
            np.diag([1, 2, 3, 4]),  # square matrix
            np.array(  # fat matrix
                [
                    [1, 0, 3, 0],
                    [0, 2, 0, 4],
                ]
            ),
        ]
    )
    def A(self, request) -> np.ndarray:
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, A, request) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        if (xp := ndi.module()) is None:
            pytest.skip(f"{ndi} unsupported on this machine.")

        A = xp.array(A, dtype=width.value)
        A = pyca.LinOp.from_array(A, enable_warnings=False)

        op = pycof.RangeSet(A)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, A) -> pyct.OpShape:
        return (1, A.shape[0])

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
        M, N = A.shape

        u, s, vh = np.linalg.svd(A)
        Q = u[(slice(None), *s.nonzero())]  # orth basis of range(A)
        Qp = np.eye(M) - Q @ Q.T  # orth basis of range(A)^\perp

        # override `in_set` if A is full rank.
        if (s > 0).sum() == min(M, N):
            in_set = True

        if in_set:
            # generate data point in range(A)
            arr = (Q @ Q.T) @ self._random_array((M,), seed=seed)
            out = np.r_[0]
        else:
            # generate data point in range(A)^\perp
            seek = True
            while seek:
                arr = Qp @ self._random_array((M,), seed=seed)
                seek = np.allclose(arr, 0)
            out = np.r_[np.inf]

        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture(params=[0, 3, 7])  # seeds
    def data_prox(self, A, request) -> conftest.DataLike:
        M, _ = A.shape
        arr = self._random_array((M,), seed=request.param)

        Q, _ = np.linalg.qr(A)
        out = Q @ Q.T @ arr
        return dict(
            in_=dict(
                arr=arr,
                tau=1,  # some random value; doesn't affect prox outcome
            ),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, A):
        N_test, dim = 10, A.shape[0]
        return self._random_array((N_test, dim))
