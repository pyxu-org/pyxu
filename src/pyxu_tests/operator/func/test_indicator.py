import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.func as pxf
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestL1Ball(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pxf.L1Ball(dim=5, radius=1)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
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
            ((5, pxf.L2Ball(dim=5, radius=1)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
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
            ((5, pxf.LInfinityBall(dim=5, radius=1)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
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
            ((5, pxf.PositiveOrthant(dim=5)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
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
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        xp = ndi.module()

        v = xp.array([1, 1], dtype=width.value)
        a = pxa.LinFunc.from_array(v, enable_warnings=False)
        op = pxf.HyperSlab(a, lb=-1, ub=2)

        return op, ndi, width

    @pytest.fixture
    def data_shape(self) -> pxt.OpShape:
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
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, A, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        xp = ndi.module()

        A = xp.array(A, dtype=width.value)
        A = pxa.LinOp.from_array(A, enable_warnings=False)

        op = pxf.RangeSet(A)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, A) -> pxt.OpShape:
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


class TestAffineSet(conftest.ProxFuncT):
    @pytest.fixture(  # test matrices
        params=[
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

    @pytest.fixture(params=["zero", "span"])  # where b should lie
    def b(self, A, request) -> np.ndarray:
        M, N = A.shape
        if request.param == "zero":
            b = np.zeros((M,))
        else:  # span
            b = A @ self._random_array((N,))
        return b

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, A, b, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        xp = ndi.module()

        with pxrt.Precision(width):
            A = xp.array(A, dtype=width.value)
            A = pxa.LinOp.from_array(A, enable_warnings=False)
            b = xp.array(b, dtype=width.value)
            op = pxf.AffineSet(A, b)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, A) -> pxt.OpShape:
        return (1, A.shape[1])

    @pytest.fixture(
        params=[  # (seed, in_set)
            (1, True),
            (1, False),
            (6, True),
            (6, False),
        ]
    )
    def data_apply(self, A, b, request):
        seed, in_set = request.param
        M, N = A.shape

        u, s, vh = np.linalg.svd(A)

        # override `in_set` if A is square. -> precondition was to be full-rank
        if M == N:
            in_set = True

        if in_set:
            # generate data point s.t. Ax=b is true
            # seed is not used here; that's ok
            arr, *_ = np.linalg.lstsq(
                A.T @ A,
                A.T @ b,
                rcond=None,  # to silence NumPy's FutureWarning
            )
            out = np.r_[0]
        else:
            # generate data point in range(A)^\perp
            b = b + self._random_array((M,), seed=seed)
            arr, *_ = np.linalg.lstsq(
                A.T @ A,
                A.T @ b,
                rcond=None,  # to silence NumPy's FutureWarning
            )
            out = np.r_[np.inf]

        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture(params=[0, 3, 7])  # seeds
    def data_prox(self, A, b, request) -> conftest.DataLike:
        M, N = A.shape
        arr = self._random_array((N,), seed=request.param)

        y, *_ = np.linalg.lstsq(
            A @ A.T,
            A @ arr - b,
            rcond=None,  # to silence NumPy's FutureWarning
        )
        out = arr - A.T @ y

        return dict(
            in_=dict(
                arr=arr,
                tau=1,  # some random value; doesn't affect prox outcome
            ),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, A):
        N_test, dim = 10, A.shape[1]
        return self._random_array((N_test, dim))
