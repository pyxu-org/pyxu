import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator.func as pxf
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pxf.L1Norm(dim=5)),),  # dim, op
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
                out=np.array([7]),
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
                out=np.array([-2, -1, 0, 0, 0]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestL2Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pxf.L2Norm(dim=5)),),  # dim, op
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
                out=np.sqrt([15]),
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
                out=(1 - 1 / np.sqrt(15)) * np.arange(-3, 2),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestSquaredL2Norm(conftest.QuadraticFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((7, pxf.SquaredL2Norm(dim=7)),),  # dim, op
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
                in_=dict(arr=np.zeros((7,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 4)),
                out=np.array([28]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((7,))),
                out=np.zeros((7,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 4)),
                out=2 * np.arange(-3, 4),
            ),
        ]
    )
    def data_grad(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((7,)),
                    tau=1,
                ),
                out=np.zeros((7,)),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 4),
                    tau=1,
                ),
                out=np.arange(-3, 4) / 3,
            ),
        ]
    )
    def data_prox(self, request):
        return request.param


# We disable PerformanceWarnings due to prox_algo="sort" used on Dask inputs.
@pytest.mark.filterwarnings("ignore::pyxu.info.warning.PerformanceWarning")
class TestSquaredL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (5, pxf.SquaredL1Norm(dim=5, prox_algo="sort")),
                (5, pxf.SquaredL1Norm(dim=5, prox_algo="root")),
            ),
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


class TestLInfinityNorm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pxf.LInfinityNorm(dim=5)),),  # dim, op
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
                out=np.array([3]),
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
                out=np.array([-2, -2, -1, 0, 1]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))


class TestL21Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((6, pxf.L21Norm(arg_shape=(2, 3))),),  # dim, op
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
                in_=dict(arr=np.zeros((6,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.array([1, 2, -3, 0, -2, -4])),
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
                    arr=np.zeros((6,)),
                    tau=1,
                ),
                out=np.zeros((6,)),
            ),
            dict(
                in_=dict(
                    arr=np.array([1, 2, -3, 0, -2, -4]),
                    tau=4,
                ),
                out=np.array([0, 0, -3 / 5, 0, 0, -4 / 5]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 10
        return self._random_array((N_test, dim))


class TestPositiveL1Norm(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pxf.PositiveL1Norm(dim=5)),),  # dim, op
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
        params=[  # 3 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([np.inf]),
            ),
            dict(
                in_=dict(arr=np.arange(5)),
                out=np.array([10]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 3 evaluation points
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
                out=np.array([0, 0, 0, 0, 0]),
            ),
            dict(
                in_=dict(
                    arr=np.arange(5),
                    tau=1,
                ),
                out=np.array([0, 0, 1, 2, 3]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))
