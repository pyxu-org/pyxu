import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


class ReLU(pyca.Map):
    # f: \bR^{M} -> \bR^{M}
    #      x     -> max(x, 0)
    def __init__(self, M: int):
        super().__init__(shape=(M, M))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        y = arr.clip(min=0)
        return y


class TestReLU(conftest.MapT):
    @pytest.fixture
    def dim(self):
        return 3

    @pytest.fixture
    def op(self, dim):
        return ReLU(M=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture(
        params=[  # 2 test points
            dict(
                in_=dict(arr=np.array([-1, 0, 5])),
                out=np.array([0, 0, 5]),
            ),
            dict(
                in_=dict(arr=np.array([-3, 1, 5000])),
                out=np.array([0, 1, 5000]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return self._random_array((N_test, dim))
