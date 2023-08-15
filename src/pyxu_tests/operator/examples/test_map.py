import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ReLU(pxa.Map):
    # f: \bR^{M} -> \bR^{M}
    #      x     -> max(x, 0)
    def __init__(self, M: int):
        super().__init__(shape=(M, M))
        self.lipschitz = np.inf

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        y = arr.clip(min=0)
        return y


class TestReLU(conftest.MapT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (3, ReLU(M=3)),
                (1, ReLU(M=1)),
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
