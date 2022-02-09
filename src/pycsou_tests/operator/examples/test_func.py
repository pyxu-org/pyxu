import numpy as np
import numpy.random as npr
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Median(pyco.Func):
    # f: \bR^{M} -> \bR
    #      x     -> median(x)
    def __init__(self):
        super().__init__(shape=(1, None))

    def lipschitz(self, **kwargs):
        return np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.median(arr, axis=-1, keepdims=True)
        return y


class TestMedian(conftest.FuncT):
    @pytest.fixture
    def op(self):
        return Median()

    @pytest.fixture
    def data_shape(self):
        return (1, None)

    @pytest.fixture
    def data_lipschitz(self):
        return dict(
            in_=dict(),
            out=np.inf,
        )

    @pytest.fixture(
        params=[  # 2 test points
            dict(
                in_=dict(arr=np.arange(-5, 6)),
                out=np.array([0]),
            ),
            dict(
                in_=dict(arr=np.arange(200, 350)),
                out=np.array([274.5]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self):
        rng, N_test = npr.default_rng(seed=2), 5
        return rng.normal(size=(N_test, 3))  # 5 test points
