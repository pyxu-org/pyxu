import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


class Map(pyco.Map):
    # f: \bR -> \bR^{3}
    #      x -> [x, x, x]
    def __init__(self):
        super().__init__(shape=(3, 1))

    def lipschitz(self, **kwargs):
        return np.sqrt(3)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        y = arr * np.ones((3,), dtype=arr.dtype)
        return y


class TestMap(conftest.MapT):
    @pytest.fixture
    def op(self):
        return Map()

    @pytest.fixture
    def data_shape(self):
        return (3, 1)

    @pytest.fixture
    def data_lipschitz(self):
        return dict(
            in_=dict(),
            out=np.sqrt(3),
        )

    @pytest.fixture
    def data_apply(self):
        return dict(
            in_=dict(arr=np.array([1])),
            out=np.array([1, 1, 1]),
        )

    @pytest.fixture
    def data_math_lipschitz(self):
        return [np.array([i]) for i in range(5)]
