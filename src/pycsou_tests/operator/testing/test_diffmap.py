import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class DiffMap(pyco.DiffMap):
    # f: \bR -> \bR^{2}
    #      x -> [sin(x), cos(x)]
    def __init__(self):
        super().__init__(shape=(2, 1))

    def lipschitz(self, **kwargs):
        return 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.concatenate([xp.sin(arr), xp.cos(arr)], axis=-1)
        return y

    def diff_lipschitz(self, **kwargs):
        return 1

    def jacobian(self, arr):
        # J_{f}: \bR -> \bR^{2}
        #          x -> [cos(x), -sin(x)]
        raise NotImplementedError  # todo: pending Matthieu feedback


class TestDiffMap(conftest.DiffMapT):
    @pytest.fixture
    def op(self):
        return DiffMap()

    @pytest.fixture
    def data_shape(self):
        return (2, 1)

    @pytest.fixture
    def data_lipschitz(self):
        return dict(in_=dict(), out=1)

    @pytest.fixture(
        params=[
            dict(
                in_=dict(arr=np.array([0])),
                out=np.array([0, 1]),
            ),
            dict(
                in_=dict(arr=np.array([np.pi / 4])),
                out=np.array([1, 1] / np.sqrt(2)),
            ),
            dict(
                in_=dict(arr=np.array([np.pi / 2])),
                out=np.array([1, 0]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self):
        return [np.array([i]) for i in range(5)]

    @pytest.fixture
    def data_diff_lipschitz(self):
        return dict(in_=dict(), out=1)

    # @pytest.fixture
    # def data_math_diff_lipschitz(self):
    #     pass  # todo
