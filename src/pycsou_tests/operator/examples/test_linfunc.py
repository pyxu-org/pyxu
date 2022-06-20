import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class ScaledSum(pyco.LinFunc):
    # f: \bR^{M} -> \bR
    #      x     -> cumsum(x).sum()
    def __init__(self, N: int):
        super().__init__(shape=(1, N))
        self._lipschitz = np.sqrt(N * (N + 1) * (2 * N + 1) / 6)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return arr.cumsum(axis=-1).sum(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr):
        xp = pycu.get_array_module(arr)
        g = xp.zeros((*arr.shape[:-1], self.dim), dtype=arr.dtype)
        g[..., :] = xp.arange(self.dim, 0, -1, dtype=arr.dtype)
        return g


class TestScaledSum(conftest.LinFuncT):
    disable_test = frozenset(conftest.LinFuncT.disable_test | {"test_interface_asloss"})

    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, dim):
        return ScaledSum(N=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_apply(self, dim):
        x = self._random_array((dim,))
        y = x.cumsum().sum(keepdims=True)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, dim):
        x = self._random_array((dim,))
        y = np.arange(dim, 0, -1)
        return dict(
            in_=dict(arr=x),
            out=y,
        )
