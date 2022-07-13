import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class CumSum(pyca.SquareOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> [x1, x1+x2, ..., x1+...+xN]
    def __init__(self, N: int):
        super().__init__(shape=(N, N))
        self._lipschitz = np.sqrt(N * (N + 1) / 2)  # Frobenius norm

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.cumsum(arr, axis=-1)
        return y

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.cumsum(arr[..., ::-1], axis=-1)[..., ::-1]
        return y


class TestCumSum(conftest.SquareOpT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, dim):
        return CumSum(N=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        arr = self._random_array((dim,))
        out = np.cumsum(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
