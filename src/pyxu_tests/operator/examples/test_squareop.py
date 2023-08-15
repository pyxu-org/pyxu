import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class CumSum(pxa.SquareOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> [x1, x1+x2, ..., x1+...+xN]
    def __init__(self, N: int):
        super().__init__(shape=(N, N))
        self.lipschitz = np.sqrt(N * (N + 1) / 2)  # Frobenius norm

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        y = arr.cumsum(axis=-1)
        return y

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        y = arr[..., ::-1].cumsum(axis=-1)[..., ::-1]
        return y


class TestCumSum(conftest.SquareOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, CumSum(N=10)),),  # dim, op
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

    @pytest.fixture
    def data_apply(self, dim):
        arr = self._random_array((dim,))
        out = np.cumsum(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
