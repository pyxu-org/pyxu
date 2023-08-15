import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Tile(pxa.LinOp):
    # f: \bR^{N} -> \bR^{N \times M = NM}
    #      x     -> [x ... x] (M times)
    def __init__(self, N: int, M: int):
        super().__init__(shape=(N * M, N))
        self.lipschitz = np.sqrt(M)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        M = self.codim // self.dim
        xp = pxu.get_array_module(arr)
        y = xp.concatenate([arr] * M, axis=-1)
        return y

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        M = self.codim // self.dim
        sh = (*arr.shape[:-1], self.dim)
        y = arr.reshape((-1, M, self.dim)).sum(axis=-2).reshape(sh)
        return y


class TestTile(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, 120, Tile(N=10, M=120 // 10)),),  # dim, codim, op
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][2], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def codim(self, _spec):
        return _spec[0][1]

    @pytest.fixture
    def data_shape(self, codim, dim):
        return (codim, dim)

    @pytest.fixture
    def data_apply(self, codim, dim):
        arr = np.arange(dim)
        out = np.arange(codim) % dim
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
