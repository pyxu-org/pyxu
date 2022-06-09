import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Tile(pyco.LinOp):
    # f: \bR^{N} -> \bR^{N \times M = NM}
    #      x     -> [x ... x] (M times)
    def __init__(self, N: int, M: int):
        super().__init__(shape=(N * M, N))
        self._lipschitz = np.sqrt(M)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        M = self.codim // self.dim
        xp = pycu.get_array_module(arr)
        y = xp.concatenate([arr] * M, axis=-1)
        return y

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        M = self.codim // self.dim
        sh = (*arr.shape[:-1], self.dim)
        y = arr.reshape((-1, M, self.dim)).sum(axis=-2).reshape(sh)
        return y


class TestTile(conftest.LinOpT):
    @pytest.fixture
    def dim(self):
        return 3

    @pytest.fixture
    def codim(self):
        return 12

    @pytest.fixture
    def op(self, codim, dim):
        return Tile(N=dim, M=codim // dim)

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
