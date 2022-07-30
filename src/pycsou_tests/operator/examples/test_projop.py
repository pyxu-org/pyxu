import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Oblique(pyca.ProjOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> (\alpha E_{N,1} + E_{N, N}) x
    def __init__(self, N: int, alpha: float):
        super().__init__(shape=(N, N))
        self._alpha = float(alpha)

    @pycrt.enforce_precision("arr")
    def apply(self, arr):
        out = np.zeros_like(arr)
        out[..., -1] = (self._alpha * arr[..., 0]) + arr[..., -1]
        return out

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr):
        out = np.zeros_like(arr)
        out[..., 0] = self._alpha * arr[..., -1]
        out[..., -1] = arr[..., -1]
        return out


class TestOblique(conftest.ProjOpT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def alpha(self):
        return 3.1

    @pytest.fixture
    def op(self, dim, alpha):
        return Oblique(N=dim, alpha=alpha)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim, alpha):
        x = self._random_array((dim,))
        y = np.zeros(dim)
        y[-1] = alpha * x[0] + x[-1]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
