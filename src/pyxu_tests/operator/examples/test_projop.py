import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class Oblique(pxa.ProjOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> (\alpha E_{N,1} + E_{N, N}) x
    def __init__(self, N: int, alpha: float):
        super().__init__(shape=(N, N))
        self._alpha = float(alpha)

    @pxrt.enforce_precision("arr")
    def apply(self, arr):
        out = np.zeros_like(arr)
        out[..., -1] = (self._alpha * arr[..., 0]) + arr[..., -1]
        return out

    @pxrt.enforce_precision("arr")
    def adjoint(self, arr):
        out = np.zeros_like(arr)
        out[..., 0] = self._alpha * arr[..., -1]
        out[..., -1] = arr[..., -1]
        return out


class TestOblique(conftest.ProjOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, 3.1, Oblique(N=10, alpha=3.1)),),  # dim, alpha, op
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
    def alpha(self, _spec):
        return _spec[0][1]

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
