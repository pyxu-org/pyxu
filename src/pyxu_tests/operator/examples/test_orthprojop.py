import itertools

import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ScaleDown(pxa.OrthProjOp):
    # Drop the last component of a vector
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    @pxrt.enforce_precision("arr")
    def apply(self, arr):
        out = arr.copy()
        out[..., -1] = 0
        return out


class TestScaleDown(conftest.OrthProjOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, ScaleDown(N=10)),),  # dim, op
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
        x = self._random_array((dim,))
        y = x.copy()
        y[-1] = 0
        return dict(
            in_=dict(arr=x),
            out=y,
        )
