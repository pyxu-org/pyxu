import itertools

import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Permutation(pxa.UnitOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> x[::-1] (reverse-ordering)
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        return pxu.read_only(arr[..., ::-1])

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return pxu.read_only(arr[..., ::-1])


class TestPermutation(conftest.UnitOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, Permutation(N=10)),),  # dim, op
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
        out = arr[::-1]
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
