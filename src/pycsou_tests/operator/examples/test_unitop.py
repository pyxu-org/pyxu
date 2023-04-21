import itertools

import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class Permutation(pyca.UnitOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> x[::-1] (reverse-ordering)
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return pycu.read_only(arr[..., ::-1])

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return pycu.read_only(arr[..., ::-1])


class TestPermutation(conftest.UnitOpT):
    @pytest.fixture(
        params=itertools.product(
            ((10, Permutation(N=10)),),  # dim, op
            pycd.NDArrayInfo,
            pycrt.Width,
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
