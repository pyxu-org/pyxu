import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Permutation(pyca.UnitOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> x[::-1] (reverse-ordering)
    def __init__(self, dim: int):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return pycu.read_only(arr[..., ::-1])

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return pycu.read_only(arr[..., ::-1])


class TestPermutation(conftest.UnitOpT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, dim):
        return Permutation(dim=dim)

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
