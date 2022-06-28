import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class ScaleDown(pyco.OrthProjOp):
    # Drop the last component of a vector
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    @pycrt.enforce_precision("arr")
    def apply(self, arr):
        out = arr.copy()
        out[..., -1] = 0
        return out


class TestScaleDown(conftest.OrthProjOpT):
    @pytest.fixture
    def dim(self):
        return 4

    @pytest.fixture
    def op(self, dim):
        return ScaleDown(dim)

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
