import itertools

import numpy as np
import pytest

import pycsou.info.deps as pycd
import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


class TestNullOp(conftest.LinOpT):
    @pytest.fixture
    def data_shape(self):
        return (3, 4)

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, data_shape, request):
        op = pyco.NullOp(shape=data_shape)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, data_shape):
        arr = np.arange(data_shape[1])
        out = np.zeros(data_shape[0])
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
