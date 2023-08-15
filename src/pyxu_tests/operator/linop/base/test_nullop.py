import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestNullOp(conftest.LinOpT):
    @pytest.fixture
    def data_shape(self):
        return (3, 4)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, data_shape, request):
        op = pxo.NullOp(shape=data_shape)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, data_shape):
        arr = np.arange(data_shape[1])
        out = np.zeros(data_shape[0])
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
