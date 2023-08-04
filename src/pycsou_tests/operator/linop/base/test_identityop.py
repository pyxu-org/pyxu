import itertools

import numpy as np
import pytest

import pycsou.info.deps as pycd
import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


class TestIdentityOp(conftest.OrthProjOpT):
    @pytest.fixture(params=[1, 10])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, dim, request):
        op = pyco.IdentityOp(dim=dim)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, dim):
        arr = np.arange(dim)
        out = arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
