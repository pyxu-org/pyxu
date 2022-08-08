import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou_tests.operator.conftest as conftest


class TestIdentityOp(conftest.OrthProjOpT):
    @pytest.fixture(params=[5])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def op(self, dim):
        return pycl.IdentityOp(dim=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        arr = np.arange(dim)
        out = arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
