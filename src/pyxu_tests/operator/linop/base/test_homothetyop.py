import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestHomothetyOp(conftest.PosDefOpT):
    @pytest.fixture(params=[-3.4, -1, 1, 2])  # cst=0 manually tested given different interface.
    def cst(self, request):
        return request.param

    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, cst, dim, request):
        op = pxo.HomothetyOp(cst=cst, dim=dim)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, cst, dim):
        arr = np.arange(dim)
        out = cst * arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    def test_math_posdef(self, op, xp, width, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op, xp, width)
