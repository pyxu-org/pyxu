import itertools

import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestHomothetyOp(conftest.PosDefOpT):
    @pytest.fixture(params=[-3.4, -1, 1, 2])  # cst=0 manually tested given different interface.
    def cst(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, cst, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.HomothetyOp(
            dim_shape=dim_shape,
            cst=cst,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, cst, dim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = cst * x
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    def test_math_posdef(self, op, xp, width, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op, xp, width)
