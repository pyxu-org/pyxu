import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestNullOp(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, codim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.NullOp(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        return op, ndi, width

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
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def codim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture
    def data_apply(self, dim_shape, codim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.zeros(codim_shape)
        return dict(
            in_=dict(arr=x),
            out=y,
        )
