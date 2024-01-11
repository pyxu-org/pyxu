import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestNullFunc(conftest.LinFuncT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.NullFunc(dim_shape=dim_shape)
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

    @pytest.fixture
    def data_apply(self, dim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.zeros((1,))
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    # Q: Why not apply LinOp's test_interface_jacobian() and rely instead on ProxDiffFunc's?
    # A: Because .jacobian() method is forwarded by NullOp(shape=(1, x)).asop(LinFunc), hence
    #    NullFunc().jacobian() is the original NullOp() object, and not the NullFunc() object.
    #    Modifying .asop() to avoid this behaviour is complex, and doesn't matter in practice.
    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        conftest.ProxDiffFuncT.test_interface_jacobian(self, op, _data_apply)
