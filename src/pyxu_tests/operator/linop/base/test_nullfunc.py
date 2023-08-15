import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestNullFunc(conftest.LinFuncT):
    @pytest.fixture(params=[1, 10])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim, request):
        op = pxo.NullFunc(dim=dim)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, dim):
        arr = np.arange(dim)
        out = np.zeros((1,))
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    # Q: Why not apply LinOp's test_interface_jacobian() and rely instead on ProxDiffFunc's?
    # A: Because .jacobian() method is forwarded by NullOp(shape=(1, x)).asop(LinFunc), hence
    #    NullFunc().jacobian() is the original NullOp() object, and not the NullFunc() object.
    #    Modifying .asop() to avoid this behaviour is complex. Moreover it only affects NullFunc,
    #    the only squeezed LinOp which we explicitly need to test for correctness.
    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        conftest.ProxDiffFuncT.test_interface_jacobian(self, op, _data_apply)
