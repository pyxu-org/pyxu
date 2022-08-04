import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou_tests.operator.conftest as conftest


class TestNullFunc(conftest.LinFuncT):
    disable_test = frozenset(
        conftest.LinFuncT.disable_test
        | {
            "test_math2_grad",  # trivially correct, but raises warning since L=0
        }
    )

    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, dim):
        return pycl.NullFunc(dim=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

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
