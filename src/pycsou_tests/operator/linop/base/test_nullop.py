import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou_tests.operator.conftest as conftest


class TestNullOp(conftest.LinOpT):
    @pytest.fixture
    def op(self, data_shape):
        return pycl.NullOp(shape=data_shape)

    @pytest.fixture
    def data_shape(self):
        return (3, 4)

    @pytest.fixture
    def data_apply(self, data_shape):
        arr = np.arange(data_shape[1])
        out = np.zeros(data_shape[0])
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    # No QUAD-precision limitations
    def test_precCM_svdvals(self, op, _gpu, width):
        self._skip_if_disabled()
        super().test_precCM_svdvals(op, _gpu, width)
