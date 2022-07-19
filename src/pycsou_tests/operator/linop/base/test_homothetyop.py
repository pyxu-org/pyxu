import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou_tests.operator.conftest as conftest


class TestHomothetyOp(conftest.PosDefOpT):
    @pytest.fixture(params=[-3.4, -1, 1, 2])  # cst=0 manually tested given different interface.
    def cst(self, request):
        return request.param

    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, cst, dim):
        return pycl.HomothetyOp(cst=cst, dim=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, cst, dim):
        arr = np.arange(dim)
        out = cst * arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    # No QUAD-precision limitations
    def test_precCM_svdvals(self, op, _gpu, width):
        self._skip_if_disabled()
        super().test_precCM_svdvals(op, _gpu, width)

    # No QUAD-precision limitations
    def test_precCM_eigvals(self, op, _gpu, width):
        self._skip_if_disabled()
        super().test_precCM_eigvals(op, _gpu, width)

    def test_math_eig(self, _op_eig, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_eig(_op_eig)

    def test_math_posdef(self, op, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op)
