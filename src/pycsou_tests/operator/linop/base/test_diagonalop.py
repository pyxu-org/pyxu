import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


# We disable PrecisionWarnings since DiagonalOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
class TestDiagonalOp(conftest.PosDefOpT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture(params=[True, False])
    def vec(self, dim, request):
        v = self._random_array((dim,))
        if request.param:  # positive-definite
            v[v <= 0] = 1
        return np.array(v, dtype=pycrt.Width.SINGLE.value)

    @pytest.fixture
    def op(self, vec):
        return pycl.DiagonalOp(vec=vec)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, vec):
        arr = np.arange(vec.size)
        out = vec * arr
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

    def test_math_eig(self, _op_eig, vec):
        if np.any(vec <= 0):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_eig(_op_eig)

    def test_math_posdef(self, op, vec):
        if np.any(vec <= 0):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op)
