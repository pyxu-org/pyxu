import dask.array as da
import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


# We disable PrecisionWarnings since DiagonalOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
class TestDiagonalOp(conftest.PosDefOpT):
    @staticmethod
    def skip_unless_numpy(op):
        xp = pycu.get_array_module(op._vec)
        if xp != np:
            pytest.skip("Mathematical test designed for backend-agnostic operators -> safe to disable.")

    @staticmethod
    def skip_if_state_mismatch(op, _gpu):
        xp = pycu.get_array_module(op._vec)
        skip = False
        if _gpu and pycd.CUPY_ENABLED:
            import cupy as cp

            if xp != cp:
                skip = True
        else:
            if xp not in {np, da}:
                skip = True
        if skip:
            msg = f"Got incompatible test configuration (type(vec), _gpu) = {type(op._vec), _gpu} -> safe to skip."
            pytest.skip(msg)

    @pytest.fixture
    def dim(self):
        return 20

    @pytest.fixture(params=[True, False])
    def is_posdef(self, request) -> bool:
        return request.param

    @pytest.fixture
    def vec(self, dim, is_posdef):
        v = self._random_array((dim,))
        if is_posdef:
            v[v <= 0] *= -1
        return v

    @pytest.fixture
    def op(self, vec, xp, width):
        vec = xp.array(vec, dtype=width.value)
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

    def test_math_eig(self, _op_eig, vec):
        if np.any(vec <= 0):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_eig(_op_eig)

    def test_math_posdef(self, op, vec):
        self.skip_unless_numpy(op)
        if np.any(vec <= 0):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op)

    def test_math_lipschitz(self, op, data_math_lipschitz):
        self.skip_unless_numpy(op)
        super().test_math_lipschitz(op, data_math_lipschitz)

    def test_math2_lipschitz(self, op):
        self.skip_unless_numpy(op)
        super().test_math2_lipschitz(op)

    def test_math_diff_lipschitz(self, op, data_math_diff_lipschitz):
        self.skip_unless_numpy(op)
        super().test_math_diff_lipschitz(op, data_math_diff_lipschitz)

    def test_math_adjoint(self, op):
        self.skip_unless_numpy(op)
        super().test_math_adjoint(op)

    def test_math_gram(self, op):
        self.skip_unless_numpy(op)
        super().test_math_gram(op)

    def test_math_cogram(self, op):
        self.skip_unless_numpy(op)
        super().test_math_cogram(op)

    def test_math_normality(self, op):
        self.skip_unless_numpy(op)
        super().test_math_normality(op)

    def test_math_selfadjoint(self, op):
        self.skip_unless_numpy(op)
        super().test_math_selfadjoint(op)

    def test_value_to_sciop(self, op, _op_sciop, _data_to_sciop):
        self.skip_if_dask(op)
        super().test_value_to_sciop(_op_sciop, _data_to_sciop)

    def test_backend_to_sciop(self, op, _op_sciop, _data_to_sciop):
        self.skip_if_dask(op)
        super().test_backend_to_sciop(_op_sciop, _data_to_sciop)

    def test_prec_to_sciop(self, op, _op_sciop, _data_to_sciop):
        self.skip_if_dask(op)
        super().test_prec_to_sciop(_op_sciop, _data_to_sciop)

    def test_value_from_sciop(self, op, _op_sciop, _data_from_sciop):
        self.skip_if_dask(op)
        super().test_value_from_sciop(_op_sciop, _data_from_sciop)

    def test_backend_svdvals(self, op, _gpu):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_backend_svdvals(op, _gpu)

    def test_precCM_svdvals(self, op, _gpu, width):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_precCM_svdvals(op, _gpu, width)

    def test_backend_eigvals(self, op, _gpu):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_backend_eigvals(op, _gpu)

    def test_precCM_eigvals(self, op, _gpu, width):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_precCM_eigvals(op, _gpu, width)

    def test_backend_from_sciop(self, op, _op_sciop, _data_from_sciop):
        self.skip_if_dask(op)
        super().test_backend_from_sciop(_op_sciop, _data_from_sciop)

    def test_prec_from_sciop(self, op, _op_sciop, _data_from_sciop):
        self.skip_if_dask(op)
        super().test_prec_from_sciop(_op_sciop, _data_from_sciop)
