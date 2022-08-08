import dask.array as da
import numpy as np
import pytest
import scipy.sparse as sp
import sparse as ssp

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


def array_initializers() -> list[callable]:
    init = [
        np.array,
        da.array,
        sp.bsr_array,
        sp.coo_array,
        sp.csc_array,
        sp.csr_array,
        sp.dia_array,
        sp.dok_array,
        sp.lil_array,
        ssp.COO.from_numpy,
        ssp.DOK.from_numpy,
        ssp.GCXS.from_numpy,
    ]
    if pycd.CUPY_ENABLED:
        import cupy as cp
        import cupyx.scipy.sparse as csp

        init += [
            # cp.array,
            # csp.coo_matrix,
            # csp.csc_matrix,
            # csp.csr_matrix,
            # csp.dia_matrix,
        ]
    return init


# We disable PrecisionWarnings since ExplicitLinOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
class ExplicitOpMixin:
    # Mixin class which must be inherited from by each concrete ExplicitLinOp sub-class.
    # Reason: sets all parameters such that users only need to provide the linear operator (in array
    # form) being tested.

    @staticmethod
    def skip_if_state_mismatch(op, _gpu):
        xp = pycu.get_array_module(op._mat, fallback=np)
        skip = False
        if _gpu and pycd.CUPY_ENABLED:
            import cupy as cp

            if xp != cp:
                skip = True
        else:
            if xp not in {np, da}:
                skip = True
        if skip:
            msg = f"Got incompatible test configuration (type(mat), _gpu) = {type(op._mat), _gpu} -> safe to skip."
            pytest.skip(msg)

    @pytest.fixture
    def matrix(self) -> np.ndarray:
        # To be specified by sub-classes
        raise NotImplementedError

    @pytest.fixture(params=array_initializers())
    def _matrix(self, matrix, width, request):
        initializer = request.param
        return initializer(matrix.astype(width.value))

    @pytest.fixture
    def xp(self, _matrix):
        # Recall (not all) explicit operator methods are backend-agnostic
        # -> limit eval to module they are defined in.
        return pycu.get_array_module(_matrix, fallback=np)  # fallback if _matrix is sparse.

    @pytest.fixture
    def op(self, _matrix):
        return self.base.from_array(_matrix)

    @pytest.fixture
    def data_shape(self, _matrix):
        return _matrix.shape

    @pytest.fixture
    def data_apply(self, _matrix, xp):
        N = _matrix.shape[1]
        arr = xp.array(self._random_array((N,)))
        out = _matrix.dot(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    def test_backend_svdvals(self, op, _gpu):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_backend_svdvals(op, _gpu)

    def test_precCM_svdvals(self, op, _gpu, width):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_precCM_svdvals(op, _gpu, width)


class ExplicitOpNormalMixin(ExplicitOpMixin):
    def test_backend_eigvals(self, op, _gpu):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_backend_eigvals(op, _gpu)

    # local override of this fixture
    # We use the complex-valued types since .eigvals() should return complex. (Exception: SelfAdjointOp)
    @pytest.mark.parametrize("width", list(pycrt._CWidth))
    def test_precCM_eigvals(self, op, _gpu, width):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_precCM_eigvals(op, _gpu, width)


class ExplicitOpSelfAdjointMixin(ExplicitOpNormalMixin):
    def test_backend_eigvals(self, op, _gpu):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_backend_eigvals(op, _gpu)

    # local override of this fixture
    # We revert back to real-valued types since .eigvals() should return real.
    @pytest.mark.parametrize("width", list(pycrt.Width))
    def test_precCM_eigvals(self, op, _gpu, width):
        self.skip_if_state_mismatch(op, _gpu)
        super().test_precCM_eigvals(op, _gpu, width)


class TestExplicitLinOp(ExplicitOpMixin, conftest.LinOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_linop as tc

        return tc.Tile(N=3, M=5).asarray()


class TestExplicitSquareOp(ExplicitOpMixin, conftest.SquareOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_squareop as tc

        return tc.CumSum(N=5).asarray()


class TestExplicitProjOp(ExplicitOpMixin, conftest.ProjOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_projop as tc

        return tc.Oblique(N=5, alpha=np.pi / 4).asarray()


class TestExplicitOrthProjOp(ExplicitOpSelfAdjointMixin, conftest.OrthProjOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(N=5).asarray()


class TestExplicitNormalOp(ExplicitOpNormalMixin, conftest.NormalOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_normalop as tc

        return tc.CircularConvolution(h=np.ones(5)).asarray()


class TestExplicitUnitOp(ExplicitOpNormalMixin, conftest.UnitOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_unitop as tc

        return tc.Permutation(N=5).asarray()


class TestExplicitSelfAdjointOp(ExplicitOpSelfAdjointMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_selfadjointop as tc

        return tc.CDO2(N=5).asarray()


class TestExplicitPosDefOp(ExplicitOpSelfAdjointMixin, conftest.PosDefOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_posdefop as tc

        return tc.CDO4(N=5).asarray()


class TestExplicitLinFunc(ExplicitOpMixin, conftest.LinFuncT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_linfunc as tc

        return tc.ScaledSum(N=5).asarray()
