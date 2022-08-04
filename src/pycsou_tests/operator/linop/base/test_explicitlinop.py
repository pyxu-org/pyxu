import dask.array as da
import numpy as np
import pytest
import scipy.sparse as sp
import sparse as ssp

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
            cp.array,
            csp.coo_matrix,
            csp.csc_matrix,
            csp.csr_matrix,
            csp.dia_matrix,
        ]
    return init


# We disable PrecisionWarnings since ExplicitLinOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
class ExplicitOpMixin:
    # Mixin class which must be inherited from by each concrete ExplicitLinOp sub-class.
    # Reason: sets all parameters such that users only need to provide the linear operator (in array
    # form) being tested.

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
    def data_apply(self, matrix, xp):
        N = matrix.shape[1]
        arr = xp.array(self._random_array((N,)))
        out = matrix.dot(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


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


class TestExplicitOrthProjOp(ExplicitOpMixin, conftest.OrthProjOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(N=5).asarray()


class TestExplicitNormalOp(ExplicitOpMixin, conftest.NormalOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_normalop as tc

        return tc.CircularConvolution(h=np.ones(5)).asarray()


class TestExplicitUnitOp(ExplicitOpMixin, conftest.UnitOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_unitop as tc

        return tc.Permutation(N=5).asarray()


class TestExplicitSelfAdjointOp(ExplicitOpMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_selfadjointop as tc

        return tc.CDO2(N=5).asarray()


class TestExplicitPosDefOp(ExplicitOpMixin, conftest.PosDefOpT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_posdefop as tc

        return tc.CDO4(N=5).asarray()


class TestExplicitLinFunc(ExplicitOpMixin, conftest.LinFuncT):
    @pytest.fixture
    def matrix(self):
        import pycsou_tests.operator.examples.test_linfunc as tc

        return tc.ScaledSum(N=5).asarray()
