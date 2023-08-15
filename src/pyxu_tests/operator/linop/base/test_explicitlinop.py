import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


# We disable PrecisionWarnings since ExplicitLinOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pyxu.info.warning.PrecisionWarning")
class ExplicitOpMixin:
    # Mixin class which must be inherited from by each concrete ExplicitLinOp sub-class.
    # Reason: sets all parameters such that users only need to provide the linear operator (in array
    # form) being tested: `matrix`.

    # Internal Helpers --------------------------------------------------------
    @staticmethod
    def spec_data() -> list[tuple[callable, pxd.NDArrayInfo, pxrt.Width]]:
        N = pxd.NDArrayInfo
        S = pxd.SparseArrayInfo
        data = []  # (array_initializer, *accepted input backend/width)

        # NUMPY inputs ------------------------
        data.extend(
            itertools.product(
                [
                    N.NUMPY.module().array,
                    S.SCIPY_SPARSE.module().bsr_matrix,
                    S.SCIPY_SPARSE.module().coo_matrix,
                    S.SCIPY_SPARSE.module().csc_matrix,
                    S.SCIPY_SPARSE.module().csr_matrix,
                    S.PYDATA_SPARSE.module().COO.from_numpy,
                    S.PYDATA_SPARSE.module().GCXS.from_numpy,
                ],
                (N.NUMPY,),
                pxrt.Width,
            )
        )

        # DASK inputs -------------------------
        data.extend(
            itertools.product(
                (N.DASK.module().array,),
                (N.DASK,),
                pxrt.Width,
            )
        )

        # CUPY inputs -------------------------
        if pxd.CUPY_ENABLED:
            cp_t = N.CUPY.module().array
            data.extend(
                itertools.product(
                    [
                        cp_t,
                        lambda _: S.CUPY_SPARSE.module().coo_matrix(cp_t(_)),
                        lambda _: S.CUPY_SPARSE.module().csc_matrix(cp_t(_)),
                        lambda _: S.CUPY_SPARSE.module().csr_matrix(cp_t(_)),
                    ],
                    (N.CUPY,),
                    pxrt.Width,
                )
            )

        return data

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def matrix(self) -> np.ndarray:
        raise NotImplementedError

    @pytest.fixture(params=spec_data())
    def _spec(self, matrix, request):
        init, ndi, width = request.param
        A = init(matrix).astype(width.value)
        op = self.base.from_array(A=A)
        return A, (op, ndi, width)

    @pytest.fixture
    def raw_init_input(self, _spec):
        return _spec[0]

    @pytest.fixture
    def spec(self, _spec):
        return _spec[1]

    @pytest.fixture
    def data_shape(self, matrix):
        return matrix.shape

    @pytest.fixture
    def data_apply(self, matrix):
        N = matrix.shape[1]
        arr = self._random_array((N,))
        out = matrix @ arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    # Tests -------------------------------------------------------------------
    def test_array_not_modified(self, op, raw_init_input):
        # Ensure ExplicitLinOp is using the raw array provided to __init__().
        # The user is expected to know what he is doing.
        assert op.mat is raw_init_input


class TestExplicitLinOp(ExplicitOpMixin, conftest.LinOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Tile(N=10, M=5).asarray()


class TestExplicitSquareOp(ExplicitOpMixin, conftest.SquareOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(N=19).asarray()


class TestExplicitProjOp(ExplicitOpMixin, conftest.ProjOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(N=19, alpha=np.pi / 4).asarray()


class TestExplicitOrthProjOp(ExplicitOpMixin, conftest.OrthProjOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(N=19).asarray()


class TestExplicitNormalOp(ExplicitOpMixin, conftest.NormalOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_normalop as tc

        return tc.CircularConvolution(h=np.ones(20)).asarray()


class TestExplicitUnitOp(ExplicitOpMixin, conftest.UnitOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(N=19).asarray()


class TestExplicitSelfAdjointOp(ExplicitOpMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(N=19).asarray()


class TestExplicitPosDefOp(ExplicitOpMixin, conftest.PosDefOpT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(N=19).asarray()


class TestExplicitLinFunc(ExplicitOpMixin, conftest.LinFuncT):
    @pytest.fixture
    def matrix(self):
        import pyxu_tests.operator.examples.test_linfunc as tc

        return tc.ScaledSum(N=19).asarray()
