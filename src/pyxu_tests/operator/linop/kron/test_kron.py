# How kron() tests work:
#
# * KronMixin auto-defines all (input,output) pairs.
#   [Caveat: we assume the base operators (op_A, op_B) are correctly implemented.]
#   (True if choosing test operators from examples/.)
#
# * To test a kron-ed operator (via kron()), inherit from KronMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.

import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest

# It is expected for DenseWarning to be raised when creating some operators, or fallback matrix ops.
pytestmark = pytest.mark.filterwarnings("ignore::pyxu.info.warning.DenseWarning")


# A/B test operators ----------------------------------------------------------
def op_linop(dim: int, codim_scale: int):
    import pyxu_tests.operator.examples.test_linop as tc

    return tc.Tile(N=dim, M=codim_scale)


def op_linfunc(dim: int):
    import pyxu_tests.operator.examples.test_linfunc as tc

    op = tc.ScaledSum(N=dim)
    return op


def op_squareop(dim: int):
    import pyxu_tests.operator.examples.test_squareop as tc

    return tc.CumSum(N=dim)


def op_normalop(dim: int):
    import pyxu_tests.operator.examples.test_normalop as tc

    rng = np.random.default_rng(seed=2)
    h = rng.normal(size=(dim,))
    return tc.CircularConvolution(h=h)


def op_unitop(dim: int):
    import pyxu_tests.operator.examples.test_unitop as tc

    return tc.Permutation(N=dim)


def op_selfadjointop(dim: int):
    import pyxu_tests.operator.examples.test_selfadjointop as tc

    return tc.SelfAdjointConvolution(N=dim)


def op_posdefop(dim: int):
    import pyxu_tests.operator.examples.test_posdefop as tc

    return tc.PSDConvolution(N=dim)


def op_projop(dim: int):
    import pyxu_tests.operator.examples.test_projop as tc

    return tc.Oblique(N=dim, alpha=np.pi / 4)


def op_orthprojop(dim: int):
    import pyxu_tests.operator.examples.test_orthprojop as tc

    return tc.ScaleDown(N=dim)


# Data Mixin ------------------------------------------------------------------
class KronMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_AB(self) -> tuple[pxt.OpT, pxt.OpT]:
        # Override in inherited class with A/B operands.
        raise NotImplementedError

    @pytest.fixture
    def op_A(self, op_AB) -> pxt.OpT:
        return op_AB[0]

    @pytest.fixture
    def op_B(self, op_AB) -> pxt.OpT:
        return op_AB[1]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_A, op_B, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxl.kron(op_A, op_B)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_A, op_B) -> pxt.OpShape:
        sh = (op_A.codim * op_B.codim, op_A.dim * op_B.dim)
        return sh

    @pytest.fixture
    def data_apply(self, op, op_A, op_B) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility
        out = np.kron(op_A.asarray(), op_B.asarray()) @ arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op, op_A, op_B) -> conftest.DataLike:
        arr = self._random_array((op.codim,), seed=20)  # random seed for reproducibility
        out = np.kron(op_A.asarray().T, op_B.asarray().T) @ arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))

    # Tests -------------------------------------------------------------------
    @pytest.mark.skip("See test body for context.")
    def test_interface_asloss(self, op_A):
        # Irrelevant for LinFuncs, and not applicable for non-Funcs.
        pass


# Test classes (Maps) ---------------------------------------------------------
class TestKronLinOp(KronMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (op_linop(3, 2), op_linop(5, 4)),
            (op_linop(3, 2), op_linfunc(7)),
            (op_linop(3, 2), op_squareop(7)),
            (op_linop(3, 2), op_normalop(7)),
            (op_linop(3, 2), op_unitop(7)),
            (op_linop(3, 2), op_selfadjointop(7)),
            (op_linop(3, 2), op_posdefop(7)),
            (op_linop(3, 2), op_projop(7)),
            (op_linop(3, 2), op_orthprojop(7)),
            (op_linfunc(3), op_squareop(7)),
            (op_linfunc(3), op_normalop(7)),
            (op_linfunc(3), op_unitop(7)),
            (op_linfunc(3), op_selfadjointop(7)),
            (op_linfunc(3), op_posdefop(7)),
            (op_linfunc(3), op_projop(7)),
            (op_linfunc(3), op_orthprojop(7)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronSquareOp(KronMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (op_linop(1, 5), op_linfunc(5)),
            (op_linfunc(1), op_squareop(5)),
            (op_linfunc(1), op_normalop(6)),
            (op_linfunc(1), op_unitop(6)),
            (op_linfunc(1), op_selfadjointop(5)),
            (op_linfunc(1), op_posdefop(7)),
            (op_linfunc(1), op_projop(6)),
            (op_linfunc(1), op_orthprojop(6)),
            (op_squareop(5), op_squareop(7)),
            (op_squareop(5), op_normalop(7)),
            (op_squareop(5), op_unitop(7)),
            (op_squareop(5), op_selfadjointop(7)),
            (op_squareop(5), op_posdefop(7)),
            (op_squareop(5), op_projop(7)),
            (op_squareop(5), op_orthprojop(7)),
            (op_normalop(5), op_projop(7)),
            (op_unitop(3), op_projop(5)),
            (op_selfadjointop(5), op_projop(3)),
            (op_posdefop(7), op_projop(4)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronNormalOp(KronMixin, conftest.NormalOpT):
    @pytest.fixture(
        params=[
            (op_normalop(4), op_normalop(5)),
            (op_normalop(4), op_unitop(5)),
            (op_normalop(4), op_selfadjointop(5)),
            (op_normalop(4), op_posdefop(5)),
            (op_normalop(4), op_orthprojop(5)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronUnitOp(KronMixin, conftest.UnitOpT):
    @pytest.fixture(
        params=[
            (op_unitop(4), op_unitop(3)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronSelfAdjointOp(KronMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(
        params=[
            (op_selfadjointop(5), op_selfadjointop(7)),
            (op_selfadjointop(5), op_posdefop(7)),
            (op_selfadjointop(5), op_orthprojop(3)),
            (op_posdefop(7), op_orthprojop(3)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronPosDefOp(KronMixin, conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (op_posdefop(5), op_posdefop(7)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronProjOp(KronMixin, conftest.ProjOpT):
    @pytest.fixture(
        params=[
            (op_projop(2), op_projop(3)),
            (op_projop(2), op_orthprojop(3)),
        ]
    )
    def op_AB(self, request):
        return request.param


class TestKronOrthProjOp(KronMixin, conftest.OrthProjOpT):
    @pytest.fixture(
        params=[
            (op_orthprojop(2), op_orthprojop(3)),
        ]
    )
    def op_AB(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestKronLinFunc(KronMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (op_linfunc(1), op_linfunc(1)),
            (op_linfunc(1), op_linfunc(5)),
            (op_linfunc(5), op_linfunc(1)),
            (op_linfunc(3), op_linfunc(6)),
        ]
    )
    def op_AB(self, request):
        return request.param
