# How block_diag() tests work:
#
# * BlockDiagMixin auto-defines all (input,output) pairs.
#   [Caveat: we assume the base operators (op_all) are correctly implemented.]
#   (True if choosing test operators from examples/.)
#
# * To test a block-diag operator, inherit from BlockDiagMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.

import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc.operator as pxo
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.blocks as pxb
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


def spec_op(klass: pxt.OpC, N: int = 2) -> list[list[pxt.OpT]]:
    # create all possible (op1, ..., opN) tuples which, when used to create a block-diagonal
    # operator, produce an operator of type `klass`.
    def op_map(dim: int):
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(M=dim)

    def op_diffmap(dim: int):
        import pyxu_tests.operator.examples.test_diffmap as tc

        return tc.Sin(M=dim)

    def op_difffunc(dim: int):
        import pyxu_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=dim)

    def op_proxfunc(dim: int):
        import pyxu_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=dim)

    def op_proxdifffunc(dim: int):
        import pyxu_tests.operator.examples.test_proxdifffunc as tc

        return tc.SquaredL2Norm(M=dim)

    def op_quadraticfunc(dim: int):
        from pyxu_tests.operator.examples.test_linfunc import ScaledSum
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        return pxo.QuadraticFunc(
            shape=(1, dim),
            Q=PSDConvolution(N=dim),
            c=ScaledSum(N=dim),
            t=1,
        )

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

    def condition(ops: list[pxt.OpT], klass: pxt.OpC) -> bool:
        # Return true if block_diag(ops) forms a klass object. [Not a sub-type]
        properties = set.intersection(*[set(op.properties()) for op in ops])
        for p in {
            pxo.Property.FUNCTIONAL,
            pxo.Property.PROXIMABLE,
            pxo.Property.DIFFERENTIABLE_FUNCTION,
            pxo.Property.QUADRATIC,
        }:
            properties.discard(p)
        _klass = pxo.Operator._infer_operator_type(properties)
        return _klass == klass

    ops = []
    for _ops in itertools.combinations(
        [
            # this list should contain operators of each type with at least 2 different sizes.
            op_map(3),
            op_map(4),
            op_diffmap(3),
            op_diffmap(4),
            op_difffunc(4),
            op_difffunc(5),
            op_proxfunc(5),
            op_proxfunc(6),
            op_proxdifffunc(5),
            op_proxdifffunc(6),
            op_quadraticfunc(5),
            op_quadraticfunc(7),
            op_linop(3, 2),
            op_linop(3, 3),
            op_linfunc(5),
            op_linfunc(10),
            op_squareop(4),
            op_squareop(5),
            op_normalop(5),
            op_normalop(6),
            op_unitop(6),
            op_unitop(7),
            op_selfadjointop(5),
            op_selfadjointop(7),
            op_posdefop(5),
            op_posdefop(7),
            op_projop(4),
            op_projop(5),
            op_orthprojop(3),
            op_orthprojop(4),
        ],
        N,
    ):
        if condition(_ops, klass):
            ops.append(_ops)
    return ops


# Data Mixin ------------------------------------------------------------------
class BlockDiagMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_all(self) -> list[pxt.OpT]:
        # Override in inherited class with sub-operators.
        raise NotImplementedError

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_all, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxb.block_diag(op_all)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_all) -> pxt.OpShape:
        dim = sum(op.dim for op in op_all)
        codim = sum(op.codim for op in op_all)
        sh = (codim, dim)
        return sh

    @pytest.fixture
    def data_apply(self, op_all) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        for op in op_all:
            p_arr = self._random_array((op.dim,), seed=3)  # random seed for reproducibility
            parts_arr.append(p_arr)

            p_out = op.apply(p_arr)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op_all) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        for op in op_all:
            p_arr = self._random_array((op.codim,), seed=3)  # random seed for reproducibility
            parts_arr.append(p_arr)

            p_out = op.adjoint(p_arr)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, data_shape[1]
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, data_shape[1]
        return self._random_array((N_test, dim))

    # Tests -------------------------------------------------------------------


# Test classes (Maps) ---------------------------------------------------------
class TestBlockDiagMap(BlockDiagMixin, conftest.MapT):
    @pytest.fixture(params=spec_op(pxo.Map))
    def op_all(self, request):
        return request.param


class TestBlockDiagDiffMap(BlockDiagMixin, conftest.DiffMapT):
    @pytest.fixture(params=spec_op(pxo.DiffMap))
    def op_all(self, request):
        return request.param


class TestBlockDiagLinOp(BlockDiagMixin, conftest.LinOpT):
    @pytest.fixture(params=spec_op(pxo.LinOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagSquareOp(BlockDiagMixin, conftest.SquareOpT):
    @pytest.fixture(params=spec_op(pxo.SquareOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagNormalOp(BlockDiagMixin, conftest.NormalOpT):
    @pytest.fixture(params=spec_op(pxo.NormalOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagUnitOp(BlockDiagMixin, conftest.UnitOpT):
    @pytest.fixture(params=spec_op(pxo.UnitOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagSelfAdjointOp(BlockDiagMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(params=spec_op(pxo.SelfAdjointOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagPosDefOp(BlockDiagMixin, conftest.PosDefOpT):
    @pytest.fixture(params=spec_op(pxo.PosDefOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagProjOp(BlockDiagMixin, conftest.ProjOpT):
    @pytest.fixture(params=spec_op(pxo.ProjOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagOrthProjOp(BlockDiagMixin, conftest.OrthProjOpT):
    @pytest.fixture(params=spec_op(pxo.OrthProjOp))
    def op_all(self, request):
        return request.param
