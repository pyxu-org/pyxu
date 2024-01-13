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

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


def spec_op(klass: pxt.OpC, N: int = 2) -> list[list[pxt.OpT]]:
    # create all possible (op1, ..., opN) tuples which, when used to create a block-diagonal
    # operator, produce an operator of type `klass`.
    def op_map(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(dim_shape=dim_shape)

    def op_func(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_func as tc

        return tc.Median(dim_shape=dim_shape)

    def op_diffmap(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_diffmap as tc

        return tc.Sin(dim_shape=dim_shape)

    def op_difffunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(dim_shape=dim_shape)

    def op_proxfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(dim_shape=dim_shape)

    def op_proxdifffunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_proxdifffunc as tc

        return tc.SquaredL2Norm(dim_shape=dim_shape)

    def op_quadraticfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        # QuadraticFunc may be defined for dim=1.
        # In this case we cannot use PSDConvolution (examples/test_posdefop.py) due to minimal domain-size restrictions.
        # We therefore use HomothetyOp without loss of generality.

        from pyxu.operator import HomothetyOp
        from pyxu_tests.operator.examples.test_linfunc import Sum

        return pxa.QuadraticFunc(
            dim_shape=dim_shape,
            codim_shape=1,
            Q=HomothetyOp(dim_shape=dim_shape, cst=3),
            c=Sum(dim_shape=dim_shape),
            t=1,
        )

    def op_linop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Sum(dim_shape=dim_shape)

    def op_linfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linfunc as tc

        op = tc.Sum(dim_shape=dim_shape)
        return op

    def op_squareop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(dim_shape=dim_shape)

    def op_normalop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_normalop as tc

        rng = np.random.default_rng(seed=2)
        conv_filter = rng.normal(size=dim_shape[-1])
        return tc.CircularConvolution(
            dim_shape=dim_shape,
            h=conv_filter,
        )

    def op_unitop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(dim_shape=dim_shape)

    def op_selfadjointop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(dim_shape=dim_shape)

    def op_posdefop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(dim_shape=dim_shape)

    def op_projop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(
            dim_shape=dim_shape,
            alpha=np.pi / 4,
        )

    def op_orthprojop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(dim_shape=dim_shape)

    def condition(ops: list[pxt.OpT], klass: pxt.OpC) -> bool:
        # Return true if block_diag(ops) forms a klass object. [Not a sub-type]
        properties = set.intersection(*[set(op.properties()) for op in ops])
        for p in {
            pxa.Property.FUNCTIONAL,
            pxa.Property.PROXIMABLE,
            pxa.Property.DIFFERENTIABLE_FUNCTION,
            pxa.Property.QUADRATIC,
        }:
            properties.discard(p)
        _klass = pxa.Operator._infer_operator_type(properties)
        return _klass == klass

    ops = []
    for _ops in itertools.combinations_with_replacement(
        [
            op_func((5, 3, 5)),
            op_difffunc((5, 3, 5)),
            op_proxfunc((5, 3, 5)),
            op_proxdifffunc((5, 3, 5)),
            op_linfunc((5, 3, 5)),
            op_quadraticfunc((5, 3, 5)),
        ],
        N,
    ):
        if condition(_ops, klass):
            ops.append(_ops)
    for _ops in itertools.combinations_with_replacement(
        [
            op_map((5, 3, 5)),
            op_diffmap((5, 3, 5)),
            op_squareop((5, 3, 5)),
            op_normalop((5, 3, 5)),
            op_unitop((5, 3, 5)),
            op_selfadjointop((5, 3, 5)),
            op_posdefop((5, 3, 5)),
            op_projop((5, 3, 5)),
            op_orthprojop((5, 3, 5)),
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
        op = pxo.block_diag(op_all)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_all) -> pxt.NDArrayShape:
        N_op = len(op_all)
        dim_shape0 = op_all[0].dim_shape
        return (N_op, *dim_shape0)

    @pytest.fixture
    def codim_shape(self, op_all) -> pxt.NDArrayShape:
        N_op = len(op_all)
        codim_shape0 = op_all[0].codim_shape
        return (N_op, *codim_shape0)

    @pytest.fixture
    def data_apply(self, dim_shape, op_all) -> conftest.DataLike:
        x = self._random_array(dim_shape)

        y = [None] * len(op_all)
        for i in range(len(op_all)):
            y[i] = op_all[i].apply(x[i])
        y = np.stack(y, axis=0)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, codim_shape, op_all) -> conftest.DataLike:
        x = self._random_array(codim_shape)

        y = [None] * len(op_all)
        for i in range(len(op_all)):
            y[i] = op_all[i].adjoint(x[i])
        y = np.stack(y, axis=0)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 20
        x = self._random_array((N_test, *dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 20
        x = self._random_array((N_test, *dim_shape))
        return x

    # Tests -------------------------------------------------------------------


# Test classes (Maps) ---------------------------------------------------------
class TestBlockDiagMap(BlockDiagMixin, conftest.MapT):
    @pytest.fixture(params=spec_op(pxa.Map))
    def op_all(self, request):
        return request.param


class TestBlockDiagDiffMap(BlockDiagMixin, conftest.DiffMapT):
    @pytest.fixture(params=spec_op(pxa.DiffMap))
    def op_all(self, request):
        return request.param


class TestBlockDiagLinOp(BlockDiagMixin, conftest.LinOpT):
    @pytest.fixture(params=spec_op(pxa.LinOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagSquareOp(BlockDiagMixin, conftest.SquareOpT):
    @pytest.fixture(params=spec_op(pxa.SquareOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagNormalOp(BlockDiagMixin, conftest.NormalOpT):
    @pytest.fixture(params=spec_op(pxa.NormalOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagUnitOp(BlockDiagMixin, conftest.UnitOpT):
    @pytest.fixture(params=spec_op(pxa.UnitOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagSelfAdjointOp(BlockDiagMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(params=spec_op(pxa.SelfAdjointOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagPosDefOp(BlockDiagMixin, conftest.PosDefOpT):
    @pytest.fixture(params=spec_op(pxa.PosDefOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagProjOp(BlockDiagMixin, conftest.ProjOpT):
    @pytest.fixture(params=spec_op(pxa.ProjOp))
    def op_all(self, request):
        return request.param


class TestBlockDiagOrthProjOp(BlockDiagMixin, conftest.OrthProjOpT):
    @pytest.fixture(params=spec_op(pxa.OrthProjOp))
    def op_all(self, request):
        return request.param
