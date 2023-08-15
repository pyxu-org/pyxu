# How TransposeRule tests work:
#
# * TransposeRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test a transposed-operator, inherit from TransposeRuleMixin and the suitable conftest.LinOpT
#   subclass which the transposed operator should abide by.


import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


# Test operators --------------------------------------------------------------
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


class TransposeRuleMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be transposed.
        raise NotImplementedError

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = op_orig.T
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_orig) -> pxt.OpShape:
        codim, dim = op_orig.shape
        return (dim, codim)

    @pytest.fixture
    def data_apply(self, op_orig) -> conftest.DataLike:
        x = self._random_array((op_orig.codim,))
        y = op_orig.adjoint(x)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, op_orig) -> conftest.DataLike:
        x = self._random_array((op_orig.dim,))
        y = op_orig.apply(x)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, op_orig) -> conftest.DataLike:
        # We know that linfuncs-to-be must have op.grad(x) = op.asarray()
        x = self._random_array((op_orig.codim,))
        y = op_orig.asarray().flatten()
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, op_orig) -> conftest.DataLike:
        # We know that linfuncs-to-be must have op.prox(x, tau) = x - op.grad(x) * tau
        x = self._random_array((op_orig.codim,))
        g = op_orig.asarray().flatten()
        tau = np.abs(self._random_array((1,)))[0]
        y = x - tau * g
        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )


class TestTransposeRuleSquareOp(TransposeRuleMixin, conftest.SquareOpT):
    @pytest.fixture
    def op_orig(self):
        return op_squareop(7)


class TestTransposeRuleNormalOp(TransposeRuleMixin, conftest.NormalOpT):
    @pytest.fixture
    def op_orig(self):
        return op_normalop(7)


class TestTransposeRuleUnitOp(TransposeRuleMixin, conftest.UnitOpT):
    @pytest.fixture
    def op_orig(self):
        return op_unitop(7)


class TestTransposeRuleSelfAdjointOp(TransposeRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def op_orig(self):
        return op_selfadjointop(7)


class TestTransposeRulePosDefOp(TransposeRuleMixin, conftest.PosDefOpT):
    @pytest.fixture
    def op_orig(self):
        return op_posdefop(7)


class TestTransposeRuleProjOp(TransposeRuleMixin, conftest.ProjOpT):
    @pytest.fixture
    def op_orig(self):
        return op_projop(7)


class TestTransposeRuleOrthProjOp(TransposeRuleMixin, conftest.OrthProjOpT):
    @pytest.fixture
    def op_orig(self):
        return op_orthprojop(7)


class TestTransposeRuleLinOp(TransposeRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            op_linfunc(7),
            op_linop(2, 3),
        ]
    )
    def op_orig(self, request):
        return request.param

    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("which", ["LM", "SM"])
    def test_value1D_svdvals(self, op, ndi, _gpu, _op_svd, k, which):
        self._skip_if_disabled()
        if k > min(op.shape):
            pytest.skip(f"k={k} too high for {op.shape}-shaped operator.")
        super().test_value1D_svdvals(op, ndi, _gpu, _op_svd, k, which)


class TestTransposeRuleLinFunc(TransposeRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            op_linop(1, 7),
            op_linfunc(1),
            op_squareop(1),
            op_normalop(1),
            op_unitop(1),
            op_selfadjointop(1),
            op_posdefop(1),
            # op_projop(1),  Not a projection op!
            # op_orthprojop(1),  Degenerate case; svd tests fail!
        ]
    )
    def op_orig(self, request):
        return request.param
