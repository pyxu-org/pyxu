# How ArgShiftRule tests work:
#
# * ArgShiftRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test an arg-shifted-operator, inherit from ArgShiftRuleMixin and the suitable REDEFINED MapT
#   subclass which the arg-shifted operator should abide by. (Redefined in this module to disable
#   problematic tests; see comments below.)
#
# Important: argshifted-operators are not module/precision-agnostic!

import collections.abc as cabc

import numpy as np
import pytest

import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest
import pycsou_tests.operator.examples.test_diffmap as tc_diffmap
import pycsou_tests.operator.examples.test_linfunc as tc_linfunc
import pycsou_tests.operator.examples.test_linop as tc_linop
import pycsou_tests.operator.examples.test_normalop as tc_normalop
import pycsou_tests.operator.examples.test_orthprojop as tc_orthprojop
import pycsou_tests.operator.examples.test_posdefop as tc_posdefop
import pycsou_tests.operator.examples.test_projop as tc_projop
import pycsou_tests.operator.examples.test_proxdifffunc as tc_proxdifffunc
import pycsou_tests.operator.examples.test_selfadjointop as tc_selfadjointop
import pycsou_tests.operator.examples.test_squareop as tc_squareop
import pycsou_tests.operator.examples.test_unitop as tc_unitop

rng = np.random.default_rng(seed=50)


class ArgShiftRuleMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pyct.OpT:
        # Override in inherited class with the operator to be arg-shifted.
        raise NotImplementedError

    @pytest.fixture
    def op_shift(self, op_orig) -> pyct.NDArray:
        # Arg-shift values applied to op_orig()
        dim = self._sanitize(op_orig.dim, 7)
        cst = self._random_array((dim,), seed=20)  # random seed for reproducibility
        return cst

    @pytest.fixture
    def op(self, op_orig, op_shift, xp, width) -> pyct.OpT:
        shift = xp.array(op_shift, dtype=width.value)  # See top-level disclaimer(s).
        return op_orig.argshift(shift)

    @pytest.fixture
    def data_shape(self, op_orig) -> pyct.Shape:
        return op_orig.shape

    @pytest.fixture
    def data_apply(self, op_orig, op_shift) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_orig.apply(arr + op_shift)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op_orig, op_shift) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_orig.grad(arr + op_shift)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op_orig, op_shift) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        tau = np.abs(self._random_array((1,), seed=21))[0]  # random seed for reproducibility
        out = op_orig.prox(arr + op_shift, tau) - op_shift
        return dict(
            in_=dict(
                arr=arr,
                tau=tau,
            ),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, self._sanitize(op.dim, 7)
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, self._sanitize(op.dim, 7)
        return self._random_array((N_test, dim))


# Redefined MapT classes ------------------------------------------------------
#
# Arg-shifted operators are not module/precision-agnostic: users are expected to apply/grad/prox()
# them using the same array-module as `shift`.
#
# Problem: some tests (namely `test_math*` family defined in `numpy_only_tests`) are explicitly
# designed to run with NumPy input/outputs only.
#
# Consequence: we need to disable these tests since there is no non-repetitive way to re-parametrize
# these functions to only run when `xp=np`.
#
# This is achieved by sub-classing `conftest.MapT` classes to override `disable_test`.
numpy_only_tests = frozenset(
    {
        "test_math_lipschitz",
        "test_math_diff_lipschitz",
        "test_math1_grad",
        "test_math2_grad",
        "test_math_prox",
        "test_math1_moreau_envelope",
        "test_math2_moreau_envelope",
    }
)


class MapT(conftest.MapT):
    disable_test = frozenset(conftest.MapT.disable_test | numpy_only_tests)


class DiffMapT(conftest.DiffMapT):
    disable_test = frozenset(conftest.DiffMapT.disable_test | numpy_only_tests)


class FuncT(conftest.FuncT):
    disable_test = frozenset(conftest.FuncT.disable_test | numpy_only_tests)


class DiffFuncT(conftest.DiffFuncT):
    disable_test = frozenset(conftest.DiffFuncT.disable_test | numpy_only_tests)


class ProxFuncT(conftest.ProxFuncT):
    disable_test = frozenset(conftest.ProxFuncT.disable_test | numpy_only_tests)


class _QuadraticFuncT(conftest._QuadraticFuncT):
    disable_test = frozenset(conftest._QuadraticFuncT.disable_test | numpy_only_tests)


class ProxDiffFuncT(conftest.ProxDiffFuncT):
    disable_test = frozenset(conftest.ProxDiffFuncT.disable_test | numpy_only_tests)


# Test classes (Maps) ---------------------------------------------------------
class TestArgShiftRuleMap(ArgShiftRuleMixin, MapT):
    @pytest.fixture
    def op_orig(self):
        import pycsou_tests.operator.examples.test_map as tc

        return tc.ReLU(M=6)


class TestArgShiftRuleDiffMap(ArgShiftRuleMixin, DiffMapT):
    @pytest.fixture(
        params=[
            tc_diffmap.Sin(M=6),
            tc_linop.Tile(N=3, M=4),
            tc_squareop.CumSum(N=7),
            tc_normalop.CircularConvolution(h=rng.normal(size=(5,))),
            tc_unitop.Rotation(ax=0, ay=0, az=np.pi / 3),
            tc_selfadjointop.CDO2(N=7),
            tc_posdefop.CDO4(N=7),
            tc_projop.CabinetProjection(angle=np.pi / 4),
            tc_orthprojop.ScaleDown(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestArgShiftRuleFunc(ArgShiftRuleMixin, FuncT):
    @pytest.fixture
    def op_orig(self):
        import pycsou_tests.operator.examples.test_func as tc

        return tc.Median()


class TestArgShiftRuleDiffFunc(ArgShiftRuleMixin, DiffFuncT):
    @pytest.fixture(params=[None, 7])
    def op_orig(self, request):
        import pycsou_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=request.param)


class TestArgShiftRuleProxFunc(ArgShiftRuleMixin, ProxFuncT):
    @pytest.fixture(params=[None, 7])
    def op_orig(self, request):
        import pycsou_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=request.param)


class TestArgShiftRuleQuadraticFunc(ArgShiftRuleMixin, _QuadraticFuncT):
    @pytest.fixture
    def op_orig(self):
        from pycsou.operator.func import QuadraticFunc

        return QuadraticFunc(
            Q=tc_posdefop.CDO4(N=7),
            c=tc_linfunc.ScaledSum(N=7),
            t=1,
        )


class TestArgShiftRuleProxDiffFunc(ArgShiftRuleMixin, ProxDiffFuncT):
    @pytest.fixture(
        params=[
            tc_proxdifffunc.SquaredL2Norm(M=7),
            tc_proxdifffunc.SquaredL2Norm(M=None),
            tc_linfunc.ScaledSum(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param
