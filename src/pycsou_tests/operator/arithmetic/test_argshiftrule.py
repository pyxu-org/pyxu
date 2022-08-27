# How ArgShiftRule tests work:
#
# * ArgShiftRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test an arg-shifted-operator, inherit from ArgShiftRuleMixin and the suitable MapT subclass
#   which the arg-shifted operator should abide by.
#
# Important: argshifted-operators are not module/precision-agnostic!

import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
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

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, op_orig, op_shift, request) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        if (xp := ndi.module()) is not None:
            shift = xp.array(op_shift, dtype=width.value)
            op = op_orig.argshift(shift)
        else:
            # Some test functions run without needing `xp`, hence it is required to add extra
            # skip-logic in `spec` as well.
            pytest.skip(f"{ndi} unsupported on this machine.")
        return op, ndi, width

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

    # Tests -------------------------------------------------------------------
    def test_interface_asloss(self, op, xp, width, op_orig):
        self._skip_if_disabled()
        if not op_orig.has(pyca.Property.FUNCTIONAL):
            pytest.skip("asloss() unavailable for non-functionals.")

        try:
            op_orig.asloss()  # detect if fails
            super().test_interface_asloss(op, xp, width)
        except NotImplementedError as exc:
            pytest.skip("asloss() unsupported by base operator.")


# Test classes (Maps) ---------------------------------------------------------
class TestArgShiftRuleMap(ArgShiftRuleMixin, conftest.MapT):
    @pytest.fixture
    def op_orig(self):
        import pycsou_tests.operator.examples.test_map as tc

        return tc.ReLU(M=6)


class TestArgShiftRuleDiffMap(ArgShiftRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            tc_diffmap.Sin(M=6),
            tc_linop.Tile(N=3, M=4),
            tc_squareop.CumSum(N=7),
            tc_normalop.CircularConvolution(h=rng.normal(size=(5,))),
            tc_unitop.Permutation(N=7),
            tc_selfadjointop.CDO2(N=7),
            tc_posdefop.CDO4(N=7),
            tc_projop.Oblique(N=6, alpha=np.pi / 4),
            tc_orthprojop.ScaleDown(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestArgShiftRuleFunc(ArgShiftRuleMixin, conftest.FuncT):
    @pytest.fixture
    def op_orig(self):
        import pycsou_tests.operator.examples.test_func as tc

        return tc.Median()


class TestArgShiftRuleDiffFunc(ArgShiftRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(params=[None, 7])
    def op_orig(self, request):
        import pycsou_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=request.param)


class TestArgShiftRuleProxFunc(ArgShiftRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(params=[None, 7])
    def op_orig(self, request):
        import pycsou_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=request.param)


class TestArgShiftRuleQuadraticFunc(ArgShiftRuleMixin, conftest._QuadraticFuncT):
    @pytest.fixture
    def op_orig(self):
        from pycsou.operator.func import QuadraticFunc

        return QuadraticFunc(
            Q=tc_posdefop.CDO4(N=7),
            c=tc_linfunc.ScaledSum(N=7),
            t=1,
        )


class TestArgShiftRuleProxDiffFunc(ArgShiftRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            tc_proxdifffunc.SquaredL2Norm(M=7),
            tc_proxdifffunc.SquaredL2Norm(M=None),
            tc_linfunc.ScaledSum(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param
