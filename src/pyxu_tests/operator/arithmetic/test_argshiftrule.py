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
import typing as typ

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest
import pyxu_tests.operator.examples.test_diffmap as tc_diffmap
import pyxu_tests.operator.examples.test_linfunc as tc_linfunc
import pyxu_tests.operator.examples.test_linop as tc_linop
import pyxu_tests.operator.examples.test_normalop as tc_normalop
import pyxu_tests.operator.examples.test_orthprojop as tc_orthprojop
import pyxu_tests.operator.examples.test_posdefop as tc_posdefop
import pyxu_tests.operator.examples.test_projop as tc_projop
import pyxu_tests.operator.examples.test_proxdifffunc as tc_proxdifffunc
import pyxu_tests.operator.examples.test_selfadjointop as tc_selfadjointop
import pyxu_tests.operator.examples.test_squareop as tc_squareop
import pyxu_tests.operator.examples.test_unitop as tc_unitop

rng = np.random.default_rng(seed=50)


class ArgShiftRuleMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be arg-shifted.
        raise NotImplementedError

    @pytest.fixture(params=[True, False])
    def op_shift(self, op_orig, request) -> typ.Union[pxt.Real, pxt.NDArray]:
        # Arg-shift values applied to op_orig()
        dim = self._sanitize(op_orig.dim, 7)
        cst = self._random_array((dim,), seed=20)  # random seed for reproducibility
        if request.param:  # scalar output
            cst = float(cst.sum())
        return cst

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, op_shift, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        if isinstance(op_shift, pxt.Real):
            shift = op_shift
        else:
            xp = ndi.module()
            shift = xp.array(op_shift, dtype=width.value)
        op = op_orig.argshift(shift)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_orig) -> pxt.OpShape:
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
    @pytest.mark.skip("undefined for argshift.")
    def test_interface_asloss(self, op, xp, width):
        pass


# Test classes (Maps) ---------------------------------------------------------
class TestArgShiftRuleMap(ArgShiftRuleMixin, conftest.MapT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(M=6)


class TestArgShiftRuleDiffMap(ArgShiftRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            tc_diffmap.Sin(M=6),
            tc_linop.Tile(N=3, M=4),
            tc_squareop.CumSum(N=7),
            tc_normalop.CircularConvolution(h=rng.normal(size=(5,))),
            tc_unitop.Permutation(N=7),
            tc_selfadjointop.SelfAdjointConvolution(N=7),
            tc_posdefop.PSDConvolution(N=7),
            tc_projop.Oblique(N=6, alpha=np.pi / 4),
            tc_orthprojop.ScaleDown(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestArgShiftRuleFunc(ArgShiftRuleMixin, conftest.FuncT):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_func as tc

        return tc.Median(dim=request.param)


class TestArgShiftRuleDiffFunc(ArgShiftRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=request.param)


class TestArgShiftRuleProxFunc(ArgShiftRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=request.param)


class TestArgShiftRuleQuadraticFunc(ArgShiftRuleMixin, conftest.QuadraticFuncT):
    @pytest.fixture(params=[0, 1])
    def op_orig(self, request):
        from pyxu_tests.operator.examples.test_linfunc import ScaledSum
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        N = 7
        op = {
            0: pxa.QuadraticFunc(shape=(1, N)),
            1: pxa.QuadraticFunc(
                shape=(1, N),
                Q=PSDConvolution(N=N),
                c=ScaledSum(N=N),
                t=1,
            ),
        }[request.param]
        return op


class TestArgShiftRuleProxDiffFunc(ArgShiftRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            tc_proxdifffunc.SquaredL2Norm(M=7),
            tc_linfunc.ScaledSum(N=7),
        ]
    )
    def op_orig(self, request):
        return request.param
