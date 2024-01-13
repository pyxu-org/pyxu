# How ArgShiftRule tests work:
#
# * ArgShiftRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR^{M1,...,MD}.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.
#            (True if choosing test operators from examples/.)                ]
#
# * To test an arg-shifted-operator, inherit from ArgShiftRuleMixin and the suitable MapT subclass
#   which the arg-shifted operator should abide by.
#
# Important: argshifted-operators are not module/precision-agnostic!

import collections.abc as cabc
import itertools

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

rng = np.random.default_rng()


class ArgShiftRuleMixin:
    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be arg-shifted.
        raise NotImplementedError

    @pytest.fixture(
        params=[
            "bcast",
            "full",
        ]
    )
    def op_shift(self, op_orig, request) -> pxt.NDArray:
        # Arg-shift values applied to op_orig().
        # [NUMPY arrays only; other fixtures/tests should cast to other backends if required.]
        cst_type = request.param

        cst = rng.standard_normal(size=op_orig.dim_shape)
        if cst_type == "full":
            pass
        elif cst_type == "bcast":
            axis = rng.integers(0, op_orig.dim_rank)
            cst = cst.sum(axis=axis, keepdims=True)
        else:
            raise NotImplementedError
        return cst

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, op_shift, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        shift = xp.array(op_shift, dtype=width.value)

        op = op_orig.argshift(shift)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.dim_shape

    @pytest.fixture
    def codim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.codim_shape

    @pytest.fixture
    def data_apply(self, op_orig, op_shift) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.apply(x + op_shift)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, op_orig, op_shift) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.grad(x + op_shift)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, op_orig, op_shift) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        tau = abs(self._random_array((1,)).item()) + 1e-2
        y = op_orig.prox(x + op_shift, tau) - op_shift

        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 20
        x = self._random_array((N_test, *op.dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 20
        x = self._random_array((N_test, *op.dim_shape))
        return x


# Test classes (Maps) ---------------------------------------------------------
class TestArgShiftRuleMap(ArgShiftRuleMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_map as tc

        dim_shape = request.param
        return tc.ReLU(dim_shape=dim_shape)


class TestArgShiftRuleDiffMap(ArgShiftRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            tc_diffmap.Sin(dim_shape=(5,)),
            tc_diffmap.Sin(dim_shape=(5, 3, 4)),
            tc_linop.Sum(dim_shape=(5,)),
            tc_linop.Sum(dim_shape=(5, 3, 4)),
            tc_squareop.CumSum(dim_shape=(5,)),
            tc_squareop.CumSum(dim_shape=(5, 3, 4)),
            tc_normalop.CircularConvolution(dim_shape=(5,), h=rng.standard_normal((5,))),
            tc_normalop.CircularConvolution(dim_shape=(5, 3, 5), h=rng.standard_normal((5,))),
            tc_unitop.Permutation(dim_shape=(5,)),
            tc_unitop.Permutation(dim_shape=(5, 3, 4)),
            tc_selfadjointop.SelfAdjointConvolution(dim_shape=(5,)),
            tc_selfadjointop.SelfAdjointConvolution(dim_shape=(5, 3, 5)),
            tc_posdefop.PSDConvolution(dim_shape=(5,)),
            tc_posdefop.PSDConvolution(dim_shape=(5, 3, 5)),
            tc_projop.Oblique(dim_shape=(5,), alpha=np.pi / 4),
            tc_projop.Oblique(dim_shape=(5, 3, 4), alpha=np.pi / 4),
            tc_orthprojop.ScaleDown(dim_shape=(5,)),
            tc_orthprojop.ScaleDown(dim_shape=(5, 3, 4)),
        ]
    )
    def op_orig(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestArgShiftRuleFunc(ArgShiftRuleMixin, conftest.FuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_func as tc

        dim_shape = request.param
        return tc.Median(dim_shape=dim_shape)


class TestArgShiftRuleDiffFunc(ArgShiftRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_difffunc as tc

        dim_shape = request.param
        return tc.SquaredL2Norm(dim_shape=dim_shape)


class TestArgShiftRuleProxFunc(ArgShiftRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_proxfunc as tc

        dim_shape = request.param
        return tc.L1Norm(dim_shape=dim_shape)


class TestArgShiftRuleQuadraticFunc(ArgShiftRuleMixin, conftest.QuadraticFuncT):
    @pytest.fixture(
        params=[
            ("default", (5,)),
            ("default", (5, 3, 5)),
            ("explicit", (5,)),
            ("explicit", (5, 3, 5)),
        ]
    )
    def op_orig(self, request):
        from pyxu_tests.operator.examples.test_linfunc import Sum
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        init_type, dim_shape = request.param
        if init_type == "default":
            op = pxa.QuadraticFunc(
                dim_shape=dim_shape,
                codim_shape=1,
            )
        else:  # "explicit"
            op = pxa.QuadraticFunc(
                dim_shape=dim_shape,
                codim_shape=1,
                Q=PSDConvolution(dim_shape=dim_shape),
                c=Sum(dim_shape=dim_shape),
                t=1,
            )
        return op


class TestArgShiftRuleProxDiffFunc(ArgShiftRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            tc_proxdifffunc.SquaredL2Norm(dim_shape=(5,)),
            tc_proxdifffunc.SquaredL2Norm(dim_shape=(5, 3, 4)),
            tc_linfunc.Sum(dim_shape=(5,)),
            tc_linfunc.Sum(dim_shape=(5, 3, 4)),
        ]
    )
    def op_orig(self, request):
        return request.param
