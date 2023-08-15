# How ScaleRule tests work:
#
# * ScaleRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test a scaled-operator, inherit from ScaleRuleMixin and the suitable conftest.MapT subclass
#   which the scaled operator should abide by.


import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest

op_scale_params = frozenset({-2.3, -1, 1, 2})
op_scale_unit = frozenset(_ for _ in op_scale_params if np.isclose(np.abs(_), 1))
op_scale_nonunit = frozenset(_ for _ in op_scale_params if ~np.isclose(np.abs(_), 1))
op_scale_positive = frozenset(_ for _ in op_scale_params if _ > 0)
op_scale_negative = frozenset(_ for _ in op_scale_params if _ < 0)
op_scale_nonidentity = frozenset(_ for _ in op_scale_params if ~np.isclose(_, 1))


class ScaleRuleMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be scaled.
        raise NotImplementedError

    @pytest.fixture(params=op_scale_params)
    def op_scale(self, request) -> pxt.Real:
        # Scaling factors applied to op_orig()
        return request.param

    @pytest.fixture(
        params=itertools.product(
            [
                "scale_left",
                "scale_right",
                "divide_right",
            ],
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, op_scale, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        mode, ndi, width = request.param
        if mode == "scale_left":
            op = op_scale * op_orig
        elif mode == "scale_right":
            op = op_orig * op_scale
        else:  # divide_right
            op = op_orig / (1 / op_scale)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_orig) -> pxt.OpShape:
        return op_orig.shape

    @pytest.fixture
    def data_apply(self, op_orig, op_scale) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_orig.apply(arr) * op_scale
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op_orig, op_scale) -> conftest.DataLike:
        codim = self._sanitize(op_orig.codim, 7)
        arr = self._random_array((codim,), seed=20)  # random seed for reproducibility
        out = op_orig.adjoint(arr) * op_scale
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op_orig, op_scale) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_orig.grad(arr) * op_scale
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op_orig, op_scale) -> conftest.DataLike:
        dim = self._sanitize(op_orig.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        tau = np.abs(self._random_array((1,), seed=21))[0]  # random seed for reproducibility
        out = op_orig.prox(arr, tau * op_scale)
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
        if not op_orig.has(pxa.Property.FUNCTIONAL):
            pytest.skip("asloss() unavailable for non-functionals.")

        try:
            op_orig.asloss()  # detect if fails
            super().test_interface_asloss(op, xp, width)
        except NotImplementedError:
            pytest.skip("asloss() unsupported by base operator.")


# Test classes (Maps) ---------------------------------------------------------
class TestScaleRuleMap(ScaleRuleMixin, conftest.MapT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(M=6)


class TestScaleRuleDiffMap(ScaleRuleMixin, conftest.DiffMapT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_diffmap as tc

        return tc.Sin(M=6)


class TestScaleRuleLinOp(ScaleRuleMixin, conftest.LinOpT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Tile(N=3, M=4)


class TestScaleRuleSquareOp(ScaleRuleMixin, conftest.SquareOpT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(N=7)


class TestScaleRuleNormalOp(ScaleRuleMixin, conftest.NormalOpT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_normalop as tc

        h = self._random_array((5,), seed=2)
        return tc.CircularConvolution(h=h)


# START UnitOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleUnitOp(ScaleRuleMixin):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(N=7)


@pytest.mark.parametrize("op_scale", op_scale_unit)
class TestScaleRuleUnitOp_UnitScale(ScaleRuleUnitOp, conftest.UnitOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonunit)
class TestScaleRuleUnitOp_NonUnitScale(ScaleRuleUnitOp, conftest.NormalOpT):
    pass


# END UnitOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class TestScaleRuleSelfAdjointOp(ScaleRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(N=7)


# START PosDefOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRulePosDefOp(ScaleRuleMixin):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(N=7)


@pytest.mark.parametrize("op_scale", op_scale_positive)
class TestScaleRulePosDefOp_PositiveScale(ScaleRulePosDefOp, conftest.PosDefOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_negative)
class TestScaleRulePosDefOp_NegativeScale(ScaleRulePosDefOp, conftest.SelfAdjointOpT):
    pass


# END PosDefOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START ProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleProjOp(ScaleRuleMixin):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(N=7, alpha=0.3)


@pytest.mark.parametrize("op_scale", [1])
class TestScaleRuleProjOp_IdentityScale(ScaleRuleProjOp, conftest.ProjOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonidentity)
class TestScaleRuleProjOp_NonIdentityScale(ScaleRuleProjOp, conftest.SquareOpT):
    pass


# END ProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START OrthProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleOrthProjOp(ScaleRuleMixin):
    @pytest.fixture
    def op_orig(self):
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(N=7)


@pytest.mark.parametrize("op_scale", [1])
class TestScaleRuleOrthProjOp_IdentityScale(ScaleRuleOrthProjOp, conftest.OrthProjOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonidentity)
class TestScaleRuleOrthProjOp_NonIdentityScale(ScaleRuleOrthProjOp, conftest.SelfAdjointOpT):
    pass


# END OrthProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Test classes (Funcs) --------------------------------------------------------
class TestScaleRuleFunc(ScaleRuleMixin, conftest.FuncT):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_func as tc

        return tc.Median(dim=request.param)


class TestScaleRuleDiffFunc(ScaleRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=request.param)


# START ProxFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleProxFunc(ScaleRuleMixin):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=request.param)


@pytest.mark.parametrize("op_scale", op_scale_positive)
class TestScaleRuleProxFunc_PositiveScale(ScaleRuleProxFunc, conftest.ProxFuncT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_negative)
class TestScaleRuleProxFunc_NegativeScale(ScaleRuleProxFunc, conftest.FuncT):
    pass


# END ProxFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START ProxDiffFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleProxDiffFunc(ScaleRuleMixin):
    @pytest.fixture(params=[7])
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_proxdifffunc as tc

        return tc.SquaredL2Norm(M=request.param)


@pytest.mark.parametrize("op_scale", op_scale_positive)
class TestScaleRuleProxDiffFunc_PositiveScale(ScaleRuleProxDiffFunc, conftest.ProxDiffFuncT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_negative)
class TestScaleRuleProxDiffFunc_NegativeScale(ScaleRuleProxDiffFunc, conftest.DiffFuncT):
    pass


# END ProxDiffFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START QuadraticFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ScaleRuleQuadraticFunc(ScaleRuleMixin):
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


@pytest.mark.parametrize("op_scale", op_scale_positive)
class TestScaleRuleQuadraticFunc_PositiveScale(ScaleRuleQuadraticFunc, conftest.QuadraticFuncT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_negative)
class TestScaleRuleQuadraticFunc_NegativeScale(ScaleRuleQuadraticFunc, conftest.DiffFuncT):
    pass


# END QuadraticFunc ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class TestScaleRuleLinFunc(ScaleRuleMixin, conftest.LinFuncT):
    @pytest.fixture()
    def op_orig(self):
        import pyxu_tests.operator.examples.test_linfunc as tc

        return tc.ScaledSum(N=7)
