# How ArgScaleRule tests work:
#
# * ArgScaleRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR^{M1,...,MD}.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.
#            (True if choosing test operators from examples/.)                ]
#
# * To test an arg-scaled-operator, inherit from ArgScaleRuleMixin and the suitable conftest.MapT
#   subclass which the arg-scaled operator should abide by.


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


class ArgScaleRuleMixin:
    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be arg-scaled.
        raise NotImplementedError

    @pytest.fixture(params=op_scale_params)
    def op_scale(self, request) -> pxt.Real:
        # Arg-scale factors applied to op_orig()
        return request.param

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, op_scale, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = op_orig.argscale(op_scale)
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.dim_shape

    @pytest.fixture
    def codim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.codim_shape

    @pytest.fixture
    def data_apply(self, op_orig, op_scale) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.apply(x * op_scale)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, op_orig, op_scale) -> conftest.DataLike:
        x = self._random_array(op_orig.codim_shape)
        y = op_orig.adjoint(x) * op_scale

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, op_orig, op_scale) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.grad(x * op_scale) * op_scale

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, op_orig, op_scale) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        tau = abs(self._random_array((1,)).item()) + 1e-2
        y = op_orig.prox(x * op_scale, tau * (op_scale**2)) / op_scale

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
class TestArgScaleRuleMap(ArgScaleRuleMixin, conftest.MapT):
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


class TestArgScaleRuleDiffMap(ArgScaleRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_diffmap as tc

        dim_shape = request.param
        return tc.Sin(dim_shape=dim_shape)


class TestArgScaleRuleLinOp(ArgScaleRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_linop as tc

        dim_shape = request.param
        return tc.Sum(dim_shape=dim_shape)


class TestArgScaleRuleSquareOp(ArgScaleRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_squareop as tc

        dim_shape = request.param
        return tc.CumSum(dim_shape=dim_shape)


class TestArgScaleRuleNormalOp(ArgScaleRuleMixin, conftest.NormalOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_normalop as tc

        dim_shape = request.param
        conv_filter = self._random_array(dim_shape[-1], seed=2)
        return tc.CircularConvolution(
            dim_shape=dim_shape,
            h=conv_filter,
        )


# START UnitOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ArgScaleRuleUnitOp(ArgScaleRuleMixin):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_unitop as tc

        dim_shape = request.param
        return tc.Permutation(dim_shape=dim_shape)


@pytest.mark.parametrize("op_scale", op_scale_unit)
class TestArgScaleRuleUnitOp_UnitScale(ArgScaleRuleUnitOp, conftest.UnitOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonunit)
class TestArgScaleRuleUnitOp_NonUnitScale(ArgScaleRuleUnitOp, conftest.NormalOpT):
    pass


# END UnitOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class TestArgScaleRuleSelfAdjointOp(ArgScaleRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        dim_shape = request.param
        return tc.SelfAdjointConvolution(dim_shape=dim_shape)


# START PosDefOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ArgScaleRulePosDefOp(ArgScaleRuleMixin):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_posdefop as tc

        dim_shape = request.param
        return tc.PSDConvolution(dim_shape=dim_shape)


@pytest.mark.parametrize("op_scale", op_scale_positive)
class TestArgScaleRulePosDefOp_PositiveScale(ArgScaleRulePosDefOp, conftest.PosDefOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_negative)
class TestArgScaleRulePosDefOp_NegativeScale(ArgScaleRulePosDefOp, conftest.SelfAdjointOpT):
    pass


# END PosDefOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START ProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ArgScaleRuleProjOp(ArgScaleRuleMixin):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_projop as tc

        dim_shape = request.param
        return tc.Oblique(
            dim_shape=dim_shape,
            alpha=np.pi / 4,
        )


@pytest.mark.parametrize("op_scale", [1])
class TestArgScaleRuleProjOp_IdentityScale(ArgScaleRuleProjOp, conftest.ProjOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonidentity)
class TestArgScaleRuleProjOp_NonIdentityScale(ArgScaleRuleProjOp, conftest.SquareOpT):
    pass


# END ProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# START OrthProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class ArgScaleRuleOrthProjOp(ArgScaleRuleMixin):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_orthprojop as tc

        dim_shape = request.param
        return tc.ScaleDown(dim_shape=dim_shape)


@pytest.mark.parametrize("op_scale", [1])
class TestArgScaleRuleOrthProjOp_IdentityScale(ArgScaleRuleOrthProjOp, conftest.OrthProjOpT):
    pass


@pytest.mark.parametrize("op_scale", op_scale_nonidentity)
class TestArgScaleRuleOrthProjOp_NonIdentityScale(ArgScaleRuleOrthProjOp, conftest.SelfAdjointOpT):
    pass


# END OrthProjOp ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Test classes (Funcs) --------------------------------------------------------
class TestArgScaleRuleFunc(ArgScaleRuleMixin, conftest.FuncT):
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


class TestArgScaleRuleDiffFunc(ArgScaleRuleMixin, conftest.DiffFuncT):
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


class TestArgScaleRuleProxFunc(ArgScaleRuleMixin, conftest.ProxFuncT):
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


class TestArgScaleRuleProxDiffFunc(ArgScaleRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_proxdifffunc as tc

        dim_shape = request.param
        return tc.SquaredL2Norm(dim_shape=dim_shape)


class TestArgScaleRuleQuadraticFunc(ArgScaleRuleMixin, conftest.QuadraticFuncT):
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


class TestArgScaleRuleLinFunc(ArgScaleRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        import pyxu_tests.operator.examples.test_linfunc as tc

        dim_shape = request.param
        return tc.Sum(dim_shape=dim_shape)
