# How AddRule tests work:
#
# * AddRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_lhs, op_rhs) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test a compound operator (via +), inherit from AddRuleMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.


import collections.abc as cabc

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


# LHS/RHS test operators ------------------------------------------------------
def op_map():
    import pycsou_tests.operator.examples.test_map as tc

    return tc.ReLU(M=7)


def op_func():
    import pycsou_tests.operator.examples.test_func as tc

    return tc.Median()


def op_diffmap():
    import pycsou_tests.operator.examples.test_diffmap as tc

    return tc.Sin(M=7)


def op_difffunc():
    import pycsou_tests.operator.examples.test_difffunc as tc

    return tc.SquaredL2Norm(M=7)


def op_proxfunc():
    import pycsou_tests.operator.examples.test_proxfunc as tc

    return tc.L1Norm(M=7)


def op_proxdifffunc():
    import pycsou_tests.operator.examples.test_proxdifffunc as tc

    return tc.SquaredL2Norm(M=7)


def op_quadraticfunc():
    from pycsou.operator.func import QuadraticFunc
    from pycsou_tests.operator.examples.test_linfunc import ScaledSum
    from pycsou_tests.operator.examples.test_posdefop import CDO4

    return QuadraticFunc(
        Q=CDO4(N=7),
        c=ScaledSum(N=7),
        t=1,
    )


def op_linop():
    import pycsou_tests.operator.examples.test_linop as tc

    return tc.Tile(N=7, M=1)


def op_linfunc():
    import pycsou_tests.operator.examples.test_linfunc as tc

    return tc.ScaledSum(N=7)


def op_squareop():
    import pycsou_tests.operator.examples.test_squareop as tc

    return tc.CumSum(N=7)


def op_normalop():
    import pycsou_tests.operator.examples.test_normalop as tc

    rng = np.random.default_rng(seed=2)
    h = rng.normal(size=(7,))
    return tc.CircularConvolution(h=h)


def op_unitop():
    import pycsou_tests.operator.examples.test_unitop as tc

    return tc.Permutation(N=7)


def op_selfadjointop():
    import pycsou_tests.operator.examples.test_selfadjointop as tc

    return tc.CDO2(N=7)


def op_posdefop():
    import pycsou_tests.operator.examples.test_posdefop as tc

    return tc.CDO4(N=7)


def op_projop():
    import pycsou_tests.operator.examples.test_projop as tc

    return tc.Oblique(N=7, alpha=np.pi / 4)


def op_orthprojop():
    import pycsou_tests.operator.examples.test_orthprojop as tc

    return tc.ScaleDown(N=7)


# Data Mixin ------------------------------------------------------------------
class AddRuleMixin:
    @pytest.fixture
    def op_lrhs(self) -> tuple[pyct.OpT, pyct.OpT]:
        # Override in inherited class with LHS/RHS operands.
        raise NotImplementedError

    @pytest.fixture
    def op_lhs(self, op_lrhs) -> pyct.OpT:
        return op_lrhs[0]

    @pytest.fixture
    def op_rhs(self, op_lrhs) -> pyct.OpT:
        return op_lrhs[1]

    @pytest.fixture(params=["lhs + rhs", "rhs + lhs"])
    def lrhs_options(self, request) -> bool:
        return request.param

    @pytest.fixture
    def op(self, op_lhs, op_rhs, lrhs_options) -> pyct.OpT:
        if lrhs_options == "lhs + rhs":
            return op_lhs + op_rhs
        elif lrhs_options == "rhs + lhs":
            return op_rhs + op_lhs
        else:
            raise NotImplementedError

    @pytest.fixture
    def data_shape(self, op_lhs, op_rhs) -> pyct.Shape:
        sh = pycu.infer_sum_shape(op_lhs.shape, op_rhs.shape)
        return sh

    @pytest.fixture
    def data_apply(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_lhs.apply(arr) + op_rhs.apply(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        arr = self._random_array((op.codim,), seed=20)  # random seed for reproducibility
        if op_lhs.has(pyca.Property.FUNCTIONAL) and (not op_rhs.has(pyca.Property.FUNCTIONAL)):
            # LHS broadcasts
            out_lhs = op_lhs.adjoint(arr.sum(keepdims=True))
            out_rhs = op_rhs.adjoint(arr)
        elif (not op_lhs.has(pyca.Property.FUNCTIONAL)) and op_rhs.has(pyca.Property.FUNCTIONAL):
            # RHS broadcasts
            out_lhs = op_lhs.adjoint(arr)
            out_rhs = op_rhs.adjoint(arr.sum(keepdims=True))
        else:
            out_lhs = op_lhs.adjoint(arr)
            out_rhs = op_rhs.adjoint(arr)

        out = out_lhs + out_rhs
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_lhs.grad(arr) + op_rhs.grad(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        tau = np.abs(self._random_array((1,), seed=21))[0]  # random seed for reproducibility

        if op_lhs.has(pyca.Property.LINEAR):
            P, G = op_rhs, op_lhs
        else:
            P, G = op_lhs, op_rhs
        out = P.prox(arr - tau * G.grad(arr), tau)

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


# Test classes (Maps) ---------------------------------------------------------
class TestAddRuleMap(AddRuleMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            (op_map(), op_map()),
            (op_map(), op_func()),
            (op_map(), op_diffmap()),
            (op_map(), op_difffunc()),
            (op_map(), op_proxfunc()),
            (op_map(), op_proxdifffunc()),
            (op_map(), op_quadraticfunc()),
            (op_map(), op_linop()),
            (op_map(), op_linfunc()),
            (op_map(), op_squareop()),
            (op_map(), op_normalop()),
            (op_map(), op_unitop()),
            (op_map(), op_selfadjointop()),
            (op_map(), op_posdefop()),
            (op_map(), op_projop()),
            (op_map(), op_orthprojop()),
            (op_func(), op_diffmap()),
            (op_func(), op_linop()),
            (op_func(), op_squareop()),
            (op_func(), op_normalop()),
            (op_func(), op_unitop()),
            (op_func(), op_selfadjointop()),
            (op_func(), op_posdefop()),
            (op_func(), op_projop()),
            (op_func(), op_orthprojop()),
            (op_diffmap(), op_proxfunc()),
            (op_proxfunc(), op_linop()),
            (op_proxfunc(), op_squareop()),
            (op_proxfunc(), op_normalop()),
            (op_proxfunc(), op_unitop()),
            (op_proxfunc(), op_selfadjointop()),
            (op_proxfunc(), op_posdefop()),
            (op_proxfunc(), op_projop()),
            (op_proxfunc(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleDiffMap(AddRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (op_diffmap(), op_diffmap()),
            (op_diffmap(), op_difffunc()),
            (op_diffmap(), op_proxdifffunc()),
            (op_diffmap(), op_quadraticfunc()),
            (op_diffmap(), op_linop()),
            (op_diffmap(), op_linfunc()),
            (op_diffmap(), op_squareop()),
            (op_diffmap(), op_normalop()),
            (op_diffmap(), op_unitop()),
            (op_diffmap(), op_selfadjointop()),
            (op_diffmap(), op_posdefop()),
            (op_diffmap(), op_projop()),
            (op_diffmap(), op_orthprojop()),
            (op_difffunc(), op_linop()),
            (op_difffunc(), op_squareop()),
            (op_difffunc(), op_normalop()),
            (op_difffunc(), op_unitop()),
            (op_difffunc(), op_selfadjointop()),
            (op_difffunc(), op_posdefop()),
            (op_difffunc(), op_projop()),
            (op_difffunc(), op_orthprojop()),
            (op_quadraticfunc(), op_linop()),
            (op_quadraticfunc(), op_squareop()),
            (op_quadraticfunc(), op_normalop()),
            (op_quadraticfunc(), op_unitop()),
            (op_quadraticfunc(), op_selfadjointop()),
            (op_quadraticfunc(), op_posdefop()),
            (op_quadraticfunc(), op_projop()),
            (op_quadraticfunc(), op_orthprojop()),
            (op_quadraticfunc(), op_linop()),
            (op_quadraticfunc(), op_squareop()),
            (op_quadraticfunc(), op_normalop()),
            (op_quadraticfunc(), op_unitop()),
            (op_quadraticfunc(), op_selfadjointop()),
            (op_quadraticfunc(), op_posdefop()),
            (op_quadraticfunc(), op_projop()),
            (op_quadraticfunc(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleLinOp(AddRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (op_linop(), op_linop()),
            (op_linop(), op_linfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleSquareOp(AddRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (op_linfunc(), op_squareop()),
            (op_linfunc(), op_normalop()),
            (op_linfunc(), op_unitop()),
            (op_linfunc(), op_selfadjointop()),
            (op_linfunc(), op_posdefop()),
            (op_linfunc(), op_projop()),
            (op_linfunc(), op_orthprojop()),
            (op_squareop(), op_squareop()),
            (op_squareop(), op_normalop()),
            (op_squareop(), op_unitop()),
            (op_squareop(), op_selfadjointop()),
            (op_squareop(), op_posdefop()),
            (op_squareop(), op_projop()),
            (op_squareop(), op_orthprojop()),
            (op_normalop(), op_normalop()),
            (op_normalop(), op_unitop()),
            (op_normalop(), op_selfadjointop()),
            (op_normalop(), op_posdefop()),
            (op_normalop(), op_projop()),
            (op_normalop(), op_orthprojop()),
            (op_unitop(), op_unitop()),
            (op_unitop(), op_selfadjointop()),
            (op_unitop(), op_posdefop()),
            (op_unitop(), op_projop()),
            (op_unitop(), op_orthprojop()),
            (op_selfadjointop(), op_projop()),
            (op_posdefop(), op_projop()),
            (op_projop(), op_projop()),
            (op_projop(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleNormalOp(AddRuleMixin, conftest.NormalOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestAddRuleUnitOp(AddRuleMixin, conftest.UnitOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestAddRuleSelfAdjointOp(AddRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(
        params=[
            (op_selfadjointop(), op_selfadjointop()),
            (op_selfadjointop(), op_posdefop()),
            (op_selfadjointop(), op_orthprojop()),
            (op_orthprojop(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRulePosDefOp(AddRuleMixin, conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (op_posdefop(), op_posdefop()),
            (op_posdefop(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleProjOp(AddRuleMixin, conftest.ProjOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestAddRuleOrthProjOp(AddRuleMixin, conftest.OrthProjOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestAddRuleFunc(AddRuleMixin, conftest.FuncT):
    @pytest.fixture(
        params=[
            (op_func(), op_func()),
            (op_func(), op_difffunc()),
            (op_func(), op_proxfunc()),
            (op_func(), op_proxdifffunc()),
            (op_func(), op_quadraticfunc()),
            (op_func(), op_linfunc()),
            (op_difffunc(), op_proxfunc()),
            (op_proxfunc(), op_proxfunc()),
            (op_proxfunc(), op_proxdifffunc()),
            (op_proxfunc(), op_quadraticfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleDiffFunc(AddRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            (op_difffunc(), op_difffunc()),
            (op_difffunc(), op_proxdifffunc()),
            (op_difffunc(), op_quadraticfunc()),
            (op_difffunc(), op_linfunc()),
            (op_proxdifffunc(), op_proxdifffunc()),
            (op_proxdifffunc(), op_quadraticfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleProxFunc(AddRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            (op_proxfunc(), op_linfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleProxDiffFunc(AddRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (op_proxdifffunc(), op_linfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleQuadraticFunc(AddRuleMixin, conftest._QuadraticFuncT):
    @pytest.fixture(
        params=[
            (op_quadraticfunc(), op_quadraticfunc()),
            (op_quadraticfunc(), op_linfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleLinFunc(AddRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (op_linfunc(), op_linfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param
