# How ChainRule tests work:
#
# * ChainRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_lhs, op_rhs) are correctly implemented.] (True if choosing test operators from examples/.)
#
# * To test a compound operator (via *), inherit from ChainRuleMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.


import collections.abc as cabc
import itertools
import warnings

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest

# It is expected for DenseWarning to be raised when creating some operators, or fallback matrix ops.
pytestmark = pytest.mark.filterwarnings("ignore::pyxu.info.warning.DenseWarning")


# LHS/RHS test operators ------------------------------------------------------
def op_map(dim: int = 7):
    import pyxu_tests.operator.examples.test_map as tc

    return tc.ReLU(M=dim)


def op_func(dim: int = 7):
    import pyxu_tests.operator.examples.test_func as tc

    return tc.Median(dim=dim)


def op_diffmap(dim: int = 7):
    import pyxu_tests.operator.examples.test_diffmap as tc

    return tc.Sin(M=dim)


def op_difffunc(dim: int = 7):
    import pyxu_tests.operator.examples.test_difffunc as tc

    return tc.SquaredL2Norm(M=dim)


def op_proxfunc(dim: int = 7):
    import pyxu_tests.operator.examples.test_proxfunc as tc

    return tc.L1Norm(M=dim)


def op_proxdifffunc(dim: int = 7):
    import pyxu_tests.operator.examples.test_proxdifffunc as tc

    return tc.SquaredL2Norm(M=dim)


def op_quadraticfunc(dim: int = 7):
    # QuadraticFunc may be defined for dim=1.
    # In this case we cannot use CD04 (examples/test_posdefop.py) due to minimal domain-size restrictions.
    # We therefore use HomothetyOp without loss of generality.

    from pyxu.operator.linop import HomothetyOp
    from pyxu_tests.operator.examples.test_linfunc import ScaledSum

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pxw.DenseWarning)
        return pxa.QuadraticFunc(
            shape=(1, dim),
            Q=HomothetyOp(dim=dim, cst=3),
            c=ScaledSum(N=dim),
            t=1,
        )


def op_linop(dim: int = 7, codim_scale: int = 1):
    import pyxu_tests.operator.examples.test_linop as tc

    return tc.Tile(N=dim, M=codim_scale)


def op_linfunc(dim: int = 7, positive: bool = False):
    import pyxu_tests.operator.examples.test_linfunc as tc

    op = tc.ScaledSum(N=dim)
    if not positive:
        op = -op
    return op


def op_squareop():
    import pyxu_tests.operator.examples.test_squareop as tc

    return tc.CumSum(N=7)


def op_normalop():
    import pyxu_tests.operator.examples.test_normalop as tc

    rng = np.random.default_rng(seed=2)
    h = rng.normal(size=(7,))
    return tc.CircularConvolution(h=h)


def op_unitop():
    import pyxu_tests.operator.examples.test_unitop as tc

    return tc.Permutation(N=7)


def op_selfadjointop():
    import pyxu_tests.operator.examples.test_selfadjointop as tc

    return tc.SelfAdjointConvolution(N=7)


def op_posdefop():
    import pyxu_tests.operator.examples.test_posdefop as tc

    return tc.PSDConvolution(N=7)


def op_projop():
    import pyxu_tests.operator.examples.test_projop as tc

    return tc.Oblique(N=7, alpha=np.pi / 4)


def op_orthprojop():
    import pyxu_tests.operator.examples.test_orthprojop as tc

    return tc.ScaleDown(N=7)


# Data Mixin ------------------------------------------------------------------
class ChainRuleMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_lrhs(self) -> tuple[pxt.OpT, pxt.OpT]:
        # Override in inherited class with LHS/RHS operands.
        raise NotImplementedError

    @pytest.fixture
    def op_lhs(self, op_lrhs) -> pxt.OpT:
        return op_lrhs[0]

    @pytest.fixture
    def op_rhs(self, op_lrhs) -> pxt.OpT:
        return op_lrhs[1]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_lhs, op_rhs, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = op_lhs * op_rhs
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_lhs, op_rhs) -> pxt.OpShape:
        sh = pxu.infer_composition_shape(op_lhs.shape, op_rhs.shape)
        return sh

    @pytest.fixture
    def data_apply(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_lhs.apply(op_rhs.apply(arr))
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        arr = self._random_array((op.codim,), seed=20)  # random seed for reproducibility
        out = op_rhs.adjoint(op_lhs.adjoint(arr))
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        out = op_lhs.grad(op_rhs.apply(arr)) @ op_rhs.jacobian(arr).asarray()
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        dim = self._sanitize(op.dim, 7)
        arr = self._random_array((dim,), seed=20)  # random seed for reproducibility
        tau = np.abs(self._random_array((1,), seed=21))[0]  # random seed for reproducibility

        if op.has(pxa.Property.PROXIMABLE):
            out = None
            if op_lhs.has(pxa.Property.PROXIMABLE) and op_rhs.has(pxa.Property.LINEAR_UNITARY):
                # prox \comp unitary
                out = op_rhs.adjoint(op_lhs.prox(op_rhs.apply(arr), tau))
            elif op.has(pxa.Property.LINEAR):
                # linfunc \comp lin[op|func]
                out = arr - tau * (op_lhs.asarray() @ op_rhs.asarray()).flatten()
            elif op_lhs.has(pxa.Property.LINEAR) and op_rhs.has(pxa.Property.PROXIMABLE):
                # linfunc \comp [prox, proxdiff, quadratic]
                cst = op_lhs.asarray().item()
                out = op_rhs.prox(arr, cst * tau)
            elif op_lhs.has(pxa.Property.QUADRATIC) and op_rhs.has(pxa.Property.LINEAR):
                # quadratic \comp linop
                A = op_rhs.asarray()
                B = op_lhs._quad_spec()[0].asarray()
                Q = tau * (A.T @ B @ A) + np.eye(op_rhs.dim)
                b = arr - tau * (A.T @ op_lhs.grad(np.zeros(op_lhs.dim)))
                out, *_ = splinalg.lstsq(Q, b)

            if out is not None:
                return dict(
                    in_=dict(
                        arr=arr,
                        tau=tau,
                    ),
                    out=out,
                )
        raise NotImplementedError

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, self._sanitize(op.dim, 7)
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, self._sanitize(op.dim, 7)
        return self._random_array((N_test, dim))

    # Tests -------------------------------------------------------------------
    @pytest.mark.skip("undefined for composition.")
    def test_interface_asloss(self, op_lhs):
        pass


# Test classes (Maps) ---------------------------------------------------------
class TestChainRuleMap(ChainRuleMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            (op_map(), op_map()),
            (op_map(dim=1), op_func()),
            (op_map(), op_diffmap()),
            (op_map(dim=1), op_difffunc()),
            (op_map(dim=1), op_proxfunc()),
            (op_map(dim=1), op_proxdifffunc()),
            (op_map(dim=1), op_quadraticfunc()),
            (op_map(), op_linop()),
            (op_map(dim=1), op_linfunc()),
            (op_map(), op_squareop()),
            (op_map(), op_normalop()),
            (op_map(), op_unitop()),
            (op_map(), op_selfadjointop()),
            (op_map(), op_posdefop()),
            (op_map(), op_projop()),
            (op_map(), op_orthprojop()),
            (op_diffmap(), op_map()),
            (op_diffmap(dim=1), op_func()),
            (op_diffmap(dim=1), op_proxfunc()),
            (op_linop(), op_map()),
            (op_linop(dim=1, codim_scale=5), op_proxfunc()),
            (op_squareop(), op_map()),
            (op_normalop(), op_map()),
            (op_unitop(), op_map()),
            (op_selfadjointop(), op_map()),
            (op_posdefop(), op_map()),
            (op_projop(), op_map()),
            (op_orthprojop(), op_map()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleDiffMap(ChainRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (op_diffmap(), op_diffmap()),
            (op_diffmap(dim=1), op_difffunc()),
            (op_diffmap(dim=1), op_proxdifffunc()),
            (op_diffmap(dim=1), op_quadraticfunc()),
            (op_diffmap(), op_linop()),
            (op_diffmap(dim=1), op_linfunc()),
            (op_diffmap(), op_squareop()),
            (op_diffmap(), op_normalop()),
            (op_diffmap(), op_unitop()),
            (op_diffmap(), op_selfadjointop()),
            (op_diffmap(), op_posdefop()),
            (op_diffmap(), op_projop()),
            (op_diffmap(), op_orthprojop()),
            (op_linop(), op_diffmap()),
            (op_linop(dim=1, codim_scale=5), op_difffunc()),
            (op_linop(dim=1, codim_scale=5), op_proxdifffunc()),
            (op_linop(dim=1, codim_scale=5), op_quadraticfunc()),
            (op_squareop(), op_diffmap()),
            (op_normalop(), op_diffmap()),
            (op_unitop(), op_diffmap()),
            (op_selfadjointop(), op_diffmap()),
            (op_posdefop(), op_diffmap()),
            (op_projop(), op_diffmap()),
            (op_orthprojop(), op_diffmap()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleLinOp(ChainRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (op_linop(), op_linop()),  # may return SquareOp, but OK
            (op_linop(dim=8, codim_scale=2), op_linop(dim=2, codim_scale=4)),
            (op_linop(dim=1, codim_scale=5), op_linfunc()),
            (op_linop(), op_squareop()),
            (op_linop(), op_normalop()),
            (op_linop(), op_unitop()),
            (op_linop(), op_selfadjointop()),
            (op_linop(), op_posdefop()),
            (op_linop(), op_projop()),
            (op_linop(), op_orthprojop()),
            (op_squareop(), op_linop()),
            (op_normalop(), op_linop()),
            (op_unitop(), op_linop()),
            (op_selfadjointop(), op_linop()),
            (op_posdefop(), op_linop()),
            (op_projop(), op_linop()),
            (op_orthprojop(), op_linop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleSquareOp(ChainRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (op_linop(), op_linop()),  # shapes must be chosen to become square.
            (op_squareop(), op_squareop()),
            (op_squareop(), op_normalop()),
            (op_squareop(), op_unitop()),
            (op_squareop(), op_selfadjointop()),
            (op_squareop(), op_posdefop()),
            (op_squareop(), op_projop()),
            (op_squareop(), op_orthprojop()),
            (op_normalop(), op_squareop()),
            (op_normalop(), op_normalop()),
            (op_normalop(), op_unitop()),
            (op_normalop(), op_selfadjointop()),
            (op_normalop(), op_posdefop()),
            (op_normalop(), op_projop()),
            (op_normalop(), op_orthprojop()),
            (op_unitop(), op_squareop()),
            (op_unitop(), op_normalop()),
            (op_unitop(), op_selfadjointop()),
            (op_unitop(), op_posdefop()),
            (op_unitop(), op_projop()),
            (op_unitop(), op_orthprojop()),
            (op_selfadjointop(), op_squareop()),
            (op_selfadjointop(), op_normalop()),
            (op_selfadjointop(), op_unitop()),
            (op_selfadjointop(), op_selfadjointop()),
            (op_selfadjointop(), op_posdefop()),
            (op_selfadjointop(), op_projop()),
            (op_selfadjointop(), op_orthprojop()),
            (op_posdefop(), op_squareop()),
            (op_posdefop(), op_normalop()),
            (op_posdefop(), op_unitop()),
            (op_posdefop(), op_selfadjointop()),
            (op_posdefop(), op_posdefop()),
            (op_posdefop(), op_projop()),
            (op_posdefop(), op_orthprojop()),
            (op_projop(), op_squareop()),
            (op_projop(), op_normalop()),
            (op_projop(), op_unitop()),
            (op_projop(), op_selfadjointop()),
            (op_projop(), op_posdefop()),
            (op_projop(), op_projop()),
            (op_projop(), op_orthprojop()),
            (op_orthprojop(), op_squareop()),
            (op_orthprojop(), op_normalop()),
            (op_orthprojop(), op_unitop()),
            (op_orthprojop(), op_selfadjointop()),
            (op_orthprojop(), op_posdefop()),
            (op_orthprojop(), op_projop()),
            (op_orthprojop(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleNormalOp(ChainRuleMixin, conftest.NormalOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestChainRuleUnitOp(ChainRuleMixin, conftest.UnitOpT):
    @pytest.fixture(
        params=[
            (op_unitop(), op_unitop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleSelfAdjointOp(ChainRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestChainRulePosDefOp(ChainRuleMixin, conftest.PosDefOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestChainRuleProjOp(ChainRuleMixin, conftest.ProjOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


class TestChainRuleOrthProjOp(ChainRuleMixin, conftest.OrthProjOpT):
    @pytest.fixture(params=[])
    def op_lrhs(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestChainRuleFunc(ChainRuleMixin, conftest.FuncT):
    @pytest.fixture(
        params=[
            (op_func(), op_map()),
            (op_func(dim=1), op_func()),
            (op_func(), op_diffmap()),
            (op_func(dim=1), op_difffunc()),
            (op_func(dim=1), op_proxfunc()),
            (op_func(dim=1), op_proxdifffunc()),
            (op_func(dim=1), op_quadraticfunc()),
            (op_func(), op_linop()),
            (op_func(dim=1), op_linfunc()),
            (op_func(), op_squareop()),
            (op_func(), op_normalop()),
            (op_func(), op_unitop()),
            (op_func(), op_selfadjointop()),
            (op_func(), op_posdefop()),
            (op_func(), op_projop()),
            (op_func(), op_orthprojop()),
            (op_difffunc(), op_map()),
            (op_proxfunc(), op_map()),
            (op_proxfunc(), op_diffmap()),
            (op_proxfunc(), op_linop()),
            (op_proxfunc(), op_squareop()),
            (op_proxfunc(), op_normalop()),
            (op_proxfunc(), op_selfadjointop()),
            (op_proxfunc(), op_posdefop()),
            (op_proxfunc(), op_projop()),
            (op_proxfunc(), op_orthprojop()),
            (op_proxdifffunc(), op_map()),
            (op_quadraticfunc(), op_map()),
            (op_quadraticfunc(dim=1), op_func()),
            (op_quadraticfunc(dim=1), op_proxfunc()),
            (op_linop(dim=1, codim_scale=1), op_func()),
            (op_linfunc(), op_map()),
            (op_linfunc(dim=1), op_func()),
            (op_linfunc(dim=1, positive=False), op_proxfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleDiffFunc(ChainRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            (op_difffunc(), op_diffmap()),
            (op_difffunc(), op_linop()),
            (op_difffunc(), op_squareop()),
            (op_difffunc(), op_normalop()),
            (op_difffunc(), op_unitop()),
            (op_difffunc(), op_selfadjointop()),
            (op_difffunc(), op_posdefop()),
            (op_difffunc(), op_projop()),
            (op_difffunc(), op_orthprojop()),
            (op_proxdifffunc(), op_diffmap()),
            (op_proxdifffunc(), op_linop()),
            (op_proxdifffunc(), op_squareop()),
            (op_proxdifffunc(), op_normalop()),
            (op_proxdifffunc(), op_selfadjointop()),
            (op_proxdifffunc(), op_posdefop()),
            (op_proxdifffunc(), op_projop()),
            (op_proxdifffunc(), op_orthprojop()),
            (op_quadraticfunc(), op_diffmap()),
            (op_quadraticfunc(dim=1), op_difffunc()),
            (op_quadraticfunc(dim=1), op_proxdifffunc()),
            (op_quadraticfunc(dim=1), op_quadraticfunc()),
            (op_linfunc(), op_diffmap()),
            (op_linfunc(dim=1), op_difffunc()),
            (op_linfunc(dim=1), op_proxdifffunc()),
            (op_linfunc(dim=1), op_quadraticfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleProxFunc(ChainRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            (op_proxfunc(), op_unitop()),
            (op_linfunc(dim=1, positive=True), op_proxfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleProxDiffFunc(ChainRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (op_proxdifffunc(), op_unitop()),
            (op_linfunc(dim=1, positive=True), op_proxdifffunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleQuadraticFunc(ChainRuleMixin, conftest.QuadraticFuncT):
    @pytest.fixture(
        params=[
            (op_quadraticfunc(dim=6), op_linop(dim=2, codim_scale=3)),
            (op_quadraticfunc(dim=1), op_linfunc()),
            (op_quadraticfunc(), op_squareop()),
            (op_quadraticfunc(), op_normalop()),
            (op_quadraticfunc(), op_unitop()),
            (op_quadraticfunc(), op_selfadjointop()),
            (op_quadraticfunc(), op_posdefop()),
            (op_quadraticfunc(), op_projop()),
            (op_quadraticfunc(), op_orthprojop()),
            (op_linfunc(dim=1, positive=True), op_quadraticfunc()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleLinFunc(ChainRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (op_linfunc(), op_linop()),
            (op_linfunc(dim=8), op_linop(dim=2, codim_scale=4)),
            (op_linfunc(dim=1), op_linfunc()),
            (op_linfunc(), op_squareop()),
            (op_linfunc(), op_normalop()),
            (op_linfunc(), op_unitop()),
            (op_linfunc(), op_selfadjointop()),
            (op_linfunc(), op_posdefop()),
            (op_linfunc(), op_projop()),
            (op_linfunc(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param
