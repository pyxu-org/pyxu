# TODO


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

    return tc.Rotation(ax=0, ay=0, az=np.pi / 3)


def op_selfadjointop():
    import pycsou_tests.operator.examples.test_selfadjointop as tc

    return tc.CDO2(N=7)


def op_posdefop():
    import pycsou_tests.operator.examples.test_posdefop as tc

    return tc.CDO4(N=7)


def op_projop():
    import pycsou_tests.operator.examples.test_projop as tc

    return tc.CabinetProjection(angle=np.pi / 4)


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

    @pytest.fixture(params=[False, True])
    def swap_lhs_rhs(self, request) -> bool:
        return request.param

    @pytest.fixture
    def op(self, op_lhs, op_rhs, swap_lhs_rhs) -> pyct.OpT:
        if swap_lhs_rhs:
            return op_rhs + op_lhs
        else:
            return op_lhs + op_rhs

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
        out = op_lhs.adjoint(arr) + op_rhs.adjoint(arr)
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
            # (op_map(), op_unitop()),
            (op_map(), op_selfadjointop()),
            (op_map(), op_posdefop()),
            # (op_map(), op_projop()),
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
            # (op_proxfunc(), op_unitop()),
            (op_proxfunc(), op_selfadjointop()),
            (op_proxfunc(), op_posdefop()),
            # (op_proxfunc(), op_projop()),
            (op_proxfunc(), op_orthprojop()),
        ]
    )
    def op_lrhs(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
