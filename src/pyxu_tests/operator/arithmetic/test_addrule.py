# How AddRule tests work:
#
# * AddRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR^{M1,...,MD}.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.
#            (True if choosing test operators from examples/.)                ]
#
# * To test a compound operator (via +), inherit from AddRuleMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.


import collections.abc as cabc
import itertools

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


# Helper methods --------------------------------------------------------------
def reshape(
    op: pxt.OpT,
    dim: pxt.NDArrayShape = None,
    codim: pxt.NDArrayShape = None,
) -> pxt.OpT:
    # Reshape an operator to new dim/codim_shape.
    # (Useful for testing advanced broadcasting rules.)
    opR = 1
    if dim is not None:
        opR = pxo.ReshapeAxes(
            dim_shape=dim,
            codim_shape=op.dim_shape,
        )

    opL = 1
    if codim is not None:
        opL = pxo.ReshapeAxes(
            dim_shape=op.codim_shape,
            codim_shape=codim,
        )

    op_reshape = opL * op * opR
    return op_reshape


# LHS/RHS test operators ------------------------------------------------------
def op_map(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_map as tc

    return tc.ReLU(dim_shape=dim_shape)


def op_func(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_func as tc

    return tc.Median(dim_shape=dim_shape)


def op_diffmap(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_diffmap as tc

    return tc.Sin(dim_shape=dim_shape)


def op_difffunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_difffunc as tc

    return tc.SquaredL2Norm(dim_shape=dim_shape)


def op_proxfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_proxfunc as tc

    return tc.L1Norm(dim_shape=dim_shape)


def op_proxdifffunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_proxdifffunc as tc

    return tc.SquaredL2Norm(dim_shape=dim_shape)


def op_quadraticfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    # QuadraticFunc may be defined for dim=1.
    # In this case we cannot use PSDConvolution (examples/test_posdefop.py) due to minimal domain-size restrictions.
    # We therefore use HomothetyOp without loss of generality.

    from pyxu.operator import HomothetyOp
    from pyxu_tests.operator.examples.test_linfunc import Sum

    return pxa.QuadraticFunc(
        dim_shape=dim_shape,
        codim_shape=1,
        Q=HomothetyOp(dim_shape=dim_shape, cst=3),
        c=Sum(dim_shape=dim_shape),
        t=1,
    )


def op_linop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_linop as tc

    return tc.Sum(dim_shape=dim_shape)


def op_linfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_linfunc as tc

    op = tc.Sum(dim_shape=dim_shape)
    return op


def op_squareop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_squareop as tc

    return tc.CumSum(dim_shape=dim_shape)


def op_normalop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_normalop as tc

    rng = np.random.default_rng(seed=2)
    conv_filter = rng.normal(size=dim_shape[-1])
    return tc.CircularConvolution(
        dim_shape=dim_shape,
        h=conv_filter,
    )


def op_unitop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_unitop as tc

    return tc.Permutation(dim_shape=dim_shape)


def op_selfadjointop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_selfadjointop as tc

    return tc.SelfAdjointConvolution(dim_shape=dim_shape)


def op_posdefop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_posdefop as tc

    return tc.PSDConvolution(dim_shape=dim_shape)


def op_projop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_projop as tc

    return tc.Oblique(
        dim_shape=dim_shape,
        alpha=np.pi / 4,
    )


def op_orthprojop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_orthprojop as tc

    return tc.ScaleDown(dim_shape=dim_shape)


class AddRuleMixin:
    # Fixtures (Public-Facing) ------------------------------------------------
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

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture(
        params=itertools.product(
            [
                "lhs + rhs",
                "rhs + lhs",
            ],
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_lhs, op_rhs, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        mode, ndi, width = request.param
        if mode == "lhs + rhs":
            op = op_lhs + op_rhs
        elif mode == "rhs + lhs":
            op = op_rhs + op_lhs
        else:
            raise NotImplementedError
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_lhs, op_rhs) -> pxt.NDArrayShape:
        sh = op_lhs.dim_shape
        return sh

    @pytest.fixture
    def codim_shape(self, op_lhs, op_rhs) -> pxt.NDArrayShape:
        sh = np.broadcast_shapes(
            op_lhs.codim_shape,
            op_rhs.codim_shape,
        )
        return sh

    @pytest.fixture
    def data_apply(self, dim_shape, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = op_lhs.apply(x) + op_rhs.apply(x)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, codim_shape, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(codim_shape)

        bcast_lhs = pxo.BroadcastAxes(
            dim_shape=op_lhs.codim_shape,
            codim_shape=codim_shape,
        )
        out_lhs = bcast_lhs.adjoint(x)
        out_lhs = op_lhs.adjoint(out_lhs)

        bcast_rhs = pxo.BroadcastAxes(
            dim_shape=op_rhs.codim_shape,
            codim_shape=codim_shape,
        )
        out_rhs = bcast_rhs.adjoint(x)
        out_rhs = op_rhs.adjoint(out_rhs)

        y = out_lhs + out_rhs
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, dim_shape, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = op_lhs.grad(x) + op_rhs.grad(x)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(op.dim_shape)
        tau = abs(self._random_array((1,)).item()) + 1e-2

        if op.has(pxa.Property.PROXIMABLE):
            y = None
            if op_lhs.has(pxa.Property.LINEAR):
                P, G = op_rhs, op_lhs
                y = P.prox(x - tau * G.grad(x), tau)
            elif op_rhs.has(pxa.Property.LINEAR):
                P, G = op_lhs, op_rhs
                y = P.prox(x - tau * G.grad(x), tau)
            elif op_lhs.has(pxa.Property.QUADRATIC) and op_rhs.has(pxa.Property.QUADRATIC):
                A = op_lhs._quad_spec()[0].asarray()
                A = A.reshape(op_lhs.dim_size, op_lhs.dim_size)

                B = op_rhs._quad_spec()[0].asarray()
                B = B.reshape(op_rhs.dim_size, op_rhs.dim_size)

                C = op_lhs._quad_spec()[1].asarray()
                C = C.reshape(1, op_lhs.dim_size)

                D = op_rhs._quad_spec()[1].asarray()
                D = D.reshape(1, op_rhs.dim_size)

                Q = tau * (A + B) + np.eye(op.dim_size)
                b = x.reshape(-1) - tau * (C + D)[0]

                y, *_ = splinalg.lstsq(Q, b)
                y = y.reshape(op.dim_shape)

            if y is not None:
                return dict(
                    in_=dict(
                        arr=x,
                        tau=tau,
                    ),
                    out=y,
                )
        raise NotImplementedError

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
class TestAddRuleMap(AddRuleMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            (op_map((5, 3, 4)), op_map((5, 3, 4))),
            (op_map((5, 3, 4)), op_func((5, 3, 4))),
            (op_map((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_map((5, 3, 4)), op_difffunc((5, 3, 4))),
            (op_map((5, 3, 4)), op_proxfunc((5, 3, 4))),
            (op_map((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_map((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_map((5, 3, 4)), op_linfunc((5, 3, 4))),
            (op_map((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_map((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_map((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_map((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_map((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_map((5, 3, 4)), op_projop((5, 3, 4))),
            (op_map((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_func((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_func((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_func((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_func((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_func((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_func((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_func((5, 3, 4)), op_projop((5, 3, 4))),
            (op_func((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_diffmap((5, 3, 4)), op_proxfunc((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_proxfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_proxfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_proxfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_proxfunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            # Advanced broadcasting (one term expands) ------------------------
            (op_map((5, 5, 5)), op_linop((5, 5, 5))),
            (op_map((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 5, 1))),
            (op_map((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 1, 5))),
            (op_map((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(1, 5, 5))),
            # Advanced broadcasting (two terms expand) ------------------------
            # We use op_linop() since it allows to move the bcast-ed axes where we want.
            (reshape(op_linop((5, 5, 5)), codim=(5, 1, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 5, 1))),
            (reshape(op_linop((5, 5, 5)), codim=(1, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 1, 5))),
            (reshape(op_linop((5, 5, 5)), codim=(5, 5, 1)), reshape(op_linop((5, 5, 5)), codim=(1, 5, 5))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleDiffMap(AddRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (op_diffmap((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_diffmap((5, 3, 4)), op_difffunc((5, 3, 4))),
            (op_diffmap((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_diffmap((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_diffmap((5, 3, 4)), op_linfunc((5, 3, 4))),
            (op_diffmap((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_diffmap((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_diffmap((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_diffmap((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_diffmap((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_diffmap((5, 3, 4)), op_projop((5, 3, 4))),
            (op_diffmap((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_difffunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_difffunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_difffunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_difffunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_difffunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_difffunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_difffunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_quadraticfunc((5, 3, 5)), op_squareop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_projop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_squareop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_projop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_orthprojop((5, 3, 5))),
            # Advanced broadcasting (one term expands) ------------------------
            (op_diffmap((5, 5, 5)), op_linop((5, 5, 5))),
            (op_diffmap((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 5, 1))),
            (op_diffmap((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 1, 5))),
            (op_diffmap((5, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(1, 5, 5))),
            # Advanced broadcasting (two terms expand) ------------------------
            # We use op_linop() since it allows to move the bcast-ed axes where we want.
            (reshape(op_linop((5, 5, 5)), codim=(5, 1, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 5, 1))),
            (reshape(op_linop((5, 5, 5)), codim=(1, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 1, 5))),
            (reshape(op_linop((5, 5, 5)), codim=(5, 5, 1)), reshape(op_linop((5, 5, 5)), codim=(1, 5, 5))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleLinOp(AddRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (op_linop((5, 3, 4)), op_linop((5, 3, 4))),
            (op_linop((5, 3, 4)), op_linfunc((5, 3, 4))),
            # Advanced broadcasting (one term expands) ------------------------
            # Linfuncs which register as LinOps since codim_rank > 1.
            (op_linfunc((5, 5, 5)), reshape(op_linfunc((5, 5, 5)), codim=(1,))),
            (op_linfunc((5, 5, 5)), reshape(op_linfunc((5, 5, 5)), codim=(1, 1))),
            (op_linfunc((5, 5, 5)), reshape(op_linfunc((5, 5, 5)), codim=(1, 1, 1))),
            # Advanced broadcasting (two terms expand) ------------------------
            # We use op_linop() since it allows to move the bcast-ed axes where we want.
            (reshape(op_linop((5, 5, 5)), codim=(5, 1, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 5, 1))),
            (reshape(op_linop((5, 5, 5)), codim=(1, 5, 5)), reshape(op_linop((5, 5, 5)), codim=(5, 1, 5))),
            (reshape(op_linop((5, 5, 5)), codim=(5, 5, 1)), reshape(op_linop((5, 5, 5)), codim=(1, 5, 5))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleSquareOp(AddRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (op_linfunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_linfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_linfunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_linfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_linfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_linfunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_linfunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_squareop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_squareop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_squareop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_squareop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_squareop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_squareop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_squareop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_normalop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_unitop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_unitop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_unitop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_unitop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_unitop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_selfadjointop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_projop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_projop((5, 3, 4)), op_orthprojop((5, 3, 4))),
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
            (op_selfadjointop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_orthprojop((5, 3, 4)), op_orthprojop((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRulePosDefOp(AddRuleMixin, conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (op_posdefop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_orthprojop((5, 3, 5))),
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
            (op_func((5, 3, 4)), op_func((5, 3, 4))),
            (op_func((1,)), op_func((1,))),
            (op_func((5, 3, 4)), op_difffunc((5, 3, 4))),
            (op_func((1,)), op_difffunc((1,))),
            (op_func((5, 3, 4)), op_proxfunc((5, 3, 4))),
            (op_func((1,)), op_proxfunc((1,))),
            (op_func((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_func((1,)), op_proxdifffunc((1,))),
            (op_func((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_func((1,)), op_quadraticfunc((1,))),
            (op_func((5, 3, 4)), op_linfunc((5, 3, 4))),
            (op_func((1,)), op_linfunc((1,))),
            (op_difffunc((5, 3, 4)), op_proxfunc((5, 3, 4))),
            (op_difffunc((1,)), op_proxfunc((1,))),
            (op_proxfunc((5, 3, 4)), op_proxfunc((5, 3, 4))),
            (op_proxfunc((1,)), op_proxfunc((1,))),
            (op_proxfunc((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_proxfunc((1,)), op_proxdifffunc((1,))),
            (op_proxfunc((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_proxfunc((1,)), op_quadraticfunc((1,))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleDiffFunc(AddRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            (op_difffunc((5, 3, 4)), op_difffunc((5, 3, 4))),
            (op_difffunc((1,)), op_difffunc((1,))),
            (op_difffunc((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_difffunc((1,)), op_proxdifffunc((1,))),
            (op_difffunc((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_difffunc((1,)), op_quadraticfunc((1,))),
            (op_difffunc((5, 3, 4)), op_linfunc((5, 3, 4))),
            (op_difffunc((1,)), op_linfunc((1,))),
            (op_proxdifffunc((5, 3, 4)), op_proxdifffunc((5, 3, 4))),
            (op_proxdifffunc((1,)), op_proxdifffunc((1,))),
            (op_proxdifffunc((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_proxdifffunc((1,)), op_quadraticfunc((1,))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleProxFunc(AddRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            (op_proxfunc((1,)), op_linfunc((1,))),
            (op_proxfunc((5, 3, 4)), op_linfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleProxDiffFunc(AddRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (op_proxdifffunc((1,)), op_linfunc((1,))),
            (op_proxdifffunc((5, 3, 4)), op_linfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleQuadraticFunc(AddRuleMixin, conftest.QuadraticFuncT):
    @pytest.fixture(
        params=[
            (op_quadraticfunc((1,)), op_quadraticfunc((1,))),
            (op_quadraticfunc((5, 3, 5)), op_quadraticfunc((5, 3, 5))),
            (op_quadraticfunc((1,)), op_linfunc((1,))),
            (op_quadraticfunc((5, 3, 5)), op_linfunc((5, 3, 5))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestAddRuleLinFunc(AddRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (op_linfunc((1,)), op_linfunc((1,))),
            (op_linfunc((5, 3, 4)), op_linfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param
