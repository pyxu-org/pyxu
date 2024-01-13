# How ChainRule tests work:
#
# * ChainRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume all tested examples are defined on \bR^{M1,...,MD}.] (This is not a problem in practice.)
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.
#            (True if choosing test operators from examples/.)                ]
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
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest

# It is expected for DenseWarning to be raised when creating some operators, or fallback matrix ops.
pytestmark = pytest.mark.filterwarnings("ignore::pyxu.info.warning.DenseWarning")


# LHS/RHS test operators ------------------------------------------------------
def op_bcast(codim_shape: pxt.NDArrayAxis) -> pxt.OpT:
    # f: \bR -> \bR^{N1,...,NK}
    #      x -> (repeat-x)

    @pxrt.enforce_precision(i="arr")
    def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        sh = arr.shape[: -_.dim_rank]

        expand = (np.newaxis,) * (_.codim_rank - 1)
        y = xp.broadcast_to(
            arr[..., *expand],
            (*sh, *_.codim_shape),
        )
        return y

    @pxrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
        axis = tuple(range(-_.codim_rank, 0))
        y = arr.sum(axis=axis)[..., np.newaxis]
        return y

    op = px_src.from_source(
        cls=pxa.LinOp,
        dim_shape=1,
        codim_shape=codim_shape,
        embed=dict(
            _name="BroadcastOp",
        ),
        apply=op_apply,
        adjoint=op_adjoint,
    )
    op.lipschitz = np.sqrt(np.prod(codim_shape))
    return op


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

    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", pxw.DenseWarning)
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


def op_linfunc(dim_shape: pxt.NDArrayShape, positive: bool) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_linfunc as tc

    op = tc.Sum(dim_shape=dim_shape)
    if not positive:
        op = -op
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


class ChainRuleMixin:
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
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_lhs, op_rhs, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = op_lhs * op_rhs
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_rhs) -> pxt.NDArrayShape:
        return op_rhs.dim_shape

    @pytest.fixture
    def codim_shape(self, op_lhs) -> pxt.NDArrayShape:
        return op_lhs.codim_shape

    @pytest.fixture
    def data_apply(self, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(op_rhs.dim_shape)
        y = op_lhs.apply(op_rhs.apply(x))

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(op_lhs.codim_shape)
        y = op_rhs.adjoint(op_lhs.adjoint(x))

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(op_rhs.dim_shape)
        y = op_rhs.apply(x)
        J_rhs = op_rhs.jacobian(x)
        z = np.tensordot(
            op_lhs.grad(y),
            J_rhs.asarray(),
            axes=op_lhs.dim_rank,
        )

        return dict(
            in_=dict(arr=x),
            out=z,
        )

    @pytest.fixture
    def data_prox(self, op, op_lhs, op_rhs) -> conftest.DataLike:
        x = self._random_array(op_rhs.dim_shape)
        tau = abs(self._random_array((1,)).item()) + 1e-2

        if op.has(pxa.Property.PROXIMABLE):
            y = None
            if op_lhs.has(pxa.Property.PROXIMABLE) and op_rhs.has(pxa.Property.LINEAR_UNITARY):
                # prox \comp unitary
                y = op_rhs.adjoint(op_lhs.prox(op_rhs.apply(x), tau))
            elif op.has(pxa.Property.LINEAR):
                # linfunc \comp lin[op|func]
                y = (
                    x
                    - tau
                    * np.tensordot(
                        op_lhs.asarray(),
                        op_rhs.asarray(),
                        axes=op_lhs.dim_rank,
                    )[0]
                )
            elif op_lhs.has(pxa.Property.LINEAR) and op_rhs.has(pxa.Property.PROXIMABLE):
                # linfunc \comp [prox, proxdiff, quadratic]
                cst = op_lhs.asarray().item()
                y = op_rhs.prox(x, cst * tau)
            elif op_lhs.has(pxa.Property.QUADRATIC) and op_rhs.has(pxa.Property.LINEAR):
                # quadratic \comp linop
                A = op_rhs.asarray()
                A = A.reshape(op_rhs.codim_size, op_rhs.dim_size)

                B = op_lhs._quad_spec()[0].asarray()
                B = B.reshape(op_lhs.dim_size, op_lhs.dim_size)

                C = op_lhs._quad_spec()[1].asarray()
                C = C.reshape(1, op_lhs.dim_size)

                Q = tau * (A.T @ B @ A) + np.eye(op_rhs.dim_size)
                b = x.reshape(-1) - tau * (C @ A)[0]

                y, *_ = splinalg.lstsq(Q, b)
                y = y.reshape(op_rhs.dim_shape)

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
class TestChainRuleMap(ChainRuleMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            (op_map((5,)), op_map((5,))),
            (op_map((5, 3, 4)), op_map((5, 3, 4))),
            (op_map((1,)), op_func((5,))),
            (op_map((1,)), op_func((5, 3, 4))),
            (op_map((5,)), op_diffmap((5,))),
            (op_map((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_map((1,)), op_difffunc((5,))),
            (op_map((1,)), op_difffunc((5, 3, 4))),
            (op_map((1,)), op_proxfunc((5,))),
            (op_map((1,)), op_proxfunc((5, 3, 4))),
            (op_map((1,)), op_proxdifffunc((5,))),
            (op_map((1,)), op_proxdifffunc((5, 3, 4))),
            (op_map((1,)), op_quadraticfunc((5,))),
            (op_map((1,)), op_quadraticfunc((5, 3, 4))),
            (op_map((1,)), op_linop((5,))),
            (op_map((5, 3)), op_linop((5, 3, 4))),
            (op_map((1,)), op_linfunc((5,), False)),
            (op_map((1,)), op_linfunc((5, 3, 4), False)),
            (op_map((5,)), op_squareop((5,))),
            (op_map((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_map((5,)), op_normalop((5,))),
            (op_map((5, 3, 4)), op_normalop((5, 3, 4))),
            (op_map((5,)), op_unitop((5,))),
            (op_map((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_map((5,)), op_selfadjointop((5,))),
            (op_map((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_map((5,)), op_posdefop((5,))),
            (op_map((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_map((5,)), op_projop((5,))),
            (op_map((5, 3, 4)), op_projop((5, 3, 4))),
            (op_map((5,)), op_orthprojop((5,))),
            (op_map((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_diffmap((5,)), op_map((5,))),
            (op_diffmap((5, 3, 4)), op_map((5, 3, 4))),
            (op_diffmap((1,)), op_func((5,))),
            (op_diffmap((1,)), op_func((5, 3, 4))),
            (op_diffmap((1,)), op_proxfunc((5,))),
            (op_diffmap((1,)), op_proxfunc((5, 3, 4))),
            (op_linop((5,)), op_map((5,))),
            (op_linop((5, 3, 4)), op_map((5, 3, 4))),
            (op_bcast((3,)), op_proxfunc((5,))),
            (op_bcast((3, 1, 2)), op_proxfunc((5, 3, 4))),
            (op_squareop((5,)), op_map((5,))),
            (op_squareop((5, 3, 4)), op_map((5, 3, 4))),
            (op_normalop((5,)), op_map((5,))),
            (op_normalop((5, 3, 4)), op_map((5, 3, 4))),
            (op_unitop((5,)), op_map((5,))),
            (op_unitop((5, 3, 4)), op_map((5, 3, 4))),
            (op_selfadjointop((5,)), op_map((5,))),
            (op_selfadjointop((5, 3, 5)), op_map((5, 3, 5))),
            (op_posdefop((5,)), op_map((5,))),
            (op_posdefop((5, 3, 5)), op_map((5, 3, 5))),
            (op_projop((5,)), op_map((5,))),
            (op_projop((5, 3, 4)), op_map((5, 3, 4))),
            (op_orthprojop((5,)), op_map((5,))),
            (op_orthprojop((5, 3, 4)), op_map((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleDiffMap(ChainRuleMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (op_diffmap((5,)), op_diffmap((5,))),
            (op_diffmap((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_diffmap((1,)), op_difffunc((5,))),
            (op_diffmap((1,)), op_difffunc((5, 3, 4))),
            (op_diffmap((1,)), op_proxdifffunc((5,))),
            (op_diffmap((1,)), op_proxdifffunc((5, 3, 4))),
            (op_diffmap((1,)), op_quadraticfunc((5,))),
            (op_diffmap((1,)), op_quadraticfunc((5, 3, 4))),
            (op_diffmap((1,)), op_linop((5,))),
            (op_diffmap((5, 3)), op_linop((5, 3, 4))),
            (op_diffmap((1,)), op_linfunc((5,), False)),
            (op_diffmap((1,)), op_linfunc((5, 3, 4), False)),
            (op_diffmap((5,)), op_squareop((5,))),
            (op_diffmap((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_diffmap((5,)), op_normalop((5,))),
            (op_diffmap((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_diffmap((5,)), op_unitop((5,))),
            (op_diffmap((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_diffmap((5,)), op_selfadjointop((5,))),
            (op_diffmap((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_diffmap((5,)), op_posdefop((5,))),
            (op_diffmap((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_diffmap((5,)), op_projop((5,))),
            (op_diffmap((5, 3, 4)), op_projop((5, 3, 4))),
            (op_diffmap((5,)), op_orthprojop((5,))),
            (op_diffmap((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_linop((5,)), op_diffmap((5,))),
            (op_linop((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_bcast(codim_shape=(5,)), op_difffunc((5,))),
            (op_bcast(codim_shape=(5,)), op_difffunc((5, 3, 4))),
            (op_bcast(codim_shape=(5,)), op_proxdifffunc((5,))),
            (op_bcast(codim_shape=(5,)), op_proxdifffunc((5, 3, 4))),
            (op_bcast(codim_shape=(5,)), op_quadraticfunc((5,))),
            (op_bcast(codim_shape=(5,)), op_quadraticfunc((5, 3, 4))),
            (op_squareop((5,)), op_diffmap((5,))),
            (op_squareop((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_normalop((5,)), op_diffmap((5,))),
            (op_normalop((5, 3, 5)), op_diffmap((5, 3, 5))),
            (op_unitop((5,)), op_diffmap((5,))),
            (op_unitop((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_selfadjointop((5,)), op_diffmap((5,))),
            (op_selfadjointop((5, 3, 5)), op_diffmap((5, 3, 5))),
            (op_posdefop((5,)), op_diffmap((5,))),
            (op_posdefop((5, 3, 5)), op_diffmap((5, 3, 5))),
            (op_projop((5,)), op_diffmap((5,))),
            (op_projop((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_orthprojop((5,)), op_diffmap((5,))),
            (op_orthprojop((5, 3, 4)), op_diffmap((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleLinOp(ChainRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (op_linop((5, 3)), op_linop((5, 3, 4))),
            (op_linop((5, 3, 4)), op_bcast((5, 3, 4))),
            (op_bcast((5, 3, 4)), op_linfunc((5, 3), False)),
            (op_linop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_linop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_linop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_linop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_linop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_linop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_linop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_squareop((5, 3)), op_linop((5, 3, 4))),
            (op_normalop((3, 5)), op_linop((3, 5, 4))),
            (op_unitop((5, 3)), op_linop((5, 3, 4))),
            (op_selfadjointop((3, 5)), op_linop((3, 5, 4))),
            (op_posdefop((3, 5)), op_linop((3, 5, 4))),
            (op_projop((5, 3)), op_linop((5, 3, 4))),
            (op_orthprojop((5, 3)), op_linop((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleSquareOp(ChainRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (op_bcast((5, 3, 4)), op_linfunc((5, 3, 4), False)),  # shapes must be chosen to become square.
            (op_squareop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_squareop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_squareop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_squareop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_squareop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_squareop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_squareop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_normalop((5, 3, 5)), op_squareop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_normalop((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_unitop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_unitop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_unitop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_unitop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_unitop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_unitop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_selfadjointop((5, 3, 5)), op_squareop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_selfadjointop((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_squareop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_unitop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_projop((5, 3, 5))),
            (op_posdefop((5, 3, 5)), op_orthprojop((5, 3, 5))),
            (op_projop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_projop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_projop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_projop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_projop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_projop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_projop((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_orthprojop((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_orthprojop((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_orthprojop((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_orthprojop((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_orthprojop((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_orthprojop((5, 3, 4)), op_projop((5, 3, 4))),
            (op_orthprojop((5, 3, 4)), op_orthprojop((5, 3, 4))),
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
            (op_unitop((5, 3, 4)), op_unitop((5, 3, 4))),
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
            (op_func((5, 3, 4)), op_map((5, 3, 4))),
            (op_func((1,)), op_func((1,))),
            (op_func((1,)), op_func((5, 3, 4))),
            (op_func((1,)), op_diffmap((1,))),
            (op_func((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_func((1,)), op_difffunc((1,))),
            (op_func((1,)), op_difffunc((5, 3, 4))),
            (op_func((1,)), op_proxfunc((1,))),
            (op_func((1,)), op_proxfunc((5, 3, 4))),
            (op_func((1,)), op_proxdifffunc((1,))),
            (op_func((1,)), op_proxdifffunc((5, 3, 4))),
            (op_func((1,)), op_quadraticfunc((1,))),
            (op_func((1,)), op_quadraticfunc((5, 3, 4))),
            (op_func((1,)), op_linop((5,))),
            (op_func((5,)), op_bcast((5,))),
            (op_func((5, 3)), op_linop((5, 3, 4))),
            (op_func((1,)), op_linfunc((1,), False)),
            (op_func((1,)), op_linfunc((1,), True)),
            (op_func((1,)), op_linfunc((5, 3, 4), False)),
            (op_func((1,)), op_linfunc((5, 3, 4), True)),
            (op_func((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_func((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_func((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_func((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_func((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_func((5, 3, 4)), op_projop((5, 3, 4))),
            (op_func((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_difffunc((1,)), op_map((1,))),
            (op_difffunc((5, 3, 4)), op_map((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_map((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_proxfunc((5, 3)), op_linop((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_bcast((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_proxfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_proxfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_proxfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_proxfunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_proxfunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_proxdifffunc((5, 3, 4)), op_map((5, 3, 4))),
            (op_quadraticfunc((5, 3, 4)), op_map((5, 3, 4))),
            (op_quadraticfunc((1,)), op_func((5, 3, 4))),
            (op_quadraticfunc((1,)), op_func((1,))),
            (op_quadraticfunc((1,)), op_proxfunc((5, 3, 4))),
            (op_quadraticfunc((1,)), op_proxfunc((1,))),
            (op_linfunc((1,), False), op_map((1,))),
            (op_linfunc((5, 3, 4), False), op_map((5, 3, 4))),
            (op_linfunc((1,), False), op_func((1,))),
            (op_linfunc((1,), False), op_func((5, 3, 4))),
            (op_linfunc((1,), False), op_proxfunc((1,))),
            (op_linfunc((1,), False), op_proxfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleDiffFunc(ChainRuleMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            (op_difffunc((1,)), op_diffmap((1,))),
            (op_difffunc((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_difffunc((5, 3)), op_linop((5, 3, 4))),
            (op_difffunc((5, 3, 4)), op_bcast((5, 3, 4))),
            (op_difffunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_difffunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_difffunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_difffunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_difffunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_difffunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_difffunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_proxdifffunc((1,)), op_diffmap((1,))),
            (op_proxdifffunc((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_proxdifffunc((5, 3)), op_linop((5, 3, 4))),
            (op_proxdifffunc((5, 3, 4)), op_bcast((5, 3, 4))),
            (op_proxdifffunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_proxdifffunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_proxdifffunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_proxdifffunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_proxdifffunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_proxdifffunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_quadraticfunc((1,)), op_diffmap((1,))),
            (op_quadraticfunc((5, 3, 4)), op_diffmap((5, 3, 4))),
            (op_quadraticfunc((1,)), op_difffunc((5,))),
            (op_quadraticfunc((1,)), op_difffunc((5, 3, 4))),
            (op_quadraticfunc((1,)), op_proxdifffunc((5,))),
            (op_quadraticfunc((1,)), op_proxdifffunc((5, 3, 4))),
            (op_quadraticfunc((1,)), op_quadraticfunc((5,))),
            (op_quadraticfunc((1,)), op_quadraticfunc((5, 3, 5))),
            (op_linfunc((1,), True), op_diffmap((1,))),
            (op_linfunc((1,), False), op_diffmap((1,))),
            (op_linfunc((5, 3, 4), True), op_diffmap((5, 3, 4))),
            (op_linfunc((5, 3, 4), False), op_diffmap((5, 3, 4))),
            (op_linfunc((1,), True), op_difffunc((5,))),
            (op_linfunc((1,), False), op_difffunc((5,))),
            (op_linfunc((1,), True), op_difffunc((5, 3, 4))),
            (op_linfunc((1,), False), op_difffunc((5, 3, 4))),
            (op_linfunc((1,), True), op_proxdifffunc((5,))),
            (op_linfunc((1,), False), op_proxdifffunc((5,))),
            (op_linfunc((1,), True), op_proxdifffunc((5, 3, 4))),
            (op_linfunc((1,), False), op_proxdifffunc((5, 3, 4))),
            (op_linfunc((1,), True), op_quadraticfunc((5,))),
            (op_linfunc((1,), False), op_quadraticfunc((5,))),
            (op_linfunc((1,), True), op_quadraticfunc((5, 3, 5))),
            (op_linfunc((1,), False), op_quadraticfunc((5, 3, 5))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleProxFunc(ChainRuleMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            (op_proxfunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_linfunc((1,), True), op_proxfunc((1,))),
            (op_linfunc((1,), True), op_proxfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleProxDiffFunc(ChainRuleMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (op_proxdifffunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_linfunc((1,), True), op_proxdifffunc((1,))),
            (op_linfunc((1,), True), op_proxdifffunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleQuadraticFunc(ChainRuleMixin, conftest.QuadraticFuncT):
    @pytest.fixture(
        params=[
            (op_quadraticfunc((5, 3)), op_linop((5, 3, 4))),
            (op_quadraticfunc((1,)), op_linfunc((5, 3, 4), False)),
            (op_quadraticfunc((5, 3, 4)), op_squareop((5, 3, 4))),
            (op_quadraticfunc((5, 3, 5)), op_normalop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 4)), op_unitop((5, 3, 4))),
            (op_quadraticfunc((5, 3, 5)), op_selfadjointop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 5)), op_posdefop((5, 3, 5))),
            (op_quadraticfunc((5, 3, 4)), op_projop((5, 3, 4))),
            (op_quadraticfunc((5, 3, 4)), op_orthprojop((5, 3, 4))),
            (op_linfunc((1,), True), op_quadraticfunc((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param


class TestChainRuleLinFunc(ChainRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (op_linfunc((1,), False), op_linop((5,))),
            (op_linfunc((5, 3), False), op_linop((5, 3, 4))),
            (op_linfunc((1,), False), op_linfunc((1,), False)),
            (op_linfunc((1,), False), op_linfunc((5, 3, 4), False)),
            (op_linfunc((5, 3, 4), False), op_squareop((5, 3, 4))),
            (op_linfunc((5, 3, 5), False), op_normalop((5, 3, 5))),
            (op_linfunc((5, 3, 4), False), op_unitop((5, 3, 4))),
            (op_linfunc((5, 3, 5), False), op_selfadjointop((5, 3, 5))),
            (op_linfunc((5, 3, 5), False), op_posdefop((5, 3, 5))),
            (op_linfunc((5, 3, 4), False), op_projop((5, 3, 4))),
            (op_linfunc((5, 3, 4), False), op_orthprojop((5, 3, 4))),
        ]
    )
    def op_lrhs(self, request):
        return request.param
