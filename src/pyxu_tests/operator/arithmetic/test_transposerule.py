# How TransposeRule tests work:
#
# * TransposeRuleMixin auto-defines all arithmetic method (input,output) pairs.
#   [Caveat: we assume the base operators (op_orig) are correctly implemented.
#            (True if choosing test operators from examples/.)                ]
#
# * To test a transposed-operator, inherit from TransposeRuleMixin and the suitable conftest.LinOpT
#   subclass which the transposed operator should abide by.


import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


# Test operators --------------------------------------------------------------
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


def op_linop(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_linop as tc

    return tc.Sum(dim_shape=dim_shape)


def op_linfunc(dim_shape: pxt.NDArrayShape) -> pxt.OpT:
    import pyxu_tests.operator.examples.test_linfunc as tc

    return tc.Sum(dim_shape=dim_shape)


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


class TransposeRuleMixin:
    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator to be transposed.
        raise NotImplementedError

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_orig, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = op_orig.T
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.codim_shape

    @pytest.fixture
    def codim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.dim_shape

    @pytest.fixture
    def data_apply(self, op_orig) -> conftest.DataLike:
        x = self._random_array(op_orig.codim_shape)
        y = op_orig.adjoint(x)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, op_orig) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.apply(x)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_grad(self, op_orig) -> conftest.DataLike:
        # We know that linfuncs-to-be must have op.grad(x) = op.asarray()
        x = self._random_array(op_orig.codim_shape)
        y = op_orig.asarray()[..., 0]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, op_orig) -> conftest.DataLike:
        # We know that linfuncs-to-be must have op.prox(x, tau) = x - op.grad(x) * tau
        x = self._random_array(op_orig.codim_shape)
        g = op_orig.asarray()[..., 0]
        tau = abs(self._random_array((1,)).item()) + 1e-2
        y = x - tau * g

        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )


class TestTransposeRuleSquareOp(TransposeRuleMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_squareop(dim_shape=dim_shape)


class TestTransposeRuleNormalOp(TransposeRuleMixin, conftest.NormalOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_normalop(dim_shape=dim_shape)


class TestTransposeRuleUnitOp(TransposeRuleMixin, conftest.UnitOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_unitop(dim_shape=dim_shape)


class TestTransposeRuleSelfAdjointOp(TransposeRuleMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_selfadjointop(dim_shape=dim_shape)


class TestTransposeRulePosDefOp(TransposeRuleMixin, conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 5),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_posdefop(dim_shape=dim_shape)


class TestTransposeRuleProjOp(TransposeRuleMixin, conftest.ProjOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_projop(dim_shape=dim_shape)


class TestTransposeRuleOrthProjOp(TransposeRuleMixin, conftest.OrthProjOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request):
        dim_shape = request.param
        return op_orthprojop(dim_shape=dim_shape)


class TestTransposeRuleLinOp(TransposeRuleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            op_linfunc((5,)),
            op_linfunc((5, 3, 4)),
            op_linop((5,)),
            op_linop((5, 3, 4)),
        ]
    )
    def op_orig(self, request):
        return request.param


class TestTransposeRuleLinFunc(TransposeRuleMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            op_linfunc((1,)),
            op_linop((1,)),
            op_squareop((1,)),
            op_normalop((1,)),
            op_unitop((1,)),
            op_selfadjointop((1,)),
            op_posdefop((1,)),
            # op_projop((1,)),  Not a projection op!
            # op_orthprojop((1,)),  Degenerate case; svd tests fail!
            op_bcast((1,)),
            op_bcast((5, 3, 4)),
        ]
    )
    def op_orig(self, request):
        return request.param
