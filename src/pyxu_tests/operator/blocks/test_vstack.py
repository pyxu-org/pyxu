# How vstack() tests work:
#
# * VStackMixin auto-defines all (input,output) pairs.
#   [Caveat: we assume the base operators (op_all) are correctly implemented.]
#   (True if choosing test operators from examples/.)
#
# * To test a vstack-ed operator, inherit from VStackMixin and the suitable conftest.MapT subclass
#   which the compound operator should abide by.

import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc.operator as pxo
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.blocks as pxb
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


def spec_op(klass: pxt.OpC, N: int = 2) -> list[list[pxt.OpT]]:
    # create all possible (op1, ..., opN) tuples which, when used to create a vstack-ed operator,
    # produce an operator of type `klass`.
    def op_map(dim: int):
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(M=dim)

    def op_diffmap(dim: int):
        import pyxu_tests.operator.examples.test_diffmap as tc

        return tc.Sin(M=dim)

    def op_difffunc(dim: int):
        import pyxu_tests.operator.examples.test_difffunc as tc

        return tc.SquaredL2Norm(M=dim)

    def op_proxfunc(dim: int):
        import pyxu_tests.operator.examples.test_proxfunc as tc

        return tc.L1Norm(M=dim)

    def op_proxdifffunc(dim: int):
        import pyxu_tests.operator.examples.test_proxdifffunc as tc

        return tc.SquaredL2Norm(M=dim)

    def op_quadraticfunc(dim: int):
        from pyxu_tests.operator.examples.test_linfunc import ScaledSum
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        return pxo.QuadraticFunc(
            shape=(1, dim),
            Q=PSDConvolution(N=dim),
            c=ScaledSum(N=dim),
            t=1,
        )

    def op_linop(dim: int, codim_scale: int):
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Tile(N=dim, M=codim_scale)

    def op_linfunc(dim: int):
        import pyxu_tests.operator.examples.test_linfunc as tc

        op = tc.ScaledSum(N=dim)
        return op

    def op_squareop(dim: int):
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(N=dim)

    def op_normalop(dim: int):
        import pyxu_tests.operator.examples.test_normalop as tc

        rng = np.random.default_rng(seed=2)
        h = rng.normal(size=(dim,))
        return tc.CircularConvolution(h=h)

    def op_unitop(dim: int):
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(N=dim)

    def op_selfadjointop(dim: int):
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(N=dim)

    def op_posdefop(dim: int):
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(N=dim)

    def op_projop(dim: int):
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(N=dim, alpha=np.pi / 4)

    def op_orthprojop(dim: int):
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(N=dim)

    def condition(ops: list[pxt.OpT], klass: pxt.OpC) -> bool:
        # Return true if vstack(ops) forms a klass object. [Not a sub-type]
        properties = frozenset.intersection(*[op.properties() for op in ops]) & {
            pxo.Property.CAN_EVAL,
            pxo.Property.DIFFERENTIABLE,
            pxo.Property.LINEAR,
        }
        properties = set(properties)

        dim = ops[0].shape[1]
        codim = sum(op.shape[0] for op in ops)
        if dim == codim:
            properties.add(pxo.Property.LINEAR_SQUARE)

        _klass = pxo.Operator._infer_operator_type(properties)
        return _klass == klass

    ops = []
    for _ops in itertools.combinations(
        [
            op_map(7),
            op_diffmap(7),
            op_difffunc(7),
            op_proxfunc(7),
            op_proxdifffunc(7),
            op_quadraticfunc(7),
            op_linop(7, 2),
            op_linfunc(7),
            op_squareop(7),
            op_normalop(7),
            op_unitop(7),
            op_selfadjointop(7),
            op_posdefop(7),
            op_projop(7),
            op_orthprojop(7),
        ],
        N,
    ):
        if condition(_ops, klass):
            ops.append(_ops)
    return ops


# Data Mixin ------------------------------------------------------------------
class VStackMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_all(self) -> list[pxt.OpT]:
        # Override in inherited class with sub-operators.
        raise NotImplementedError

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, op_all, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxb.vstack(op_all)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_all) -> pxt.OpShape:
        dim = op_all[0].dim
        codim = sum(op.codim for op in op_all)
        sh = (codim, dim)
        return sh

    @pytest.fixture
    def data_apply(self, op_all, data_shape) -> conftest.DataLike:
        arr = self._random_array((data_shape[1],), seed=3)  # random seed for reproducibility
        parts_out = []
        for op in op_all:
            p_out = op.apply(arr)
            parts_out.append(p_out)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op_all, data_shape) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        for op in op_all:
            p_arr = self._random_array((op.codim,), seed=3)  # random seed for reproducibility
            parts_arr.append(p_arr)

            p_out = op.adjoint(p_arr)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.sum(parts_out, axis=0)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, data_shape[1]
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape) -> cabc.Collection[np.ndarray]:
        N_test, dim = 5, data_shape[1]
        return self._random_array((N_test, dim))

    # Tests -------------------------------------------------------------------


# Test classes (Maps) ---------------------------------------------------------
class TestVStackMap(VStackMixin, conftest.MapT):
    @pytest.fixture(params=spec_op(pxo.Map))
    def op_all(self, request):
        return request.param


class TestVStackDiffMap(VStackMixin, conftest.DiffMapT):
    @pytest.fixture(params=spec_op(pxo.DiffMap))
    def op_all(self, request):
        return request.param


class TestVStackLinOp(VStackMixin, conftest.LinOpT):
    @pytest.fixture(params=spec_op(pxo.LinOp))
    def op_all(self, request):
        return request.param


class TestVStackSquareOp(VStackMixin, conftest.SquareOpT):
    @pytest.fixture(params=spec_op(pxo.SquareOp))
    def op_all(self, request):
        return request.param
