# How hstack() tests work:
#
# * HStackMixin auto-defines all (input,output) pairs.
#   [Caveat: we assume the base operators (op_all) are correctly implemented.]
#   (True if choosing test operators from examples/.)
#
# * To test a hstack-ed operator, inherit from HStackMixin and the suitable conftest.MapT subclass
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


def spec_op_map(klass: pxt.OpC, N: int = 2) -> list[list[pxt.OpT]]:
    # create all possible (op1, ..., opN) tuples which, when used to create a hstack-ed operator,
    # produce an operator of type `klass`. (Necessarily a non-functional.)
    def op_map(dim: int):
        import pyxu_tests.operator.examples.test_map as tc

        return tc.ReLU(M=dim)

    def op_diffmap(dim: int):
        import pyxu_tests.operator.examples.test_diffmap as tc

        return tc.Sin(M=dim)

    def op_linop(dim: int, codim_scale: int):
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Tile(N=dim, M=codim_scale)

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
        # Return true if hstack(ops) forms a klass object. [Not a sub-type]
        op_sum = ops[0]  # hack to get the output type (modulo square-ness)
        for op in ops[1:]:
            op_sum = op_sum + op
        properties = op_sum.properties()

        codim = ops[0].codim
        dim = sum(op.dim for op in ops)
        if (pxo.Property.LINEAR in properties) and (codim == dim):
            properties = pxo.SquareOp.properties()
        else:
            properties = properties & pxo.LinOp.properties()

        _klass = pxo.Operator._infer_operator_type(properties)
        return _klass == klass

    ops = []
    for _ops in itertools.combinations_with_replacement(
        [
            op_map(7),
            op_diffmap(7),
            op_linop(7, 1),
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


def spec_op_func(klass: pxt.OpC, N: int = 2) -> list[list[pxt.OpT]]:
    # create all possible (op1, ..., opN) tuples which, when used to create a hstack-ed operator,
    # produce an operator of type `klass`. (Necessarily a functional.)
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

    def op_linfunc(dim: int):
        import pyxu_tests.operator.examples.test_linfunc as tc

        op = tc.ScaledSum(N=dim)
        return op

    def condition(ops: list[pxt.OpT], klass: pxt.OpC) -> bool:
        # Return true if hstack(ops) forms a klass object. [Not a sub-type]
        op_sum = ops[0]  # hack to get the output type (modulo square-ness)
        for op in ops[1:]:
            op_sum = op_sum + op
        _klass = pxo.Operator._infer_operator_type(op_sum.properties())
        return _klass == klass

    ops = []
    for _ops in itertools.combinations_with_replacement(
        [
            op_difffunc(9),
            op_proxfunc(9),
            op_proxdifffunc(9),
            op_quadraticfunc(9),
            op_linfunc(9),
        ],
        N,
    ):
        if condition(_ops, klass):
            ops.append(_ops)
    return ops


# Data Mixin ------------------------------------------------------------------
class HStackMixin:
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
        op = pxb.hstack(op_all)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_all) -> pxt.OpShape:
        dim = sum(op.dim for op in op_all)
        codim = op_all[0].codim
        sh = (codim, dim)
        return sh

    @pytest.fixture
    def data_apply(self, op_all) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        for op in op_all:
            p_arr = self._random_array((op.dim,), seed=3)  # random seed for reproducibility
            parts_arr.append(p_arr)
            p_out = op.apply(p_arr)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.sum(parts_out, axis=0)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op_all) -> conftest.DataLike:
        arr = self._random_array((op_all[0].codim,), seed=4)  # random seed for reproducibility
        parts_out = []
        for op in op_all:
            p_out = op.adjoint(arr)
            parts_out.append(p_out)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op_all) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        for op in op_all:
            p_arr = self._random_array((op.dim,), seed=5)  # random seed for reproducibility
            parts_arr.append(p_arr)
            p_out = op.grad(p_arr)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op_all) -> conftest.DataLike:
        parts_arr = []
        parts_out = []
        tau = np.abs(self._random_array((1,), seed=30))[0]  # random seed for reproducibility
        for op in op_all:
            p_arr = self._random_array((op.dim,), seed=6)  # random seed for reproducibility
            parts_arr.append(p_arr)
            p_out = op.prox(arr=p_arr, tau=tau)
            parts_out.append(p_out)
        arr = np.concatenate(parts_arr)
        out = np.concatenate(parts_out)
        return dict(
            in_=dict(
                arr=arr,
                tau=tau,
            ),
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


class HStackFuncMixin(HStackMixin):
    # Tests -------------------------------------------------------------------
    def test_interface_asloss(self, op, xp, width):
        self._skip_if_disabled()
        with pytest.raises(NotImplementedError):
            # asloss() is ambiguous for hstack-ed operators -> must fail.
            op.asloss()


# Test classes (Maps) ---------------------------------------------------------
class TestHStackMap(HStackMixin, conftest.MapT):
    @pytest.fixture(params=spec_op_map(pxo.Map))
    def op_all(self, request):
        return request.param


class TestHStackDiffMap(HStackMixin, conftest.DiffMapT):
    @pytest.fixture(params=spec_op_map(pxo.DiffMap))
    def op_all(self, request):
        return request.param


class TestHStackLinOp(HStackMixin, conftest.LinOpT):
    @pytest.fixture(params=spec_op_map(pxo.LinOp))
    def op_all(self, request):
        return request.param


class TestHStackSquareOp(HStackMixin, conftest.SquareOpT):
    @pytest.fixture(params=spec_op_map(pxo.SquareOp))
    def op_all(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestHStackFunc(HStackFuncMixin, conftest.FuncT):
    @pytest.fixture(params=spec_op_func(pxo.Func))
    def op_all(self, request):
        return request.param


class TestHStackProxFunc(HStackFuncMixin, conftest.ProxFuncT):
    @pytest.fixture(params=spec_op_func(pxo.ProxFunc))
    def op_all(self, request):
        return request.param


class TestHStackDiffFunc(HStackFuncMixin, conftest.DiffFuncT):
    @pytest.fixture(params=spec_op_func(pxo.DiffFunc))
    def op_all(self, request):
        return request.param


class TestHStackProxDiffFunc(HStackFuncMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(params=spec_op_func(pxo.ProxDiffFunc))
    def op_all(self, request):
        return request.param


class TestHStackQuadraticFunc(HStackFuncMixin, conftest.QuadraticFuncT):
    @pytest.fixture(params=spec_op_func(pxo.QuadraticFunc))
    def op_all(self, request):
        return request.param


class TestHStackLinFunc(HStackFuncMixin, conftest.LinFuncT):
    @pytest.fixture(params=spec_op_func(pxo.LinFunc))
    def op_all(self, request):
        return request.param
