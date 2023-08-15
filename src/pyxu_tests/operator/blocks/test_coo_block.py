# How coo_block() tests work:
#
# * COOBlockMixin auto-defines all (input,output) pairs.
#   [Caveat: we assume the base operators (op_blocks) are correctly implemented.]
#   (True if choosing test operators from examples/.)
#   [Caveat: we assume the punch-out indices (op_hole) are not out-of-bound.]
#
# * To test a coo_block-ed operator, inherit from COOBlockMixin and the suitable conftest.MapT
#   subclass which the compound operator should abide by.

import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc.operator as pxo
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.blocks as pxb
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


# test operators --------------------------------------------------------------
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


# Data Mixin ------------------------------------------------------------------
class COOBlockMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_blocks(self) -> list[list[pxt.OpT]]:
        # Override in inherited class with a 2D nested sequence of operators to be fed to block(, order=1).
        raise NotImplementedError

    @pytest.fixture
    def op_hole(self) -> list[tuple[int, int]]:
        # Override in inherited class with grid indices to punch out of `op_blocks()`.
        raise NotImplementedError

    @pytest.fixture
    def op_gt(self, op_blocks, op_hole) -> pxt.OpT:
        # Replace holes in op_block with 0s -> will serve as the ground-truth.
        op_GT = []
        for _i, row in enumerate(op_blocks):
            op_GT.append([])
            for _j, op in enumerate(row):
                if (_i, _j) in op_hole:
                    _op = pxl.NullOp(shape=op.shape)
                else:
                    _op = op
                op_GT[_i].append(_op)

        op = pxb.block(op_GT, order=1)
        return op

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
            [True, False],  # parallel
        )
    )
    def spec(self, op_blocks, op_hole, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width, parallel = request.param

        data, i, j = [], [], []
        for _i, row in enumerate(op_blocks):
            for _j, op in enumerate(row):
                if (_i, _j) not in op_hole:
                    data.append(op)
                    i.append(_i)
                    j.append(_j)
        N_row, N_col = max(i) + 1, max(j) + 1
        op = pxb.coo_block(
            (data, (i, j)),
            grid_shape=(N_row, N_col),
            parallel=parallel,
        )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, op_gt) -> pxt.OpShape:
        return op_gt.shape

    @pytest.fixture
    def data_apply(self, op_gt) -> conftest.DataLike:
        arr = self._random_array((op_gt.dim,), seed=17)  # random seed for reproducibility
        out = op_gt.apply(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, op_gt) -> conftest.DataLike:
        arr = self._random_array((op_gt.codim,), seed=17)  # random seed for reproducibility
        out = op_gt.adjoint(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, op_gt) -> conftest.DataLike:
        arr = self._random_array((op_gt.dim,), seed=17)  # random seed for reproducibility
        out = op_gt.grad(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, op_gt) -> conftest.DataLike:
        arr = self._random_array((op_gt.dim,), seed=17)  # random seed for reproducibility
        tau = np.abs(self._random_array((1,), seed=31))[0]  # random seed for reproducibility
        out = op_gt.prox(arr=arr, tau=tau)
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


class COOBlockFuncMixin(COOBlockMixin):
    @pytest.fixture
    def op_hole(self) -> list[tuple[int, int]]:
        return []  # horizontal coo_block() objects cannot have any holes.

    # Tests -------------------------------------------------------------------
    def test_interface_asloss(self, op, xp, width):
        self._skip_if_disabled()
        with pytest.raises(NotImplementedError):
            # asloss() is ambiguous for block-defined operators -> must fail.
            op.asloss()


# Test classes (Maps) ---------------------------------------------------------
class TestCOOBlockMap(COOBlockMixin, conftest.MapT):
    @pytest.fixture(
        params=[
            [
                [op_map(3), op_diffmap(3)],
                [op_proxdifffunc(3), op_proxfunc(3)],
            ]
        ]
    )
    def op_blocks(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [(0, 1), (1, 0)],
        ]
    )
    def op_hole(self, request):
        return request.param


class TestCOOBlockDiffMap(COOBlockMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            [
                [op_linfunc(3), op_difffunc(3)],
                [op_squareop(3), op_diffmap(3)],
            ]
        ]
    )
    def op_blocks(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [(0, 0), (1, 1)],
        ]
    )
    def op_hole(self, request):
        return request.param


class TestCOOBlockLinOp(COOBlockMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            [
                [op_linop(3, 2), op_normalop(6)],
                [op_linfunc(3), op_linfunc(6)],
            ]
        ]
    )
    def op_blocks(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [(0, 1), (1, 0)],
        ]
    )
    def op_hole(self, request):
        return request.param


class TestCOOBlockSquareOp(COOBlockMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            [
                [op_squareop(7), op_posdefop(7)],
                [op_orthprojop(7), op_unitop(7)],
            ]
        ]
    )
    def op_blocks(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [(0, 1), (1, 0)],
        ]
    )
    def op_hole(self, request):
        return request.param


# Test classes (Funcs) --------------------------------------------------------
class TestCOOBlockFunc(COOBlockFuncMixin, conftest.FuncT):
    @pytest.fixture(params=[])
    def op_blocks(self, request):
        return request.param


class TestCOOBlockProxFunc(COOBlockFuncMixin, conftest.ProxFuncT):
    @pytest.fixture(
        params=[
            [[op_proxfunc(7), op_proxdifffunc(4)]],
        ]
    )
    def op_blocks(self, request):
        return request.param


class TestCOOBlockDiffFunc(COOBlockFuncMixin, conftest.DiffFuncT):
    @pytest.fixture(
        params=[
            [[op_difffunc(5), op_proxdifffunc(7)]],
            [[op_difffunc(7), op_linfunc(8)]],
        ]
    )
    def op_blocks(self, request):
        return request.param


class TestCOOBlockProxDiffFunc(COOBlockFuncMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            [[op_proxdifffunc(3), op_quadraticfunc(5)]],
            [[op_proxdifffunc(3), op_linfunc(5)]],
        ]
    )
    def op_blocks(self, request):
        return request.param


class TestCOOBlockQuadraticFunc(COOBlockFuncMixin, conftest.QuadraticFuncT):
    @pytest.fixture(
        params=[
            [[op_quadraticfunc(5), op_linfunc(3)]],
            [[op_quadraticfunc(5), op_quadraticfunc(7)]],
        ]
    )
    def op_blocks(self, request):
        return request.param


class TestCOOBlockLinFunc(COOBlockFuncMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            [[op_linfunc(3), op_linfunc(1)]],
        ]
    )
    def op_blocks(self, request):
        return request.param
