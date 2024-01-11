import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ExplicitLinOpMixin:
    # Helpers -----------------------------------------------------------------
    @staticmethod
    def configs() -> list[tuple[pxd.NDArrayInfo, bool, cabc.Callable]]:
        N = pxd.NDArrayInfo
        S = pxd.SparseArrayInfo
        cfg = []  # (backend, only_2D, array_initializer)

        cfg += [  # NUMPY inputs ------------------------
            (N.NUMPY, False, N.NUMPY.module().array),
            (N.NUMPY, True, S.SCIPY_SPARSE.module().bsr_matrix),
            (N.NUMPY, True, S.SCIPY_SPARSE.module().coo_matrix),
            (N.NUMPY, True, S.SCIPY_SPARSE.module().csc_matrix),
            (N.NUMPY, True, S.SCIPY_SPARSE.module().csr_matrix),
        ]
        cfg += [  # DASK inputs -------------------------
            (N.DASK, False, N.DASK.module().array),
        ]
        if pxd.CUPY_ENABLED:  # CUPY inputs -------------
            cfg += [
                (N.CUPY, False, N.CUPY.module().array),
                (N.CUPY, True, S.CUPY_SPARSE.module().coo_matrix),
                (N.CUPY, True, S.CUPY_SPARSE.module().csc_matrix),
                (N.CUPY, True, S.CUPY_SPARSE.module().csr_matrix),
            ]
        return cfg

    # Fixtures (Public-Facing; override in sub-classes) -----------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Ground-truth matrix-free operator. (Any dim/codim-shape allowed.)
        raise NotImplementedError

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    @pytest.fixture
    def dim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.dim_shape

    @pytest.fixture
    def codim_shape(self, op_orig) -> pxt.NDArrayShape:
        return op_orig.codim_shape

    @pytest.fixture
    def data_apply(self, op_orig) -> conftest.DataLike:
        x = self._random_array(op_orig.dim_shape)
        y = op_orig.apply(x)
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(
        params=itertools.product(
            pxrt.Width,
            configs(),
        )
    )
    def spec(self, op_orig, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        width, (ndi, only_2d, initialize) = request.param
        self._skip_if_unsupported(ndi)
        if only_2d:
            self._skip_unless_2D(op_orig)

        mat = initialize(
            op_orig.asarray(
                xp=ndi.module(),
                dtype=width.value,
            )
        )
        assert mat.dtype == width.value

        op = self.base.from_array(
            A=mat,
            dim_rank=op_orig.dim_rank,
            enable_warnings=False,
        )
        return op, ndi, width


class TestExplicitLinOp(ExplicitLinOpMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            ((1,), (5,)),
            ((5,), (3, 5)),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu.operator import BroadcastAxes

        dim_shape, codim_shape = request.param
        op = BroadcastAxes(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        return op


class TestExplicitSquareOp(ExplicitLinOpMixin, conftest.SquareOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_squareop import CumSum

        dim_shape = request.param
        op = CumSum(dim_shape=dim_shape)
        return op


class TestExplicitNormalOp(ExplicitLinOpMixin, conftest.NormalOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_normalop as tc

        dim_shape = request.param
        conv_filter = self._random_array(dim_shape[-1])
        op = tc.CircularConvolution(
            dim_shape=dim_shape,
            h=conv_filter,
        )
        return op


class TestExplicitUnitOp(ExplicitLinOpMixin, conftest.UnitOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_unitop import Permutation

        dim_shape = request.param
        op = Permutation(dim_shape=dim_shape)
        return op


class TestExplicitSelfAdjointOp(ExplicitLinOpMixin, conftest.SelfAdjointOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        dim_shape = request.param
        op = tc.SelfAdjointConvolution(dim_shape=dim_shape)
        return op


class TestExplicitPosDefOp(ExplicitLinOpMixin, conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        dim_shape = request.param
        op = PSDConvolution(dim_shape=dim_shape)
        return op


class TestExplicitProjOp(ExplicitLinOpMixin, conftest.ProjOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_projop import Oblique

        dim_shape = request.param
        op = Oblique(
            dim_shape=dim_shape,
            alpha=np.pi / 4,
        )
        return op


class TestExplicitOrthProjOp(ExplicitLinOpMixin, conftest.OrthProjOpT):
    @pytest.fixture(
        params=[
            (5,),
            (3, 4, 5),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_orthprojop import ScaleDown

        dim_shape = request.param
        op = ScaleDown(dim_shape=dim_shape)
        return op


class TestExplicitLinFunc(ExplicitLinOpMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def op_orig(self, request) -> pxt.OpT:
        from pyxu_tests.operator.examples.test_linfunc import Sum

        dim_shape = request.param
        op = Sum(dim_shape=dim_shape)
        return op
