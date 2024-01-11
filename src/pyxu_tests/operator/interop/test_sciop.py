import itertools

import pytest
import scipy.sparse.linalg as spsl

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop as pxio
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class SciOpMixin:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        # Override in inherited class with the operator used to create the sci_op.
        # Must be (1D -> 1D) op due to LinearOperator limitations.
        raise NotImplementedError

    @pytest.fixture
    def dim_shape(self, op_orig) -> pxt.NDArrayShape:
        return (op_orig.dim_size,)

    @pytest.fixture
    def codim_shape(self, op_orig) -> pxt.NDArrayShape:
        return (op_orig.codim_size,)

    @pytest.fixture
    def data_apply(self, dim_shape, op_orig) -> conftest.DataLike:
        arr = self._random_array(dim_shape)
        out = op_orig.apply(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_adjoint(self, codim_shape, op_orig) -> conftest.DataLike:
        arr = self._random_array(codim_shape)
        out = op_orig.adjoint(arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


# from_sciop() ================================================================
class FromSciOpMixin(SciOpMixin):
    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # DASK-based sci_ops unsupported
            ],
            pxrt.Width,
        )
    )
    def spec(self, op_orig, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        if ndi == pxd.NDArrayInfo.CUPY:
            xpl = pxu.import_module("cupyx.scipy.sparse.linalg")
        else:  # NUMPY
            xpl = spsl

        A = op_orig.asarray(xp=ndi.module(), dtype=width.value)
        B = xpl.aslinearoperator(A)
        op = pxio.from_sciop(cls=self.base, sp_op=B)

        return op, ndi, width


class TestFromSciOpLinFunc(FromSciOpMixin, conftest.LinFuncT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linfunc as tc

        return tc.Sum(dim_shape=(7,))


class TestFromSciOpLinOp(FromSciOpMixin, conftest.LinOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Sum(dim_shape=(15,))


class TestFromSciOpSquareOp(FromSciOpMixin, conftest.SquareOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(dim_shape=(7,))


class TestFromSciOpNormalOp(FromSciOpMixin, conftest.NormalOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_normalop as tc

        conv_filter = self._random_array((5,), seed=2)
        return tc.CircularConvolution(dim_shape=(5,), h=conv_filter)


class TestFromSciOpUnitOp(FromSciOpMixin, conftest.UnitOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(dim_shape=(7,))


class TestFromSciOpSelfAdjointOp(FromSciOpMixin, conftest.SelfAdjointOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(dim_shape=(7,))


class TestFromSciOpPosDefOp(FromSciOpMixin, conftest.PosDefOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(dim_shape=(7,))


class TestFromSciOpProjOp(FromSciOpMixin, conftest.ProjOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(dim_shape=(7,), alpha=0.3)


class TestFromSciOpOrthProjOp(FromSciOpMixin, conftest.OrthProjOpT):
    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(dim_shape=(7,))


# to_sciop() ==================================================================
class ToSciOpMixin(SciOpMixin):
    # We piggy back on LinOpT sub-classes to re-use most of their fixture logic.
    # All non-sciop-related tests are disabled however since op_orig()s are assumed correct.
    # The goal is rather to test the new tests defined below.
    #
    # Each sub-class of this Mixin sets `disable_test` adequately.

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # DASK-based sci_ops unsupported
            ],
            pxrt.Width,
        )
    )
    def spec(self, op_orig, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        self._skip_unless_2D(op_orig)
        return op_orig, ndi, width  # trivially forward op_orig() here

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op_sciop(self, op, _gpu, width) -> spsl.LinearOperator:
        A = pxio.to_sciop(op, dtype=width.value, gpu=_gpu)
        return A

    @pytest.fixture(
        params=[
            "matvec",
            "matmat",
            "rmatvec",
            "rmatmat",
        ]
    )
    def _data_to_sciop(self, op, xp, width, request) -> conftest.DataLike:
        N_test = 10
        f = lambda _: self._random_array(_, xp=xp, width=width)
        op_array = op.asarray(xp=xp, dtype=width.value)
        mode = request.param
        if mode == "matvec":
            arr = f((op.dim_size,))
            out_gt = op_array @ arr
            var = "x"
        elif mode == "matmat":
            arr = f((op.dim_size, N_test))
            out_gt = op_array @ arr
            var = "X"
        elif mode == "rmatvec":
            arr = f((op.codim_size,))
            out_gt = op_array.T @ arr
            var = "x"
        elif mode == "rmatmat":
            arr = f((op.codim_size, N_test))
            out_gt = op_array.T @ arr
            var = "X"
        return dict(
            in_={var: arr},
            out=out_gt,
            mode=mode,  # for test_xxx_sciop()
        )

    # Tests -------------------------------------------------------------------
    def test_value_to_sciop(self, op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert out.shape == out_gt.shape
        assert self._metric(out, out_gt, as_dtype=out_gt.dtype)

    def test_backend_to_sciop(self, op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert pxu.get_array_module(out) == pxu.get_array_module(out_gt)

    def test_prec_to_sciop(self, op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert out.dtype == out_gt.dtype


class TestToSciOpLinFunc(ToSciOpMixin, conftest.LinFuncT):
    disable_test = frozenset({name for name in dir(conftest.LinFuncT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linfunc as tc

        return tc.Sum(dim_shape=(7,))


class TestToSciOpLinOp(ToSciOpMixin, conftest.LinOpT):
    disable_test = frozenset({name for name in dir(conftest.LinOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_linop as tc

        return tc.Sum(dim_shape=(15,))


class TestToSciOpSquareOp(ToSciOpMixin, conftest.SquareOpT):
    disable_test = frozenset({name for name in dir(conftest.SquareOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_squareop as tc

        return tc.CumSum(dim_shape=(7,))


class TestToSciOpNormalOp(ToSciOpMixin, conftest.NormalOpT):
    disable_test = frozenset({name for name in dir(conftest.NormalOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_normalop as tc

        conv_filter = self._random_array((5,), seed=2)
        return tc.CircularConvolution(dim_shape=(5,), h=conv_filter)


class TestToSciOpUnitOp(ToSciOpMixin, conftest.UnitOpT):
    disable_test = frozenset({name for name in dir(conftest.UnitOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_unitop as tc

        return tc.Permutation(dim_shape=(7,))


class TestToSciOpSelfAdjointOp(ToSciOpMixin, conftest.SelfAdjointOpT):
    disable_test = frozenset({name for name in dir(conftest.SelfAdjointOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_selfadjointop as tc

        return tc.SelfAdjointConvolution(dim_shape=(7,))


class TestToSciOpPosDefOp(ToSciOpMixin, conftest.PosDefOpT):
    disable_test = frozenset({name for name in dir(conftest.PosDefOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_posdefop as tc

        return tc.PSDConvolution(dim_shape=(7,))


class TestToSciOpProjOp(ToSciOpMixin, conftest.ProjOpT):
    disable_test = frozenset({name for name in dir(conftest.ProjOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_projop as tc

        return tc.Oblique(dim_shape=(7,), alpha=0.3)


class TestToSciOpOrthProjOp(ToSciOpMixin, conftest.OrthProjOpT):
    disable_test = frozenset({name for name in dir(conftest.OrthProjOpT) if name.startswith("test")})

    @pytest.fixture
    def op_orig(self) -> pxt.OpT:
        import pyxu_tests.operator.examples.test_orthprojop as tc

        return tc.ScaleDown(dim_shape=(7,))
