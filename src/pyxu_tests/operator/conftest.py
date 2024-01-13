import collections.abc as cabc
import copy
import itertools
import typing as typ

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
from pyxu.abc.operator import _core_operators


def get_test_class(cls: pxt.OpC) -> "MapT":
    # Find the correct MapT subclass designed to test `cls`.
    is_test = lambda _: all(
        [
            hasattr(_, "base"),
            hasattr(_, "interface"),
        ]
    )
    candidates = {_ for _ in globals().values() if is_test(_)}
    for clsT in candidates:
        if clsT.base == cls:
            return clsT
    else:
        raise ValueError(f"No known test type for {cls}.")


# Naming conventions
# ------------------
#
# * test_<property>(op, ...):
#       Verify property values.
#
# * data_<method>(op, ...)
#       Return mappings of the form dict(in_=dict(), out=Any), where:
#         * in_ are kwargs to `op.<method>()`;
#         * out denotes the output of `op.method(**data[in_])`.
#
# * data_math_<method>()
#       Special test data for mathematical identities.
#
# * test_[value,backend,
#         prec,precCM,
#         transparent,math,
#         interface,
#         chunk,
#        ]_<method>(op, ...)
#       Verify that <method>, returns
#       * value: right output values
#       * backend: right output type
#       * prec: input/output have same precision
#       * precCM: output respects context-manager choice
#       * transparent: referential-transparency, i.e. no side-effects
#       * math: mathematical identities hold
#       * interface: objects have the right interface
#       * chunk: output preserves input chunks (stack-dims and/or core-dims)
#
DataLike = cabc.Mapping[str, typ.Any]


class MapT(ct.DisableTestMixin):
    # Class Properties --------------------------------------------------------
    base = pxa.Map
    interface: cabc.Set[str] = frozenset(
        {
            "dim_shape",
            "dim_size",
            "dim_rank",
            "codim_shape",
            "codim_size",
            "codim_rank",
            "asop",
            "argshift",
            "argscale",
            "apply",
            "__call__",
            "lipschitz",
            "estimate_lipschitz",
            "expr",
            "squeeze",
            "transpose",
            "reshape",
            "broadcast_to",
            "subsample",
            "rechunk",
        }
    )

    # Internal helpers --------------------------------------------------------
    @staticmethod
    def _skip_if_unsupported(ndi: pxd.NDArrayInfo):
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")

    @staticmethod
    def _skip_unless_NUMPY_CUPY(ndi: pxd.NDArrayInfo):
        N = pxd.NDArrayInfo
        if ndi not in {N.NUMPY, N.CUPY}:
            pytest.skip("Only NUMPY/CUPY backends supported.")

    @staticmethod
    def _skip_unless_DASK(ndi: pxd.NDArrayInfo):
        N = pxd.NDArrayInfo
        if ndi is not N.DASK:
            pytest.skip("Only DASK backend supported.")

    @staticmethod
    def _skip_unless_2D(op: pxt.OpT):
        if not ((op.dim_rank == 1) and (op.codim_rank == 1)):
            pytest.skip("Only 2D operators supported.")

    @staticmethod
    def _random_array(
        shape: pxt.NDArrayShape,
        seed: int = None,
        xp: pxt.ArrayModule = pxd.NDArrayInfo.NUMPY.module(),
        width: pxrt.Width = pxrt.Width.DOUBLE,
    ) -> pxt.NDArray:
        rng = xp.random.default_rng(seed)
        x = rng.standard_normal(size=shape, dtype=width.value)
        return x

    @staticmethod
    def _check_has_interface(op: pxt.OpT, klass: "MapT"):
        # Verify `op` has the public interface of `klass`.
        assert klass.interface <= frozenset(dir(op))

    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # Function used to assess if computed values returned by arithmetic methods are correct.
        #
        # Users may override this function to introduce an alternative metric when justified.
        # The default metric is point-wise match.
        #
        # Parameters
        # ----------
        # a, b: pxt.NDArray
        #    (...) arrays.
        # as_dtype: pxt.DType
        #    dtype used to compare the values. (Not always relevant depending on the metric.)
        #
        # Returns
        # -------
        # match: bool
        #    True if all (...) arrays match. (Broadcasting rules apply.)
        return ct.allclose(a, b, as_dtype)

    @classmethod
    def _check_value1D(
        cls,
        func,
        data: DataLike,
        dtype: pxt.DType = None,  # use in_["arr"].dtype
    ):
        in_ = data["in_"]
        with pxrt.EnforcePrecision(False):
            out = func(**in_)
        out_gt = data["out"]

        dtype = ct.sanitize(dtype, default=in_["arr"].dtype)
        assert out.shape == out_gt.shape
        assert cls._metric(out, out_gt, as_dtype=dtype)

    @classmethod
    def _check_valueND(
        cls,
        func,
        data: DataLike,
        dtype: pxt.DType = None,  # use in["arr"].dtype
    ):
        sh_extra = (2, 1, 3)  # prepend input/output shape by this amount.

        in_ = data["in_"].copy()
        arr = in_["arr"]
        xp = pxu.get_array_module(arr)
        arr = xp.broadcast_to(arr, (*sh_extra, *arr.shape))
        in_.update(arr=arr)
        with pxrt.EnforcePrecision(False):
            out = func(**in_)
        out_gt = np.broadcast_to(data["out"], (*sh_extra, *data["out"].shape))

        dtype = ct.sanitize(dtype, default=in_["arr"].dtype)
        assert out.shape == out_gt.shape
        assert cls._metric(out, out_gt, as_dtype=dtype)

    @classmethod
    def _check_chunk(
        cls,
        func,
        data: DataLike,
        preserve_core: bool,
    ):
        in_ = data["in_"].copy()
        arr = in_["arr"]
        xp = pxu.get_array_module(arr)
        arr = xp.broadcast_to(
            arr,
            shape=(5, 1, 3, *arr.shape),
            chunks=(2, 2, 2, *arr.chunks),
        )
        in_.update(arr=arr)
        with pxrt.EnforcePrecision(False):
            out = func(**in_)

        assert out.chunks[:3] == arr.chunks[:3]
        if preserve_core:
            assert out.chunks[3:] == arr.chunks[3:]

    @staticmethod
    def _check_backend(func, data: DataLike):
        in_ = data["in_"]
        with pxrt.EnforcePrecision(False):
            out = func(**in_)

        assert type(out) == type(in_["arr"])  # noqa: E721

    @staticmethod
    def _check_prec(func, data: DataLike):
        in_ = data["in_"]
        with pxrt.EnforcePrecision(False):
            out = func(**in_)
            assert out.dtype == in_["arr"].dtype

    @staticmethod
    def _check_precCM(
        func,
        data: DataLike,
        widths: cabc.Collection[pxrt.Width] = pxrt.Width,
    ):
        stats = dict()
        for w in widths:
            with pxrt.Precision(w):
                out = func(**data["in_"])
            stats[w] = out.dtype == w.value
        assert all(stats.values())

    @staticmethod
    def _check_precCM_func(func, width: pxrt.Width):
        with pxrt.Precision(width):
            out = func()
        assert out.dtype == width.value

    @classmethod
    def _check_no_side_effect(cls, func, data: DataLike):
        # idea:
        # * eval func() once [out_1]
        # * in-place update out_1, ex: scale by constant factor [scale]
        # * re-eval func() [out_2]
        # * assert out_1 == out_2, i.e. input was not modified
        data = copy.deepcopy(data)
        in_, out_gt = data["in_"], data["out"]
        scale = 10
        out_gt *= scale

        with pxrt.EnforcePrecision(False):
            out_1 = func(**in_)
            if pxu.copy_if_unsafe(out_1) is not out_1:
                # out_1 is read_only -> safe
                return
            else:
                # out_1 is writeable -> test correctness
                out_1 *= scale
                out_2 = func(**in_)
                out_2 *= scale

            try:
                # The large scale introduced to assess transparency may give rise to
                # operator-dependant round-off errors. We therefore assess transparency at
                # FP32-precision to avoid false negatives.
                assert cls._metric(out_1, out_gt, as_dtype=pxrt.Width.SINGLE.value)
                assert cls._metric(out_2, out_gt, as_dtype=pxrt.Width.SINGLE.value)
            except AssertionError:
                # Function is non-transparent, but which backend caused it?
                ndi = pxd.NDArrayInfo.from_obj(out_1)
                raise Exception(f"Not transparent to {ndi} inputs.")

    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def spec(self) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        # override in subclass to return:
        # * the operator to test;
        # * the backend of accepted input arrays;
        # * the precision of accepted input arrays.
        #   If of type CWidth, it implies arithmetic methods accept real-valued (..., M1,...,MD,2) arrays, i.e. a
        #   bijective view of complex-valued (..., M1,...,MD) arrays.
        #
        # The triplet (op, backend, precision) must be provided since some operators may not be
        # backend/precision-agnostic.
        raise NotImplementedError

    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        # override in subclass with dimension of op.
        # Don't return `op.dim_shape`: hard-code what you are expecting.
        raise NotImplementedError

    @pytest.fixture
    def codim_shape(self) -> pxt.NDArrayShape:
        # override in subclass with co-dimension of op.
        # Don't return `op.codim_shape`: hard-code what you are expecting.
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.apply(). [1D means no stacking dimensions allowed.]
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    @pytest.fixture
    def data_math_lipschitz(self) -> cabc.Collection[np.ndarray]:
        # override in subclass with at least 2 evaluation points for op.apply().
        # Used to verify if op.apply() satisfies the Lipschitz condition.
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def op(self, spec) -> pxt.OpT:
        return spec[0]

    @pytest.fixture
    def ndi(self, spec) -> pxd.NDArrayInfo:
        ndi_ = spec[1]
        self._skip_if_unsupported(ndi_)
        return ndi_

    @pytest.fixture
    def xp(self, ndi) -> pxt.ArrayModule:
        return ndi.module()

    @pytest.fixture
    def width(self, spec, complex_valued) -> pxrt.Width:
        if complex_valued:
            return spec[2].real
        else:
            return spec[2]

    @pytest.fixture
    def complex_valued(self, spec) -> bool:
        return isinstance(spec[2], pxrt.CWidth)

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _gpu(self, ndi) -> bool:
        # Boolean flag needed by some methods
        return ndi == pxd.NDArrayInfo.CUPY

    @pytest.fixture(params=_core_operators())
    def _klass(self, request) -> pxt.OpC:
        # Returns some operator in the Operator hierarchy.
        # Do not override in subclass: for internal use only to test `op.asop()`.
        return request.param

    @pytest.fixture
    def _data_estimate_lipschitz(self) -> DataLike:
        # Generate Cartesian product of inputs.
        # For internal use only to test `op.test_math_lipschitz()`.
        data = dict(in_=dict())
        return data

    @pytest.fixture
    def _data_apply(self, data_apply, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.apply()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_apply["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_apply["out"],
        )
        return data

    # Tests -------------------------------------------------------------------
    def test_interface(self, op):
        self._skip_if_disabled()
        self._check_has_interface(op, self.__class__)

    def test_dim_shape(self, op, dim_shape):
        self._skip_if_disabled()
        assert op.dim_shape == dim_shape

    def test_dim_size(self, op, dim_shape):
        self._skip_if_disabled()
        assert op.dim_size == np.prod(dim_shape)

    def test_dim_rank(self, op, dim_shape):
        self._skip_if_disabled()
        assert op.dim_rank == len(dim_shape)

    def test_codim_shape(self, op, codim_shape):
        self._skip_if_disabled()
        assert op.codim_shape == codim_shape

    def test_codim_size(self, op, codim_shape):
        self._skip_if_disabled()
        assert op.codim_size == np.prod(codim_shape)

    def test_codim_rank(self, op, codim_shape):
        self._skip_if_disabled()
        assert op.codim_rank == len(codim_shape)

    def test_value1D_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_value1D(op.apply, _data_apply)

    def test_valueND_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_valueND(op.apply, _data_apply)

    def test_backend_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_backend(op.apply, _data_apply)

    def test_chunk_apply(self, op, ndi, _data_apply):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.apply, _data_apply, False)

    def test_prec_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_prec(op.apply, _data_apply)

    def test_precCM_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_precCM(op.apply, _data_apply)

    def test_transparent_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_no_side_effect(op.apply, _data_apply)

    def test_value1D_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_value1D(op.__call__, _data_apply)

    def test_valueND_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_valueND(op.__call__, _data_apply)

    def test_backend_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_backend(op.__call__, _data_apply)

    def test_chunk_call(self, op, ndi, _data_apply):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.__call__, _data_apply, False)

    def test_prec_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_prec(op.__call__, _data_apply)

    def test_precCM_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_precCM(op.__call__, _data_apply)

    def test_transparent_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_no_side_effect(op.__call__, _data_apply)

    def test_interface_lipschitz(self, op):
        # .lipschitz() always returns a float.
        self._skip_if_disabled()
        L = op.lipschitz
        assert isinstance(L, pxt.Real)

    def test_interface2_lipschitz(self, op):
        # .lipschitz() can be manually set.
        # [and negative values are illegal.]
        self._skip_if_disabled()
        L_orig = op.lipschitz

        L_new = abs(self._random_array((1,)).item())
        op.lipschitz = L_new
        assert isinstance(op.lipschitz, pxt.Real)
        with pxrt.EnforcePrecision(False):
            assert np.isclose(op.lipschitz, L_new)

        with pytest.raises(Exception):
            op.lipschitz = -1

        # reset state for next tests
        op.lipschitz = L_orig

    def test_precCM_lipschitz(self, op, width):
        self._skip_if_disabled()
        func = lambda: op.lipschitz
        self._check_precCM_func(func, width)

    # We disable RuntimeWarnings which may arise due to NaNs. (See comment below.)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_math_lipschitz(
        self,
        op,
        xp,
        width,
        data_math_lipschitz,
        _data_estimate_lipschitz,
    ):
        # \norm{f(x) - f(y)}{2} \le L * \norm{x - y}{2}
        # L provided by `estimate_lipschitz()`.
        self._skip_if_disabled()
        with pxrt.EnforcePrecision(False):
            L = op.estimate_lipschitz(**_data_estimate_lipschitz["in_"])

            data = xp.array(data_math_lipschitz, dtype=width.value)
            data = list(itertools.combinations(data, 2))
            x = xp.stack([_[0] for _ in data], axis=0)
            y = xp.stack([_[1] for _ in data], axis=0)

            lhs = xp.sum(
                (op.apply(x) - op.apply(y)) ** 2,
                axis=tuple(range(-op.codim_rank, 0)),
            )
            # .apply() may return INFs, in which case `INF-INF=NaN` may arise above.
            # less_equal() is not able to handle NaNs, so the former are overwritten by a sensible
            # value in this context, i.e. 0.
            lhs[xp.isnan(lhs)] = 0

            rhs = (L**2) * xp.sum(
                (x - y) ** 2,
                axis=tuple(range(-op.dim_rank, 0)),
            )
            success = ct.less_equal(lhs, rhs, as_dtype=width.value)
            assert all(success)

    def test_interface_asop(self, op, _klass):
        # * .asop() is no-op if same klass or parent
        # * .asop() has correct interface otherwise.
        # * Expect an exception if cast is illegal. (shape-issues)
        self._skip_if_disabled()
        P = pxa.Property
        if _klass in op.__class__.__mro__:
            op2 = op.asop(_klass)
            assert op2 is op
        elif (P.FUNCTIONAL in _klass.properties()) and not (op.codim_size == op.codim_rank == 1):
            # Casting to functionals when codim != 1.
            with pytest.raises(Exception):
                op2 = op.asop(_klass)
        elif (P.LINEAR_SQUARE in _klass.properties()) and (op.codim_size != op.dim_size):
            # Casting non-square operators to square linops.
            with pytest.raises(Exception):
                op2 = op.asop(_klass)
        else:
            op2 = op.asop(_klass)
            klassT = get_test_class(_klass)
            self._check_has_interface(op2, klassT)

    def test_value_asop(self, op, _klass):
        # Ensure encapsulated arithmetic fields are forwarded.
        # We only test arithmetic fields known to belong to any Map subclass:
        # * estimate_lipschitz (method)
        # * apply (method)
        # * __call__ (special method)
        self._skip_if_disabled()
        try:
            op2 = op.asop(_klass)
        except Exception:
            # nothing to test since `op.asop(_klass)` is illegal.
            return
        else:
            assert op2.estimate_lipschitz == op.estimate_lipschitz
            assert op2.apply == op.apply
            assert op2.__call__ == op.__call__


class FuncT(MapT):
    # Class Properties --------------------------------------------------------
    base = pxa.Func

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def codim_shape(self) -> pxt.NDArrayShape:
        return (1,)

    # Tests -------------------------------------------------------------------
    def test_codim_shape(self, op):
        self._skip_if_disabled()
        assert op.codim_shape == (1,)

    def test_codim_size(self, op):
        self._skip_if_disabled()
        assert op.codim_size == 1

    def test_codim_rank(self, op):
        self._skip_if_disabled()
        assert op.codim_rank == 1


class DiffMapT(MapT):
    # Class Properties --------------------------------------------------------
    base = pxa.DiffMap
    interface = frozenset(
        MapT.interface
        | {
            "jacobian",
            "diff_lipschitz",
            "estimate_diff_lipschitz",
        }
    )

    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def data_math_diff_lipschitz(self) -> cabc.Collection[np.ndarray]:
        # override in subclass with at least 2 evaluation points for op.apply().
        # Used to verify if op.jacobian() satisfies the diff_Lipschitz condition.
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _data_estimate_diff_lipschitz(self) -> DataLike:
        # Generate Cartesian product of inputs.
        # For internal use only to test `op.test_math_diff_lipschitz()`.
        data = dict(in_=dict())
        return data

    # Tests -------------------------------------------------------------------
    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        self._check_has_interface(J, LinOpT)

    def test_interface_diff_lipschitz(self, op):
        # .diff_lipschitz() always returns a float.
        self._skip_if_disabled()
        dL = op.diff_lipschitz
        assert isinstance(dL, pxt.Real)

    def test_interface2_diff_lipschitz(self, op):
        # .diff_lipschitz() can be manually set.
        # [and negative values are illegal.]
        self._skip_if_disabled()
        dL_orig = op.diff_lipschitz

        dL_new = abs(self._random_array((1,)).item())
        op.diff_lipschitz = dL_new
        assert isinstance(op.diff_lipschitz, pxt.Real)
        with pxrt.EnforcePrecision(False):
            assert np.isclose(op.diff_lipschitz, dL_new)

        with pytest.raises(Exception):
            op.diff_lipschitz = -1

        # reset state for next tests
        op.diff_lipschitz = dL_orig

    def test_precCM_diff_lipschitz(self, op, width):
        self._skip_if_disabled()
        func = lambda: op.diff_lipschitz
        self._check_precCM_func(func, width)

    # We disable RuntimeWarnings which may arise due to NaNs. (See comment below.)
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_math_diff_lipschitz(
        self,
        op,
        xp,
        width,
        data_math_diff_lipschitz,
        _data_estimate_diff_lipschitz,
    ):
        # \norm{J(x) - J(y)}{F} \le dL * \norm{x - y}{2}
        # dL provided by `estimate_diff_lipschitz()`.
        self._skip_if_disabled()
        with pxrt.EnforcePrecision(False):
            dL = op.estimate_diff_lipschitz(**_data_estimate_diff_lipschitz["in_"])

            J = lambda _: op.jacobian(_).asarray(dtype=width.value)

            stats = []  # (x, y, condition success)
            data = xp.array(data_math_diff_lipschitz, dtype=width.value)
            for x, y in itertools.combinations(data, 2):
                lhs = np.sum((J(x) - J(y)) ** 2)
                rhs = (dL**2) * xp.sum((x - y) ** 2)
                if np.isnan(lhs):
                    # J() may return INFs, in which case `INF-INF=NaN` may arise above. less_equal()
                    # is not able to handle NaNs, so the former are overwritten by a sensible value
                    # in this context, i.e. 0.
                    lhs = 0
                success = ct.less_equal(lhs, rhs, as_dtype=width.value)
                stats.append((lhs, rhs, success))

            assert all(_[2] for _ in stats)


class DiffFuncT(FuncT, DiffMapT):
    # Class Properties --------------------------------------------------------
    base = pxa.DiffFunc
    interface = frozenset(FuncT.interface | DiffMapT.interface | {"grad"})
    disable_test = frozenset(FuncT.disable_test | DiffMapT.disable_test)

    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def data_grad(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.grad(). [1D means no stacking dimensions allowed.]
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _data_grad(self, data_grad, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.grad()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_grad["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_grad["out"],
        )
        return data

    # Tests -------------------------------------------------------------------
    def test_value1D_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_value1D(op.grad, _data_grad)

    def test_valueND_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_valueND(op.grad, _data_grad)

    def test_backend_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_backend(op.grad, _data_grad)

    def test_chunk_grad(self, op, ndi, _data_grad):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.grad, _data_grad, True)

    def test_prec_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_prec(op.grad, _data_grad)

    def test_precCM_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_precCM(op.grad, _data_grad)

    def test_transparent_grad(self, op, _data_grad):
        self._skip_if_disabled()
        self._check_no_side_effect(op.grad, _data_grad)

    def test_math1_grad(self, op, width, _data_grad):
        # .jacobian/.grad outputs are consistent.
        self._skip_if_disabled()
        arr = _data_grad["in_"]["arr"]
        with pxrt.EnforcePrecision(False):
            J = op.jacobian(arr).asarray(dtype=width.value)
            g = op.grad(arr)

            assert len(J.shape) == len(g.shape) + 1
            assert J.size == g.size
            assert J.shape[1:] == g.shape
            assert ct.allclose(J, g, as_dtype=arr.dtype)

    def test_math2_grad(self, op, xp, width, _data_estimate_lipschitz):
        # f(x - \frac{1}{L} \grad_{f}(x)) <= f(x)
        # L provided by `estimate_lipschitz()`
        self._skip_if_disabled()
        with pxrt.EnforcePrecision(False):
            L = op.estimate_lipschitz(**_data_estimate_lipschitz["in_"])
            if np.isclose(L, 0):
                return  # trivially true since f(x) = cst
            elif np.isclose(L, np.inf):
                return  # trivially true since f(x) <= f(x)

            N_test = 10
            rhs = self._random_array((N_test, *op.dim_shape), xp=xp, width=width)
            lhs = rhs - op.grad(rhs) / L

            assert np.all(
                ct.less_equal(
                    op.apply(lhs),
                    op.apply(rhs),
                    as_dtype=width.value,
                )
            )


class ProxFuncT(FuncT):
    # Class Properties --------------------------------------------------------
    base = pxa.ProxFunc
    interface = frozenset(
        FuncT.interface
        | {
            "prox",
            "fenchel_prox",
            "moreau_envelope",
        }
    )

    # Fixtures (Public-Facing) ------------------------------------------------
    @pytest.fixture
    def data_prox(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.prox(). [1D means no stacking dimensions allowed.]
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def data_fenchel_prox(self, data_prox) -> DataLike:
        # override in subclass with 1D input/outptus of op.fenchel_prox(). [1D means no stacking dimensions allowed.]
        # Default value: inferred from data_prox().
        p_arr = data_prox["in_"]["arr"]
        p_tau = data_prox["in_"]["tau"]
        p_out = data_prox["out"]
        return dict(
            in_=dict(
                arr=p_arr / p_tau,
                sigma=1 / p_tau,
            ),
            out=(p_arr - p_out) / p_tau,
        )

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _op_m(self, op) -> tuple[float, pxa.DiffFunc]:
        mu = 1.1
        return mu, op.moreau_envelope(mu)

    @pytest.fixture
    def _data_prox(self, data_prox, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.prox()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_prox["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_prox["out"],
        )
        return data

    @pytest.fixture
    def _data_fenchel_prox(self, data_fenchel_prox, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.fenchel_prox()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_fenchel_prox["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_fenchel_prox["out"],
        )
        return data

    # Tests -------------------------------------------------------------------
    def test_value1D_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_value1D(op.prox, _data_prox)

    def test_valueND_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_valueND(op.prox, _data_prox)

    def test_backend_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_backend(op.prox, _data_prox)

    def test_chunk_prox(self, op, ndi, _data_prox):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.prox, _data_prox, True)

    def test_prec_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_prec(op.prox, _data_prox)

    def test_precCM_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_precCM(op.prox, _data_prox)

    def test_transparent_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_no_side_effect(op.prox, _data_prox)

    def test_math_prox(self, op, xp, width, _data_prox):
        # Ensure y = prox_{tau f}(x) minimizes:
        # 2\tau f(z) + \norm{z - x}{2}^{2}, for any z \in \bR^{dim_shape}
        self._skip_if_disabled()
        in_ = _data_prox["in_"]
        y = op.prox(**in_)

        N_test = 10
        x = in_["arr"]
        z = x + self._random_array(
            (N_test, *op.dim_shape),
            xp=xp,
            width=width,
        )

        def g(z):
            a = 2 * in_["tau"] * op.apply(z)
            b = xp.sum(
                (x - z) ** 2,
                axis=tuple(range(-op.dim_rank, 0)),
                keepdims=True,
            )
            return a + b

        assert np.all(ct.less_equal(g(y), g(z), as_dtype=y.dtype))

    def test_value1D_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_value1D(op.fenchel_prox, _data_fenchel_prox)

    def test_valueND_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_valueND(op.fenchel_prox, _data_fenchel_prox)

    def test_backend_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_backend(op.fenchel_prox, _data_fenchel_prox)

    def test_chunk_fenchel_prox(self, op, ndi, _data_fenchel_prox):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.fenchel_prox, _data_fenchel_prox, True)

    def test_prec_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_prec(op.fenchel_prox, _data_fenchel_prox)

    def test_precCM_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_precCM(op.fenchel_prox, _data_fenchel_prox)

    def test_transparent_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_no_side_effect(op.fenchel_prox, _data_fenchel_prox)

    def test_interface_moreau_envelope(self, _op_m):
        self._skip_if_disabled()
        _, op_m = _op_m
        self._check_has_interface(op_m, DiffFuncT)

    def test_math1_moreau_envelope(self, op, _op_m, _data_apply):
        # op_m.apply() lower-bounds op.apply()
        self._skip_if_disabled()
        _, op_m = _op_m
        arr = _data_apply["in_"]["arr"]
        lhs = op_m.apply(arr)
        rhs = op.apply(arr)

        assert ct.less_equal(lhs, rhs, as_dtype=rhs.dtype)

    def test_math2_moreau_envelope(self, op, _op_m, _data_apply):
        # op_m.grad(x) * mu = x - op.prox(x, mu)
        self._skip_if_disabled()
        mu, op_m = _op_m
        arr = _data_apply["in_"]["arr"]
        lhs = op_m.grad(arr) * mu
        rhs = arr - op.prox(arr, mu)

        assert self._metric(lhs, rhs, as_dtype=arr.dtype)


class ProxDiffFuncT(ProxFuncT, DiffFuncT):
    # Class Properties --------------------------------------------------------
    base = pxa.ProxDiffFunc
    interface = frozenset(ProxFuncT.interface | DiffFuncT.interface)


@pytest.mark.filterwarnings("ignore::pyxu.info.warning.DenseWarning")
class QuadraticFuncT(ProxDiffFuncT):
    # Class Properties --------------------------------------------------------
    base = pxa.QuadraticFunc
    interface = ProxDiffFuncT.interface

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array((N_test, *op.dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array((N_test, *op.dim_shape))
        return x

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture(params=["svd", "trace"])  # supported methods
    def _data_estimate_diff_lipschitz(self, xp, width, _gpu, request) -> DataLike:
        data = dict(
            in_=dict(
                # Some combination of these kwargs are used in estimate_diff_lipschitz().
                # We provide them all, knowing that implementations should take what they need only.
                method=request.param,
                gpu=_gpu,
                xp=xp,
                dtype=width.value,
            )
        )
        return data

    # Tests -------------------------------------------------------------------
    # [fenchel_]prox() use CG internally.
    # To avoid CG convergence issues, correctness is assesed at lowest precision only.
    def test_value1D_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_value1D(op.prox, _data_prox, dtype=np.dtype(np.half))

    def test_valueND_prox(self, op, _data_prox):
        self._skip_if_disabled()
        self._check_valueND(op.prox, _data_prox, dtype=np.dtype(np.half))

    def test_value1D_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_value1D(op.fenchel_prox, _data_fenchel_prox, dtype=np.dtype(np.half))

    def test_valueND_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_valueND(op.fenchel_prox, _data_fenchel_prox, dtype=np.dtype(np.half))

    def test_math_quad_spec(self, op, xp, width):
        # f(0) == t
        # \grad_{f}(0) == c
        # J_{f}(e_{i}) - c == Q_{i}
        self._skip_if_disabled()
        Q, c, t = op._quad_spec()
        Q = Q.asarray(xp=xp, dtype=width.value)
        c = c.asarray(xp=xp, dtype=width.value)

        # f(0) == t
        x = xp.zeros(op.dim_shape, dtype=width.value)
        with pxrt.EnforcePrecision(False):
            out = op.apply(x)
        assert self._metric(out, t, as_dtype=width.value)

        # \grad_{f}(0) == c
        with pxrt.EnforcePrecision(False):
            out = op.grad(x)
        assert self._metric(out, c, as_dtype=width.value)

        # J_{f}(e_{i}) - c == Q_{i}
        x = xp.reshape(
            xp.eye(op.dim_size, dtype=width.value),
            (*op.dim_shape, *op.dim_shape),
        )
        with pxrt.EnforcePrecision(False):
            out = op.grad(x) - c
        assert self._metric(out, Q, as_dtype=width.value)

    def test_math2_diff_lipschitz(self, op, _data_estimate_diff_lipschitz, _gpu):
        # op.estimate_diff_lipschitz(svd) <= op.estimate_diff_lipschitz(trace)
        self._skip_if_disabled()
        kwargs = _data_estimate_diff_lipschitz["in_"].copy()
        kwargs.pop("method")  # over-written below ...
        dtype = kwargs["dtype"]

        dL_ub = op.estimate_diff_lipschitz(method="trace", **kwargs)
        dL_opt = ct.flaky(  # might raise exception in GPU-mode
            op.estimate_diff_lipschitz,
            args=dict(method="svd", **kwargs),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

        try:
            assert ct.less_equal(dL_opt, dL_ub, as_dtype=dtype).item()
        except AssertionError:
            if _gpu is True:
                # even if dL_opt computed, not guaranteed to be correct in GPU-mode
                pytest.xfail("svdvals() sparse-evaled via CuPy flaky.")
            else:
                raise

    def test_math3_diff_lipschitz(self, op, _data_estimate_diff_lipschitz, _gpu):
        # op.estimate_diff_lipschitz(svd) computes the optimal diff-Lipschitz constant.
        self._skip_if_disabled()
        kwargs = _data_estimate_diff_lipschitz["in_"].copy()
        kwargs.pop("method")  # over-written below ...
        dtype = kwargs["dtype"]

        dL_svds = ct.flaky(  # might raise an assertion in GPU-mode
            func=op.estimate_diff_lipschitz,
            args=dict(method="svd", **kwargs),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

        Q, _, _ = op._quad_spec()
        QL_svds = ct.flaky(  # might raise an assertion in GPU-mode
            func=Q.estimate_lipschitz,
            args=dict(method="svd", **kwargs),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

        # Comparison is done with user-specified metric since operator may compute `dL_svds` via an
        # approximate `.apply()' method.
        try:
            cast = lambda x: np.array([x], dtype=dtype)
            assert self._metric(cast(dL_svds), cast(QL_svds), as_dtype=dtype)
        except AssertionError:
            if _gpu is True:
                # even if dL_opt computed, not guaranteed to be correct in GPU-mode
                pytest.xfail("svdvals() sparse-evaled via CuPy flaky.")
            else:
                raise


@pytest.mark.filterwarnings("ignore::pyxu.info.warning.DenseWarning")
class LinOpT(DiffMapT):
    # Class Properties --------------------------------------------------------
    base = pxa.LinOp
    interface = frozenset(
        DiffMapT.interface
        | {
            "adjoint",
            "asarray",
            "cogram",
            "dagger",
            "from_array",
            "gram",
            "pinv",
            "svdvals",
            "T",
        }
    )

    # Internal helpers --------------------------------------------------------
    @classmethod
    def _check_value1D_vals(cls, func, kwargs, ground_truth):
        N = pxd.NDArrayInfo
        xp = N.from_flag(kwargs["gpu"]).module()
        if kwargs["gpu"]:
            ground_truth = xp.array(ground_truth)

        k = kwargs["k"]
        out = func(**kwargs)
        idx = xp.argsort(xp.abs(out))
        assert out.size == k  # obtain N_vals asked for
        assert ct.allclose(out[idx], out, out.dtype)  # sorted in ascending magnitude

        # and output is correct (in magnitude)
        idx_gt = xp.argsort(xp.abs(ground_truth))
        out = out[idx][-k:]
        gt = ground_truth[idx_gt][-k:]

        assert cls._metric(xp.abs(out), xp.abs(gt), as_dtype=out.dtype)

    @staticmethod
    def _check_backend_vals(func, _gpu, width):
        N = pxd.NDArrayInfo
        out = func(k=1, gpu=_gpu, dtype=width.value)
        assert N.from_obj(out) == N.from_flag(_gpu)

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def data_adjoint(self, op) -> DataLike:
        # override in subclass with 1D input/outputs of op.adjoint(). [1D means no stacking dimensions allowed.]
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation just tests A(0) = 0. Subsequent math tests ensure .adjoint() output
        # is consistent with .apply().
        return dict(
            in_=dict(arr=np.zeros(op.codim_shape)),
            out=np.zeros(op.dim_shape),
        )

    @pytest.fixture
    def data_pinv(self, op, _damp, data_apply) -> DataLike:
        # override in subclass with 1D input/outputs of op.pinv(). [1D means no stacking dimensions allowed.]
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation: auto-computes pinv() at output points specified to test op.apply().

        # Idea: use scipy.linalg.lstsq() to compute pseudo-inverse.
        # Caveat: lstsq() understands 2D operators only, so we must 2D-ize the array representation.

        # Safe implementation of --------------------------
        #   A = op.gram().asarray(xp=np, dtype=pxrt.Width.DOUBLE.value)
        B = np.reshape(
            op.asarray(
                xp=np,
                dtype=pxrt.Width.DOUBLE.value,
            ),
            (op.codim_size, op.dim_size),
        )
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim_size):
            A[i, i] += _damp

        arr = data_apply["out"].reshape(op.codim_size)
        out, *_ = splinalg.lstsq(A, B.T @ arr)
        data = dict(
            in_=dict(
                arr=arr.reshape(op.codim_shape),
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out.reshape(op.dim_shape),
        )
        return data

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array((N_test, *op.dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array((N_test, *op.dim_shape))
        return x

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _data_adjoint(self, data_adjoint, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.adjoint()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_adjoint["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_adjoint["out"],
        )
        return data

    @pytest.fixture
    def _data_estimate_lipschitz(self, xp, width, _gpu) -> DataLike:
        data = dict(
            in_=dict(
                # method=...  # defined in tests as needed
                gpu=_gpu,
                xp=xp,
                dtype=width.value,
            )
        )
        return data

    @pytest.fixture
    def _op_svd(self, op) -> np.ndarray:
        # compute all singular values, sorted in ascending order.
        D = splinalg.svd(
            a=np.reshape(
                op.asarray(),
                (op.codim_size, op.dim_size),
            ),
            full_matrices=False,
            compute_uv=False,
        )
        return np.sort(D)

    @pytest.fixture(params=[1, 2])
    def _damp(self, request) -> float:
        # candidate dampening factors for .pinv() & .dagger()
        # We do not test damp=0 since ill-conditioning of some operators would require a lot of
        # manual _damp-correcting to work.
        return request.param

    @pytest.fixture
    def _data_pinv(self, data_pinv, xp, width, complex_valued) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.pinv()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_pinv["in_"])
        arr = xp.array(in_["arr"], dtype=width.value)
        arr = ct.chunk_array(arr, complex_valued)
        in_.update(arr=arr)
        data = dict(
            in_=in_,
            out=data_pinv["out"],
        )
        return data

    @pytest.fixture
    def _op_dagger(self, op, _damp) -> pxa.LinOp:
        op_d = op.dagger(damp=_damp)
        return op_d

    @pytest.fixture
    def _data_apply_dagger(self, _data_pinv) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.dagger().apply()`.
        #
        # Clone of _data_pinv, with all arguments unsupported by .apply() removed.
        data = copy.deepcopy(_data_pinv)
        data["in_"] = dict(arr=data["in_"]["arr"])
        return data

    @pytest.fixture(
        params=itertools.product(
            pxd.supported_array_modules(),
            pxrt.Width,
        )
    )
    def _op_array(self, op, xp, width, request) -> pxt.NDArray:
        # Ground-truth array which should be returned by .asarray()

        # op() only defined for specifid xp/width combos.
        # Idea: compute .asarray() using backend/precision supported by op(), then cast to
        # user-desired backend/precision.
        A_gt = xp.zeros((*op.codim_shape, *op.dim_shape), dtype=width.value)

        for i in range(op.dim_size):
            e = xp.zeros(op.dim_shape, dtype=width.value)
            idx = np.unravel_index(i, op.dim_shape)
            e[idx] = 1
            with pxrt.EnforcePrecision(False):
                A_gt[..., *idx] = op.apply(e)

        xp_, width_ = request.param
        A = xp_.array(pxu.to_NUMPY(A_gt), dtype=width_.value)
        return A

    # Tests -------------------------------------------------------------------
    def test_value1D_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_value1D(op.adjoint, _data_adjoint)

    def test_valueND_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_valueND(op.adjoint, _data_adjoint)

    def test_backend_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_backend(op.adjoint, _data_adjoint)

    def test_chunk_adjoint(self, op, ndi, _data_adjoint):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.adjoint, _data_adjoint, False)

    def test_prec_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_prec(op.adjoint, _data_adjoint)

    def test_precCM_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_precCM(op.adjoint, _data_adjoint)

    def test_transparent_adjoint(self, op, _data_adjoint):
        self._skip_if_disabled()
        self._check_no_side_effect(op.adjoint, _data_adjoint)

    def test_math_adjoint(self, op, xp, width):
        # <op.adjoint(x), y> = <x, op.apply(y)>
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, *op.codim_shape), xp=xp, width=width)
        y = self._random_array((N, *op.dim_shape), xp=xp, width=width)

        ip = lambda a, b, D: (a * b).sum(axis=tuple(range(-D, 0)))  # (N, M1,...,MD) * (N, M1,...,MD) -> (N,)
        with pxrt.EnforcePrecision(False):
            lhs = ip(op.adjoint(x), y, D=op.dim_rank)
            rhs = ip(x, op.apply(y), D=op.codim_rank)

        assert self._metric(lhs, rhs, as_dtype=width.value)

    def test_math2_lipschitz(self, op, _data_estimate_lipschitz, _gpu):
        # op.estimate_lipschitz(svd) <= op.estimate_lipschitz(trace)
        self._skip_if_disabled()
        kwargs = _data_estimate_lipschitz["in_"]
        dtype = kwargs["dtype"]

        L_ub = op.estimate_lipschitz(method="trace", **kwargs)
        L_opt = ct.flaky(  # might raise exception in GPU-mode
            op.estimate_lipschitz,
            args=dict(method="svd", **kwargs),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

        try:
            assert ct.less_equal(L_opt, L_ub, as_dtype=dtype).item()
        except AssertionError:
            if _gpu is True:
                # even if L_opt computed, not guaranteed to be correct in GPU-mode
                pytest.xfail("svdvals() sparse-evaled via CuPy flaky.")
            else:
                raise

    def test_math3_lipschitz(self, op, _data_estimate_lipschitz, _op_svd, _gpu):
        # op.estimate_lipschitz(svd) computes the optimal Lipschitz constant.
        self._skip_if_disabled()
        kwargs = _data_estimate_lipschitz["in_"]
        dtype = kwargs["dtype"]

        L_svds = ct.flaky(  # might raise an assertion in GPU-mode
            func=op.estimate_lipschitz,
            args=dict(method="svd", **kwargs),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )
        # Comparison is done with user-specified metric since operator may compute `L_svds` via an
        # approximate `.apply()' method.
        try:
            cast = lambda x: np.array([x], dtype=dtype)
            assert self._metric(cast(L_svds), cast(_op_svd.max()), as_dtype=dtype)
        except AssertionError:
            if _gpu is True:
                # even if L_opt computed, not guaranteed to be correct in GPU-mode
                pytest.xfail("svdvals() sparse-evaled via CuPy flaky.")
            else:
                raise

    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        assert J is op

    @pytest.mark.parametrize("k", [1, 2])
    def test_value1D_svdvals(self, op, ndi, width, _gpu, _op_svd, k):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(ndi)

        if k > len(_op_svd):
            pytest.skip("Not enough singular values. (Skipping higher-order tests.)")

        ct.flaky(
            func=self._check_value1D_vals,
            args=dict(
                func=op.svdvals,
                kwargs=dict(k=k, gpu=_gpu, dtype=width.value),
                ground_truth=_op_svd,
            ),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

    def test_backend_svdvals(self, op, ndi, width, _gpu):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(ndi)
        ct.flaky(
            func=self._check_backend_vals,
            args=dict(func=op.svdvals, _gpu=_gpu, width=width),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

    def test_precCM_svdvals(self, op, ndi, _gpu, width):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(ndi)
        data = dict(in_=dict(k=1, gpu=_gpu, dtype=width.value))
        ct.flaky(
            func=self._check_precCM,
            args=dict(func=op.svdvals, data=data, widths=(width,)),
            condition=_gpu is True,
            reason="svdvals() sparse-evaled via CuPy flaky.",
        )

    def test_value1D_pinv(self, op, _data_pinv):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_value1D(op.pinv, _data_pinv, dtype=np.dtype(np.half))

    def test_valueND_pinv(self, op, _data_pinv):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_valueND(op.pinv, _data_pinv, dtype=np.dtype(np.half))

    def test_backend_pinv(self, op, _data_pinv):
        self._skip_if_disabled()
        self._check_backend(op.pinv, _data_pinv)

    def test_chunk_pinv(self, op, ndi, _data_pinv):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_chunk(op.pinv, _data_pinv, False)

    def test_prec_pinv(self, op, _data_pinv):
        self._skip_if_disabled()
        self._check_prec(op.pinv, _data_pinv)

    def test_precCM_pinv(self, op, _data_pinv):
        self._skip_if_disabled()
        self._check_precCM(op.pinv, _data_pinv)

    def test_interface_dagger(self, _op_dagger):
        self._skip_if_disabled()
        self._check_has_interface(_op_dagger, LinOpT)

    def test_value1D_call_dagger(self, _op_dagger, _data_apply_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.__call__, _data_apply_dagger, dtype=np.dtype(np.half))

    def test_valueND_call_dagger(self, _op_dagger, _data_apply_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.__call__, _data_apply_dagger, dtype=np.dtype(np.half))

    def test_backend_call_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_backend(_op_dagger.__call__, _data_apply_dagger)

    def test_prec_call_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_prec(_op_dagger.__call__, _data_apply_dagger)

    def test_precCM_call_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_precCM(_op_dagger.__call__, _data_apply_dagger)

    def test_value1D_apply_dagger(self, _op_dagger, _data_apply_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.apply, _data_apply_dagger, dtype=np.dtype(np.half))

    def test_valueND_apply_dagger(self, _op_dagger, _data_apply_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.apply, _data_apply_dagger, dtype=np.dtype(np.half))

    def test_backend_apply_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_backend(_op_dagger.apply, _data_apply_dagger)

    def test_prec_apply_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_prec(_op_dagger.apply, _data_apply_dagger)

    def test_precCM_apply_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_precCM(_op_dagger.apply, _data_apply_dagger)

    def test_interface_gram(self, op):
        self._skip_if_disabled()
        self._check_has_interface(op.gram(), SelfAdjointOpT)

    def test_math_gram(self, op, xp, width):
        # op_g.apply == op_g.adjoint == adjoint \comp apply
        self._skip_if_disabled()
        op_g = op.gram()

        N_test = 30
        x = self._random_array((N_test, *op.dim_shape), xp=xp, width=width)

        with pxrt.EnforcePrecision(False):
            assert self._metric(op_g.apply(x), op_g.adjoint(x), as_dtype=width.value)
            assert self._metric(op_g.apply(x), op.adjoint(op.apply(x)), as_dtype=width.value)

    def test_interface_cogram(self, op):
        self._skip_if_disabled()
        self._check_has_interface(op.cogram(), SelfAdjointOpT)

    def test_math_cogram(self, op, xp, width):
        # op_cg.apply == op_cg.adjoint == apply \comp adjoint
        self._skip_if_disabled()
        op_cg = op.cogram()

        N_test = 30
        x = self._random_array((N_test, *op.codim_shape), xp=xp, width=width)

        with pxrt.EnforcePrecision(False):
            assert self._metric(op_cg.apply(x), op_cg.adjoint(x), as_dtype=width.value)
            assert self._metric(op_cg.apply(x), op.apply(op.adjoint(x)), as_dtype=width.value)

    def test_value_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pxu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert A.shape == _op_array.shape
        assert self._metric(_op_array, A, as_dtype=dtype)

    def test_backend_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pxu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert pxu.get_array_module(A) == xp

    def test_prec_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pxu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert A.dtype == dtype


class LinFuncT(ProxDiffFuncT, LinOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.LinFunc
    interface = frozenset(ProxDiffFuncT.interface | LinOpT.interface)
    disable_test = frozenset(ProxDiffFuncT.disable_test | LinOpT.disable_test)

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def data_grad(self, op) -> DataLike:
        # We know for linfuncs that op.grad(x) = op.asarray()
        x = self._random_array(op.dim_shape)
        y = op.asarray()[0]
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_adjoint(self, data_grad) -> DataLike:
        # We know for linfuncs that op.adj(x) = op.grad(x) * x
        x = self._random_array((1,))
        y = x * data_grad["out"]
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_prox(self, data_grad) -> DataLike:
        # We know for linfuncs that op.prox(x, tau) = x - op.grad(x) * tau
        x = data_grad["in_"]["arr"].copy()
        g = data_grad["out"]
        tau = np.abs(self._random_array((1,))).item()
        y = x - tau * g
        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )


class SquareOpT(LinOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.SquareOp
    interface = frozenset(LinOpT.interface | {"trace"})

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _op_trace(self, op) -> float:
        # True trace
        A = op.asarray()
        tr = 0
        for i in range(op.dim_size):
            codim_idx = np.unravel_index(i, op.codim_shape)
            dim_idx = np.unravel_index(i, op.dim_shape)
            tr += A[*codim_idx, *dim_idx]
        return float(tr)

    @pytest.fixture
    def _data_trace(self, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.trace()`.
        data = dict(
            in_=dict(
                method="explicit",  # hutchpp() variant is time-consuming: tested in test_linalg.py
                xp=xp,
                dtype=width.value,
            )
        )
        return data

    # Tests -------------------------------------------------------------------
    def test_square_size(self, op):
        self._skip_if_disabled()
        assert op.dim_size == op.codim_size

    def test_interface_trace(self, op, _data_trace):
        self._skip_if_disabled()
        tr = op.trace(**_data_trace["in_"])
        assert isinstance(tr, pxt.Real)

    def test_value_trace(self, op, _data_trace, _op_trace, width):
        # tr(op) == op.trace(method=explicit)
        self._skip_if_disabled()
        tr = op.trace(**_data_trace["in_"])
        assert ct.allclose(tr, _op_trace, as_dtype=width.value)

    def test_precCM_trace(self, op, _data_trace, width):
        self._skip_if_disabled()
        func = lambda: op.trace(**_data_trace["in_"])
        self._check_precCM_func(func, width)


class NormalOpT(SquareOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.NormalOp

    # Tests -------------------------------------------------------------------
    def test_math_normality(self, op, xp, width):
        # AA^{*} = A^{*}A
        self._skip_if_disabled()
        N = 20

        x = self._random_array((N, *op.codim_shape), xp=xp, width=width)
        lhs = op.apply(op.adjoint(x))

        y = x.reshape((N, *op.dim_shape))
        rhs = op.adjoint(op.apply(y))

        assert self._metric(
            lhs.reshape(N, -1),
            rhs.reshape(N, -1),
            as_dtype=width.value,
        )


class UnitOpT(NormalOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.UnitOp

    # Internal helpers --------------------------------------------------------
    @classmethod
    def _check_identity(cls, operator, xp, width):
        N_test = 30
        x = cls._random_array((N_test, *operator.dim_shape), xp=xp, width=width)
        assert ct.allclose(operator.apply(x), x, as_dtype=width.value)
        assert ct.allclose(operator.adjoint(x), x, as_dtype=width.value)

    # Fixtures (Internal) -----------------------------------------------------
    @pytest.fixture
    def _op_svd(self, op) -> np.ndarray:
        D = np.ones(op.dim_size)
        return D

    # Tests -------------------------------------------------------------------
    def test_math_gram(self, op, xp, width):
        # op_g == I
        self._skip_if_disabled()
        self._check_identity(op.gram(), xp, width)

    def test_math_cogram(self, op, xp, width):
        # op_cg == I
        self._skip_if_disabled()
        self._check_identity(op.cogram(), xp, width)

    def test_math_norm(self, op, xp, width):
        # norm preservation
        self._skip_if_disabled()
        N = 30

        # \norm{U x} = \norm{x}
        x1 = self._random_array((N, *op.dim_shape), xp=xp, width=width)
        lhs1 = xp.sum(op.apply(x1) ** 2, axis=tuple(range(-op.codim_rank, 0)))
        rhs1 = xp.sum(x1**2, axis=tuple(range(-op.dim_rank, 0)))
        assert self._metric(lhs1, rhs1, as_dtype=width.value)

        # \norm{U^{*} x} = \norm{x}
        x2 = self._random_array((N, *op.codim_shape), xp=xp, width=width)
        lhs2 = xp.sum(op.adjoint(x2) ** 2, axis=tuple(range(-op.dim_rank, 0)))
        rhs2 = xp.sum(x2**2, axis=tuple(range(-op.codim_rank, 0)))
        assert self._metric(lhs2, rhs2, as_dtype=width.value)


class SelfAdjointOpT(NormalOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.SelfAdjointOp

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    # Tests -------------------------------------------------------------------
    def test_square_shape(self, op):
        # dim_shape == codim_shape
        #
        # This test is stricter than test_square_size().
        # Reason: self-adjoint-ness implies apply/adjoint can be swapped, but
        #         this only holds in dim/codim have identical shapes.
        self._skip_if_disabled()
        assert op.dim_shape == op.codim_shape

    def test_math_selfadjoint(self, op, xp, width):
        # A = A^{*}
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, *op.dim_shape), xp=xp, width=width)

        lhs = op.apply(x)
        rhs = op.adjoint(x)
        assert ct.allclose(lhs, rhs, as_dtype=width.value)


class PosDefOpT(SelfAdjointOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.PosDefOp

    # Tests -------------------------------------------------------------------
    def test_math_posdef(self, op, xp, width):
        # <Ax,x> > 0
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, *op.dim_shape), xp=xp, width=width)

        ip = lambda a, b, D: (a * b).sum(axis=tuple(range(-D, 0)))  # (N, M1,...,MD) * (N, M1,...,MD) -> (N,)
        assert np.all(
            ct.less_equal(
                0,
                ip(op.apply(x), x, D=op.dim_rank),
                as_dtype=width.value,
            )
        )


class ProjOpT(SquareOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.ProjOp

    # Fixtures (Public-Facing; auto-inferred) ---------------------------------
    #           but can be overidden manually if desired ----------------------
    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    # Tests -------------------------------------------------------------------
    def test_square_shape(self, op):
        # dim_shape == codim_shape
        self._skip_if_disabled()
        SelfAdjointOpT.test_square_shape(self, op)

    def test_math_idempotent(self, op, xp, width):
        # op.apply = op.apply^2
        self._skip_if_disabled()
        N = 30
        x = self._random_array((N, *op.dim_shape), xp=xp, width=width)
        y = op.apply(x)
        z = op.apply(y)

        assert self._metric(y, z, as_dtype=width.value)


class OrthProjOpT(ProjOpT, SelfAdjointOpT):
    # Class Properties --------------------------------------------------------
    base = pxa.OrthProjOp
    interface = frozenset(ProjOpT.interface | SelfAdjointOpT.interface)
    disable_test = frozenset(ProjOpT.disable_test | SelfAdjointOpT.disable_test)
