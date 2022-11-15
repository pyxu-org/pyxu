import collections.abc as cabc
import contextlib
import copy
import inspect
import itertools
import math
import types
import typing as typ
import warnings

import dask.array as da
import numpy as np
import numpy.random as npr
import pytest
import scipy.linalg as splinalg
import scipy.sparse.linalg as spsl

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.complex as pycuc
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw
from pycsou.abc.operator import _core_operators


def get_test_class(cls: pyct.OpC) -> "MapT":
    # Find the correct MapT subclass designed to test `cls`.
    is_test = lambda _: all(
        [
            hasattr(_, "base"),
            hasattr(_, "interface"),
            hasattr(_, "disable_test"),
        ]
    )
    candidates = {_ for _ in globals().values() if is_test(_)}
    for clsT in candidates:
        if clsT.base == cls:
            return clsT
    else:
        raise ValueError(f"No known test type for {cls}.")


def isclose(
    a: typ.Union[pyct.Real, pyct.NDArray],
    b: typ.Union[pyct.Real, pyct.NDArray],
    as_dtype: pyct.DType,
) -> pyct.NDArray:
    """
    Equivalent of `xp.isclose`, but where atol is automatically chosen based on `as_dtype`.

    This function always returns a computed array, i.e. NumPy/CuPy output.
    """
    atol = {
        np.dtype(np.half): 3e-2,  # former pycrt.Width.HALF
        pycrt.Width.SINGLE.value: 2e-4,
        pycrt._CWidth.SINGLE.value: 2e-4,
        pycrt.Width.DOUBLE.value: 1e-8,
        pycrt._CWidth.DOUBLE.value: 1e-8,
    }
    # Numbers obtained by:
    # * \sum_{k >= (p+1)//2} 2^{-k}, where p=<number of mantissa bits>; then
    # * round up value to 3 significant decimal digits.
    # N_mantissa = [10, 23, 52, 112] for [half, single, double, quad] respectively.

    if (prec := atol.get(as_dtype)) is None:
        # should occur for integer types only
        prec = atol[pycrt.Width.DOUBLE.value]
    cast = lambda x: pycu.compute(x)
    eq = np.isclose(cast(a), cast(b), atol=prec)
    return eq


def allclose(
    a: pyct.NDArray,
    b: pyct.NDArray,
    as_dtype: pyct.DType,
) -> bool:
    """
    Equivalent of `all(isclose)`, but where atol is automatically chosen based on `as_dtype`.
    """
    return bool(np.all(isclose(a, b, as_dtype)))


def less_equal(
    a: pyct.NDArray,
    b: pyct.NDArray,
    as_dtype: pyct.DType,
) -> pyct.NDArray:
    """
    Equivalent of `a <= b`, but where equality tests are done at a chosen numerical precision.

    This function always returns a computed array, i.e. NumPy/CuPy output.
    """
    x = pycu.compute(a <= b)
    y = isclose(a, b, as_dtype)
    return x | y


# Naming conventions
# ------------------
#
# * data_<property>()
#       Return expected object of `op.<property>`.
#
# * test_<property>(op, ...):
#       Verify property values.
#
# * data_<method>(op, ...)
#       Return mappings of the form dict(in_=dict(), out=Any), where:
#         * in_ are kwargs to `op.<method>()`;
#         * out denotes the output of `op.method(**data[in_])`.
#
# * test_[value,backend,prec,precCM,transparent]_<method>(op, ...)
#       Verify that <method>, returns
#       * value: right output values
#       * backend: right output type
#       * prec: input/output have same precision
#       * precCM: output respects context-manager choice
#       * transparent: referential-transparency, i.e. no side-effects.
#
# * data_math_<method>()
#       Special test data for mathematical identities.
#
# * test_math_<method>()
#       Verify mathematical identities involving <method>.
#
# * test_interface_[<method>]()
#       Verify objects have the right interface.
DataLike = cabc.Mapping[str, typ.Any]


class MapT:
    # Class Properties --------------------------------------------------------
    base = pyca.Map
    disable_test: cabc.Set[str] = frozenset()
    interface: cabc.Set[str] = frozenset(
        {
            "shape",
            "dim",
            "codim",
            "asop",
            "argshift",
            "argscale",
            "apply",
            "__call__",
            "lipschitz",
            "expr",
            "squeeze",
        }
    )

    # Internal helpers --------------------------------------------------------
    @staticmethod
    def _random_array(
        shape: pyct.NDArrayShape,
        seed: int = 0,
        xp: pyct.ArrayModule = np,
        width: pycrt.Width = pycrt.Width.DOUBLE,
    ):
        rng = npr.default_rng(seed)
        x = rng.normal(size=shape)
        return xp.array(x, dtype=width.value)

    @staticmethod
    def _sanitize(x, default):
        if x is not None:
            return x
        else:
            return default

    def _skip_if_disabled(self):
        # Get name of function which invoked me.
        my_frame = inspect.currentframe()
        up_frame = inspect.getouterframes(my_frame)[1].frame
        up_finfo = inspect.getframeinfo(up_frame)
        up_fname = up_finfo.function
        print(up_fname)
        if up_fname in self.disable_test:
            pytest.skip("disabled test")

    @staticmethod
    def _check_has_interface(op: pyca.Map, klass: "MapT"):
        # Verify `op` has the public interface of `klass`.
        assert klass.interface <= frozenset(dir(op))

    @classmethod
    def _metric(
        cls,
        a: pyct.NDArray,
        b: pyct.NDArray,
        as_dtype: pyct.DType,
    ) -> bool:
        # Function used to assess if computed values returned by arithmetic methods are correct.
        #
        # Users may override this function to introduce an alternative metrics when justified.
        # The default metric is point-wise match.
        #
        # Parameters
        # ----------
        # a, b: pyct.NDArray
        #    (..., N) arrays
        # as_dtype: pyct.DType
        #    dtype used to compare the values. (Not always relevant depending on the metric.)
        #
        # Returns
        # -------
        # match: bool
        #    True if all (...) arrays match.
        return allclose(a, b, as_dtype)

    @classmethod
    def _check_value1D(
        cls,
        func,
        data: DataLike,
        dtype: pyct.DType = None,
    ):
        in_ = data["in_"]
        with pycrt.EnforcePrecision(False):
            out = func(**in_)
        out_gt = data["out"]

        dtype = MapT._sanitize(dtype, in_["arr"].dtype)
        assert out.ndim == in_["arr"].ndim
        assert cls._metric(out, out_gt, as_dtype=dtype)

    @classmethod
    def _check_valueND(
        cls,
        func,
        data: DataLike,
        dtype: pyct.DType = None,
    ):
        sh_extra = (2, 1, 3)  # prepend input/output shape by this amount.

        in_ = data["in_"]
        arr = in_["arr"]
        xp = pycu.get_array_module(arr)
        arr = xp.broadcast_to(arr, (*sh_extra, *arr.shape))
        in_.update(arr=arr)
        with pycrt.EnforcePrecision(False):
            out = func(**in_)
        out_gt = np.broadcast_to(data["out"], (*sh_extra, *data["out"].shape))

        dtype = MapT._sanitize(dtype, in_["arr"].dtype)
        assert out.ndim == in_["arr"].ndim
        assert cls._metric(out, out_gt, as_dtype=dtype)

    @staticmethod
    def _check_backend(func, data: DataLike):
        in_ = data["in_"]
        with pycrt.EnforcePrecision(False):
            out = func(**in_)

        assert type(out) == type(in_["arr"])

    @staticmethod
    def _check_prec(func, data: DataLike):
        in_ = data["in_"]
        with pycrt.EnforcePrecision(False):
            out = func(**in_)
            assert out.dtype == in_["arr"].dtype

    @staticmethod
    def _check_precCM(
        func,
        data: DataLike,
        widths: cabc.Collection[pycrt.Width] = pycrt.Width,
    ):
        stats = dict()
        for w in widths:
            with pycrt.Precision(w):
                out = func(**data["in_"])
            stats[w] = out.dtype == w.value
        assert all(stats.values())

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

        with pycrt.EnforcePrecision(False):
            out_1 = func(**in_)
            if pycu.copy_if_unsafe(out_1) is not out_1:
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
                assert cls._metric(out_1, out_gt, as_dtype=pycrt.Width.SINGLE.value)
                assert cls._metric(out_2, out_gt, as_dtype=pycrt.Width.SINGLE.value)
            except AssertionError as exc:
                # Function is non-transparent, but which backend caused it?
                N = pycd.NDArrayInfo
                ndi = N.from_obj(out_1)
                if ndi == N.CUPY:
                    # warn about CuPy-only non-transparency.
                    msg = "\n".join(
                        [
                            f"{func} is not transparent when applied to CuPy inputs.",
                            f"If the same test fails for non-CuPy inputs, then {func}'s implementation is at fault -> user fix required.",
                            f"If the same test passes for non-CuPy inputs, then this warning can be safely ignored.",
                        ]
                    )
                    warnings.warn(msg, pycuw.NonTransparentWarning)
                else:
                    raise

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def spec(self) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        # override in subclass to return:
        # * the operator to test;
        # * the backend of accepted input arrays;
        # * the precision of accepted input arrays.
        #
        # The triplet (op, backend, precision) must be provided since some operators may not be
        # backend/precision-agnostic.
        raise NotImplementedError

    @pytest.fixture
    def op(self, spec) -> pyct.OpT:
        return spec[0]

    @pytest.fixture
    def xp(self, spec) -> pyct.ArrayModule:
        ndi = spec[1]
        if (xp_ := ndi.module()) is not None:
            return xp_
        else:
            pytest.skip(f"{ndi} unsupported on this machine.")

    @pytest.fixture
    def width(self, spec) -> pycrt.Width:
        return spec[2]

    @pytest.fixture(params=_core_operators())
    def _klass(self, request) -> pyct.OpC:
        # Returns some operator in the Operator hierarchy.
        # Do not override in subclass: for internal use only to test `op.asop()`.
        return request.param

    @pytest.fixture
    def data_shape(self) -> pyct.OpShape:
        # override in subclass with the shape of op.
        # Don't return `op.shape`: hard-code what you are expecting.
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.apply().
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

    @pytest.fixture
    def _data_apply(self, data_apply, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.apply()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_apply["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
        data = dict(
            in_=in_,
            out=data_apply["out"],
        )
        return data

    # Tests -------------------------------------------------------------------
    def test_interface(self, op):
        self._skip_if_disabled()
        self._check_has_interface(op, self.__class__)

    def test_shape(self, op, data_shape):
        self._skip_if_disabled()
        assert op.shape == data_shape

    def test_dim(self, op, data_shape):
        self._skip_if_disabled()
        assert op.dim == data_shape[1]

    def test_codim(self, op, data_shape):
        self._skip_if_disabled()
        assert op.codim == data_shape[0]

    def test_lipschitz(self, op, width):
        # Ensure:
        # * _lipschitz matches .lipschitz() after being called once.
        self._skip_if_disabled()
        L_computed = op.lipschitz()
        L_memoized = op._lipschitz
        assert allclose(L_computed, L_memoized, as_dtype=width.value)

    def test_value1D_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_value1D(op.apply, _data_apply)

    def test_valueND_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_valueND(op.apply, _data_apply)

    def test_backend_apply(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_backend(op.apply, _data_apply)

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

    def test_prec_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_prec(op.__call__, _data_apply)

    def test_precCM_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_precCM(op.__call__, _data_apply)

    def test_transparent_call(self, op, _data_apply):
        self._skip_if_disabled()
        self._check_no_side_effect(op.__call__, _data_apply)

    # todo: nan=(np.inf-np.inf)<np.inf returns False
    def test_math_lipschitz(
        self,
        op,
        xp,
        width,
        data_math_lipschitz,
    ):
        # \norm{f(x) - f(y)}{2} \le L * \norm{x - y}{2}
        self._skip_if_disabled()
        with pycrt.EnforcePrecision(False):
            L = op.lipschitz()

            data = xp.array(data_math_lipschitz, dtype=width.value)
            data = list(itertools.combinations(data, 2))
            x = xp.stack([_[0] for _ in data], axis=0)
            y = xp.stack([_[1] for _ in data], axis=0)

            lhs = pylinalg.norm(op.apply(x) - op.apply(y), axis=-1)
            rhs = L * pylinalg.norm(x - y, axis=-1)
            success = less_equal(lhs, rhs, as_dtype=width.value)
            assert all(success)

    def test_interface_asop(self, op, _klass):
        # * .asop() is no-op if same klass or parent
        # * .asop() has correct interface otherwise.
        # * Expect an exception if cast is illegal. (shape-issues, domain-agnostic linfuncs)
        self._skip_if_disabled()
        P = pyca.Property
        if _klass in op.__class__.__mro__:
            op2 = op.asop(_klass)
            assert op2 is op
        elif (P.FUNCTIONAL in _klass.properties()) and (op.codim > 1):
            # Casting to functionals when codim != 1.
            with pytest.raises(Exception):
                op2 = op.asop(_klass)
        elif ({P.FUNCTIONAL, P.LINEAR} <= _klass.properties()) and (op.dim is None):
            # Casting to domain-agnostic linear functionals.
            with pytest.raises(Exception):
                op2 = op.asop(_klass)
        elif (P.LINEAR_SQUARE in _klass.properties()) and (op.codim != op.dim):
            # Casting non-square operators to square linops.
            # Includes domain-agnostic functionals.
            with pytest.raises(Exception):
                op2 = op.asop(_klass)
        else:
            op2 = op.asop(_klass)
            klassT = get_test_class(_klass)
            self._check_has_interface(op2, klassT)

    def test_value_asop(self, op, width, _klass):
        # Ensure encapsulated arithmetic fields are forwarded.
        # We only test fields known to belong to any Map subclass:
        # * _lipschitz (attribute)
        # * lipschitz (method)
        # * apply (method)
        # * __call__ (special method)
        self._skip_if_disabled()
        try:
            op2 = op.asop(_klass)
        except:
            # nothing to test since `op.asop(_klass)` is illegal.
            return
        else:
            assert allclose(op2._lipschitz, op._lipschitz, as_dtype=width.value)
            assert allclose(op2.lipschitz(), op.lipschitz(), as_dtype=width.value)
            assert op2.apply == op.apply
            assert op2.__call__ == op.__call__


class FuncT(MapT):
    # Class Properties --------------------------------------------------------
    base = pyca.Func
    interface = frozenset(MapT.interface | {"asloss"})

    # Tests -------------------------------------------------------------------
    def test_codim(self, op):
        self._skip_if_disabled()
        assert op.codim == 1

    def test_interface_asloss(self, op, xp, width):
        # op.asloss() sub-classes Func if data provided, transparent otherwise.
        # Disable this test if asloss() not defined for a functional.
        self._skip_if_disabled()

        N_dim = self._sanitize(op.dim, default=3)
        data = self._random_array((N_dim,), xp=xp, width=width)
        try:
            assert op.asloss() is op
            self._check_has_interface(op.asloss(data), self.__class__)
        except NotImplementedError as exc:
            msg = f"{op.asloss} is undefined, but `test_interface_asloss()` was not disabled."
            assert False, msg


class DiffMapT(MapT):
    # Class Properties --------------------------------------------------------
    base = pyca.DiffMap
    interface = frozenset(
        MapT.interface
        | {
            "jacobian",
            "diff_lipschitz",
        }
    )

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def data_math_diff_lipschitz(self) -> cabc.Collection[np.ndarray]:
        # override in subclass with at least 2 evaluation points for op.apply().
        # Used to verify if op.jacobian() satisfies the diff_Lipschitz condition.
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    # Tests -------------------------------------------------------------------
    def test_diff_lipschitz(self, op, width):
        # Ensure:
        # * _diff_lipschitz matches .diff_lipschitz() after being called once.
        self._skip_if_disabled()
        dL_computed = op.diff_lipschitz()
        dL_memoized = op._diff_lipschitz
        assert allclose(dL_computed, dL_memoized, as_dtype=width.value)

    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        self._check_has_interface(J, LinOpT)

    def test_math_diff_lipschitz(
        self,
        op,
        xp,
        width,
        data_math_diff_lipschitz,
    ):
        # \norm{J(x) - J(y)}{F} \le diff_L * \norm{x - y}{2}
        self._skip_if_disabled()
        with pycrt.EnforcePrecision(False):
            dL = op.diff_lipschitz()

            J = lambda _: op.jacobian(_).asarray(dtype=width.value).flatten()
            # .flatten() used to consistently compare jacobians via the L2 norm.
            # (Allows one to re-use this test for scalar-valued DiffMaps.)

            stats = []  # (x, y, condition success)
            data = xp.array(data_math_diff_lipschitz, dtype=width.value)
            for x, y in itertools.combinations(data, 2):
                lhs = pylinalg.norm(J(x) - J(y))
                rhs = dL * pylinalg.norm(x - y)
                success = less_equal(lhs, rhs, as_dtype=width.value)
                stats.append((lhs, rhs, success))

            assert all(_[2] for _ in stats)


class DiffFuncT(FuncT, DiffMapT):
    # Class Properties --------------------------------------------------------
    base = pyca.DiffFunc
    interface = frozenset(FuncT.interface | DiffMapT.interface | {"grad"})
    disable_test = frozenset(FuncT.disable_test | DiffMapT.disable_test)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def data_grad(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.grad().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    @pytest.fixture
    def _data_grad(self, data_grad, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.grad()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_grad["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
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
        with pycrt.EnforcePrecision(False):
            J = op.jacobian(arr).asarray(dtype=width.value)
            g = op.grad(arr)

            assert J.size == g.size
            assert allclose(J.squeeze(), g, as_dtype=arr.dtype)

    def test_math2_grad(self, op, xp, width):
        # f(x - \frac{1}{L} \grad_{f}(x)) <= f(x)
        self._skip_if_disabled()
        with pycrt.EnforcePrecision(False):
            L = op.lipschitz()
            if np.isclose(L, 0):
                return  # trivially true since f(x) = cst
            elif np.isclose(L, np.inf):
                return  # trivially true since f(x) <= f(x)

            N_test, N_dim = 5, self._sanitize(op.dim, default=3)
            rhs = self._random_array((N_test, N_dim), xp=xp, width=width)
            lhs = rhs - op.grad(rhs) / L

            assert np.all(less_equal(op.apply(lhs), op.apply(rhs), as_dtype=width.value))


class ProxFuncT(FuncT):
    # Class Properties --------------------------------------------------------
    base = pyca.ProxFunc
    interface = frozenset(
        FuncT.interface
        | {
            "prox",
            "fenchel_prox",
            "moreau_envelope",
        }
    )

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def _op_m(self, op) -> tuple[float, pyca.DiffFunc]:
        mu = 1.1
        return mu, op.moreau_envelope(mu)

    @pytest.fixture
    def data_prox(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.prox().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    @pytest.fixture
    def data_fenchel_prox(self, data_prox) -> DataLike:
        # override in subclass with 1D input/outptus of op.fenchel_prox().
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

    @pytest.fixture
    def _data_prox(self, data_prox, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.prox()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_prox["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
        data = dict(
            in_=in_,
            out=data_prox["out"],
        )
        return data

    @pytest.fixture
    def _data_fenchel_prox(self, data_fenchel_prox, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.fenchel_prox()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_fenchel_prox["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
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
        # 2\tau f(z) - \norm{z - x}{2}^{2}, for any z \in \bR^{N}
        self._skip_if_disabled()
        in_ = _data_prox["in_"]
        y = op.prox(**in_)

        N_test, N_dim = 5, y.shape[-1]
        x = self._random_array((N_test, N_dim), xp=xp, width=width) + in_["arr"]

        def g(x):
            a = 2 * in_["tau"] * op.apply(x)
            b = pylinalg.norm(in_["arr"] - x, axis=-1, keepdims=True) ** 2
            return a + b

        assert np.all(less_equal(g(y), g(x), as_dtype=y.dtype))

    def test_value1D_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_value1D(op.fenchel_prox, _data_fenchel_prox)

    def test_valueND_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_valueND(op.fenchel_prox, _data_fenchel_prox)

    def test_backend_fenchel_prox(self, op, _data_fenchel_prox):
        self._skip_if_disabled()
        self._check_backend(op.fenchel_prox, _data_fenchel_prox)

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

        assert less_equal(lhs, rhs, as_dtype=rhs.dtype)

    def test_math2_moreau_envelope(self, op, _op_m, _data_apply):
        # op_m.grad(x) * mu = x - op.prox(x, mu)
        self._skip_if_disabled()
        mu, op_m = _op_m
        arr = _data_apply["in_"]["arr"]
        lhs = op_m.grad(arr) * mu
        rhs = arr - op.prox(arr, mu)

        assert allclose(lhs, rhs, as_dtype=arr.dtype)


class ProxDiffFuncT(ProxFuncT, DiffFuncT):
    # Class Properties --------------------------------------------------------
    base = pyca.ProxDiffFunc
    interface = frozenset(ProxFuncT.interface | DiffFuncT.interface)


class _QuadraticFuncT(ProxDiffFuncT):
    # Class Properties --------------------------------------------------------
    base = pyca._QuadraticFunc
    interface = frozenset(ProxDiffFuncT.interface | {"_hessian"})

    @pytest.fixture
    def data_math_lipschitz(self, op):
        N_test, dim = 5, self._sanitize(op.dim, 3)
        return self._random_array((N_test, dim), seed=5)

    @pytest.fixture
    def data_math_diff_lipschitz(self, op):
        N_test, dim = 5, self._sanitize(op.dim, 3)
        return self._random_array((N_test, dim), seed=5)

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


@pytest.mark.filterwarnings("ignore::pycsou.util.warning.DenseWarning")
class LinOpT(DiffMapT):
    # Class Properties --------------------------------------------------------
    base = pyca.LinOp
    interface = frozenset(
        DiffMapT.interface
        | {
            "adjoint",
            "asarray",
            "cogram",
            "dagger",
            "from_array",
            "from_sciop",
            "gram",
            "pinv",
            "svdvals",
            "T",
            "to_sciop",
        }
    )

    # Internal helpers --------------------------------------------------------
    def _skip_unless_NUMPY_CUPY(self, xp, gpu):
        N = pycd.NDArrayInfo
        xp_ = {True: N.CUPY, False: N.NUMPY}[gpu].module()
        if xp != xp_:
            pytest.skip("Only NUMPY/CUPY backends supported.")

    coupled_gpu_which = pytest.mark.parametrize(
        ["_gpu", "which"],
        [
            (False, "LM"),
            (False, "SM"),
            pytest.param(
                *(True, "LM"),
                marks=pytest.mark.xfail(
                    reason="`which=LM` sparse-evaled via CuPy flaky.",
                    strict=False,  # fails based on matrix structure.
                ),
            ),
            pytest.param(
                *(True, "SM"),
                marks=pytest.mark.xfail(
                    True,
                    reason="`which=SM` unsupported by CuPy",
                    strict=False,
                ),
            ),
        ],
    )

    @classmethod
    def _check_value1D_vals(cls, func, kwargs, ground_truth):
        N = pycd.NDArrayInfo
        xp = {True: N.CUPY, False: N.NUMPY}[kwargs["gpu"]].module()
        if kwargs["gpu"]:
            ground_truth = xp.array(ground_truth)

        k = kwargs["k"]
        out = func(**kwargs)
        idx = xp.argsort(xp.abs(out))
        assert out.size == k  # obtain N_vals asked for
        assert allclose(out[idx], out, out.dtype)  # sorted in ascending magnitude

        # and output is correct (in magnitude)
        which = kwargs["which"]
        idx_gt = xp.argsort(xp.abs(ground_truth))
        if which == "SM":
            out = out[idx][:k]
            gt = ground_truth[idx_gt][:k]
        else:  # LM
            out = out[idx][-k:]
            gt = ground_truth[idx_gt][-k:]

        # When seeking the smallest-magnitude singular values via svdvals(), the iterative algorithm
        # used converges to values close to the ground-truth (GT).
        # However, for 0-valued singular vectors, the relative error between converged solution and
        # GT is sometimes slightly higher than the relative tolerance set for FP32/FP64 to consider
        # them identical.
        # We therefore assess [svd,eig]vals() outputs at FP32-precision only.
        # This precision is enough to query an operator's spectrum for further diagnostics.
        assert cls._metric(xp.abs(out), xp.abs(gt), as_dtype=pycrt.Width.SINGLE.value)

    @staticmethod
    def _check_backend_vals(func, _gpu):
        N = pycd.NDArrayInfo
        xp_truth = {True: N.CUPY, False: N.NUMPY}[_gpu].module()

        out = func(k=1, gpu=_gpu)
        assert pycu.get_array_module(out) == xp_truth

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def data_adjoint(self, op) -> DataLike:
        # override in subclass with 1D input/outputs of op.adjoint().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation just tests A(0) = 0. Subsequent math tests ensure .adjoint() output
        # is consistent with .apply().
        return dict(
            in_=dict(arr=np.zeros(op.codim)),
            out=np.zeros(op.dim),
        )

    @pytest.fixture
    def _data_adjoint(self, data_adjoint, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.adjoint()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_adjoint["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
        data = dict(
            in_=in_,
            out=data_adjoint["out"],
        )
        return data

    @pytest.fixture
    def data_math_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, op) -> cabc.Collection[np.ndarray]:
        N_test = 5
        return self._random_array((N_test, op.dim))

    @pytest.fixture
    def _op_svd(self, op) -> np.ndarray:
        # compute all singular values, sorted in ascending order.
        D = np.linalg.svd(
            op.asarray(),
            full_matrices=False,
            compute_uv=False,
        )
        return np.sort(D)

    @pytest.fixture
    def _op_T(self, op) -> pyca.LinOp:
        op_T = op.T
        return op_T

    @pytest.fixture
    def _op_TT(self, _op_T) -> pyca.LinOp:
        op_TT = _op_T.T
        return op_TT

    @pytest.fixture(
        params=[
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not pycd.CUPY_ENABLED,
                    reason="GPU unsupported on this machine.",
                ),
            ),
        ]
    )
    def _gpu(self, request) -> bool:
        # Do not override in subclass: for use only to test methods taking a `gpu` parameter.
        return request.param

    @pytest.fixture(params=[1, 2])
    def _damp(self, request) -> float:
        # candidate dampening factors for .pinv() & .dagger()
        # We do not test damp=0 since ill-conditioning of some operators would require a lot of
        # manual _damp-correcting to work.
        return request.param

    @pytest.fixture
    def data_pinv(self, op, _damp, data_apply) -> DataLike:
        # override in subclass with 1D input/outputs of op.pinv().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation: auto-computes pinv() at output points specified to test op.apply().

        # Safe implementation of --------------------------
        #   A = op.gram().asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        B = op.asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        arr = data_apply["out"]
        out, *_ = splinalg.lstsq(A, B.T @ arr)
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def _data_pinv(self, data_pinv, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.pinv()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_pinv["in_"])
        in_.update(arr=xp.array(in_["arr"], dtype=width.value))
        data = dict(
            in_=in_,
            out=data_pinv["out"],
        )
        return data

    @pytest.fixture
    def _op_dagger(self, op, _damp) -> pyca.LinOp:
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

    @pytest.fixture
    def data_pinvT(self, op, _damp, data_apply) -> DataLike:
        # override in subclass with 1D input/outputs of op.dagger().adjoint().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation: auto-computes .adjoint() at input points specified to test op.apply().

        # Safe implementation of --------------------------
        #   A = op.gram().asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        B = op.asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        A = B.T @ B
        # -------------------------------------------------
        for i in range(op.dim):
            A[i, i] += _damp

        arr = data_apply["in_"]["arr"]
        out, *_ = splinalg.lstsq(A, arr)
        out = B @ out
        data = dict(
            in_=dict(
                arr=arr,
                damp=_damp,
                kwargs_init=dict(),
                kwargs_fit=dict(),
            ),
            out=out,
        )
        return data

    @pytest.fixture
    def _data_adjoint_dagger(self, data_pinvT, xp, width) -> DataLike:
        # Generate Cartesian product of inputs.
        # Do not override in subclass: for internal use only to test `op.dagger().adjoint()`.
        # Outputs are left unchanged: different tests should transform them as required.
        in_ = copy.deepcopy(data_pinvT["in_"])
        data = dict(
            in_=dict(arr=xp.array(in_["arr"], dtype=width.value)),
            out=data_pinvT["out"],
        )
        return data

    @pytest.fixture(
        params=itertools.product(
            pycd.supported_array_modules(),
            pycrt.Width,
        )
    )
    def _op_array(self, op, xp, width, request) -> pyct.NDArray:
        # Ground-truth array which should be returned by .asarray()

        # op() only defined for specifid xp/width combos.
        # Idea: compute .asarray() using backend/precision supported by op(), then cast to
        # user-desired backend/precision.
        A_gt = xp.zeros((op.codim, op.dim), dtype=width.value)
        for i in range(op.dim):
            e = xp.zeros((op.dim,), dtype=width.value)
            e[i] = 1
            with pycrt.EnforcePrecision(False):
                A_gt[:, i] = op.apply(e)

        N = pycd.NDArrayInfo
        xp_, width_ = request.param
        if N.from_obj(A_gt) == N.CUPY:
            A_gt = A_gt.get()
        return xp_.array(A_gt, dtype=width_.value)

    @pytest.fixture
    def _op_sciop(self, op, _gpu, width) -> spsl.LinearOperator:
        A = op.to_sciop(dtype=width.value, gpu=_gpu)
        return A

    @pytest.fixture(
        params=[
            "matvec",
            "matmat",
            "rmatvec",
            "rmatmat",
        ]
    )
    def _data_to_sciop(self, op, xp, width, _gpu, request) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.to_sciop()`.
        self._skip_unless_NUMPY_CUPY(xp, _gpu)

        N_test = 7
        f = lambda _: self._random_array(_, xp=xp, width=width)
        op_array = op.asarray(xp=xp, dtype=width.value)
        mode = request.param
        if mode == "matvec":
            arr = f((op.dim,))
            out_gt = op_array @ arr
            var = "x"
        elif mode == "matmat":
            arr = f((op.dim, N_test))
            out_gt = op_array @ arr
            var = "X"
        elif mode == "rmatvec":
            arr = f((op.codim,))
            out_gt = op_array.T @ arr
            var = "x"
        elif mode == "rmatmat":
            arr = f((op.codim, N_test))
            out_gt = op_array.T @ arr
            var = "X"
        return dict(
            in_={var: arr},
            out=out_gt,
            mode=mode,  # for test_xxx_sciop()
        )

    @pytest.fixture
    def _data_from_sciop(self, _data_to_sciop) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.from_sciop()`.
        mode = _data_to_sciop["mode"]
        var = dict(
            matvec="x",
            matmat="X",
            rmatvec="x",
            rmatmat="X",
        )[mode]
        mode = dict(
            matvec="apply",
            matmat="apply",
            rmatvec="adjoint",
            rmatmat="adjoint",
        )[mode]

        arr = _data_to_sciop["in_"][var].T
        out = _data_to_sciop["out"].T
        return dict(
            in_=dict(arr=arr),
            out=out,
            mode=mode,
        )

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
        x = self._random_array((N, op.codim), xp=xp, width=width)
        y = self._random_array((N, op.dim), xp=xp, width=width)

        ip = lambda a, b: (a * b).sum(axis=-1)  # (N, Q) * (N, Q) -> (N,)
        with pycrt.EnforcePrecision(False):
            lhs = ip(op.adjoint(x), y)
            rhs = ip(x, op.apply(y))

        assert self._metric(lhs, rhs, as_dtype=width.value)

    def test_math2_lipschitz(self, op, xp, width):
        # op.lipschitz('fro') upper bounds op.lipschitz('svds')
        self._skip_if_disabled()
        L_svds = op.lipschitz(recompute=True, algo="svds")
        L_fro = op.lipschitz(recompute=True, algo="fro", xp=xp, enable_warnings=False)
        assert less_equal(
            L_svds,
            L_fro,
            as_dtype=width.value,
            # as_dtype=pycrt.Width.SINGLE.value,
        ).item()

    def test_math3_lipschitz(self, op, _op_svd, width):
        # op.lipschitz('svds') computes the optimal Lipschitz constant.
        self._skip_if_disabled()
        L_svds = op.lipschitz(recompute=True, algo="svds")
        # Comparison is done with user-specified metric since operator may compute `L_svds` via an
        # approximate `.apply()' method.
        cast = lambda x: np.array([x], dtype=width.value)
        assert self._metric(cast(L_svds), cast(_op_svd.max()), as_dtype=width.value)

    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        assert J is op

    @pytest.mark.parametrize("k", [1, 2])
    @coupled_gpu_which
    def test_value1D_svdvals(self, op, xp, _gpu, _op_svd, k, which):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        data = dict(k=k, which=which, gpu=_gpu)
        self._check_value1D_vals(op.svdvals, data, _op_svd)

    def test_backend_svdvals(self, op, xp, _gpu):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        self._check_backend_vals(op.svdvals, _gpu)

    def test_precCM_svdvals(self, op, xp, _gpu, width):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        data = dict(in_=dict(k=1, gpu=_gpu))
        self._check_precCM(op.svdvals, data, (width,))

    def test_interface_T(self, op, _op_T):
        self._skip_if_disabled()
        if op.dim == 1:
            klass = LinFuncT
        else:
            klass = LinOpT
        self._check_has_interface(_op_T, klass)

    def test_interface_TT(self, op, _op_TT):
        # Transposing twice returns operator with initial shape.
        self._skip_if_disabled()
        assert _op_TT.shape == op.shape

    def test_value1D_call_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_value1D(_op_T.__call__, _data_adjoint)

    def test_valueND_call_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_valueND(_op_T.__call__, _data_adjoint)

    def test_backend_call_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_backend(_op_T.__call__, _data_adjoint)

    def test_prec_call_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_prec(_op_T.__call__, _data_adjoint)

    def test_precCM_call_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_precCM(_op_T.__call__, _data_adjoint)

    def test_value1D_apply_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_value1D(_op_T.apply, _data_adjoint)

    def test_valueND_apply_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_valueND(_op_T.apply, _data_adjoint)

    def test_backend_apply_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_backend(_op_T.apply, _data_adjoint)

    def test_prec_apply_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_prec(_op_T.apply, _data_adjoint)

    def test_precCM_apply_T(self, _op_T, _data_adjoint):
        self._skip_if_disabled()
        self._check_precCM(_op_T.apply, _data_adjoint)

    def test_value1D_adjoint_T(self, _op_T, _data_apply):
        self._skip_if_disabled()
        self._check_value1D(_op_T.adjoint, _data_apply)

    def test_valueND_adjoint_T(self, _op_T, _data_apply):
        self._skip_if_disabled()
        self._check_valueND(_op_T.adjoint, _data_apply)

    def test_backend_adjoint_T(self, _op_T, _data_apply):
        self._skip_if_disabled()
        self._check_backend(_op_T.adjoint, _data_apply)

    def test_prec_adjoint_T(self, _op_T, _data_apply):
        self._skip_if_disabled()
        self._check_prec(_op_T.adjoint, _data_apply)

    def test_precCM_adjoint_T(self, _op_T, _data_apply):
        self._skip_if_disabled()
        self._check_precCM(_op_T.adjoint, _data_apply)

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

    def test_value1D_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.adjoint, _data_adjoint_dagger, dtype=np.dtype(np.half))

    def test_valueND_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        # To avoid CG convergence issues, correctness is assesed at lowest precision only.
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.adjoint, _data_adjoint_dagger, dtype=np.dtype(np.half))

    def test_backend_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_backend(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_prec_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_prec(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_precCM_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_precCM(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_interface_gram(self, op):
        self._skip_if_disabled()
        if op.dim > 1:
            klass = SelfAdjointOpT
        else:
            klass = LinFuncT
        self._check_has_interface(op.gram(), klass)

    def test_math_gram(self, op, xp, width):
        # op_g.apply == op_g.adjoint == adjoint \comp apply
        self._skip_if_disabled()
        op_g = op.gram()
        x = self._random_array((30, op.dim), xp=xp, width=width)

        with pycrt.EnforcePrecision(False):
            assert self._metric(op_g.apply(x), op_g.adjoint(x), as_dtype=width.value)
            assert self._metric(op_g.apply(x), op.adjoint(op.apply(x)), as_dtype=width.value)

    def test_interface_cogram(self, op):
        self._skip_if_disabled()
        if op.codim > 1:
            klass = SelfAdjointOpT
        else:
            klass = LinFuncT
        self._check_has_interface(op.cogram(), klass)

    def test_math_cogram(self, op, xp, width):
        # op_cg.apply == op_cg.adjoint == apply \comp adjoint
        self._skip_if_disabled()
        op_cg = op.cogram()
        x = self._random_array((30, op.codim), xp=xp, width=width)

        with pycrt.EnforcePrecision(False):
            assert self._metric(op_cg.apply(x), op_cg.adjoint(x), as_dtype=width.value)
            assert self._metric(op_cg.apply(x), op.apply(op.adjoint(x)), as_dtype=width.value)

    def test_value_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pycu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert A.shape == _op_array.shape
        assert self._metric(_op_array.T, A.T, as_dtype=dtype)

    def test_backend_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pycu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert pycu.get_array_module(A) == xp

    def test_prec_asarray(self, op, _op_array):
        self._skip_if_disabled()
        xp = pycu.get_array_module(_op_array)
        dtype = _op_array.dtype

        A = op.asarray(xp=xp, dtype=dtype)
        assert A.dtype == dtype

    def test_value_to_sciop(self, _op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(_op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert out.shape == out_gt.shape
        assert self._metric(out, out_gt, as_dtype=out_gt.dtype)

    def test_backend_to_sciop(self, _op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(_op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert pycu.get_array_module(out) == pycu.get_array_module(out_gt)

    def test_prec_to_sciop(self, _op_sciop, _data_to_sciop):
        self._skip_if_disabled()
        func = getattr(_op_sciop, _data_to_sciop["mode"])
        out = func(**_data_to_sciop["in_"])
        out_gt = _data_to_sciop["out"]
        assert out.dtype == out_gt.dtype

    def test_interface_from_sciop(self, _op_sciop):
        self._skip_if_disabled()
        op = self.base.from_sciop(_op_sciop)
        self._check_has_interface(op, self.__class__)

    def test_value_from_sciop(self, _op_sciop, _data_from_sciop):
        self._skip_if_disabled()
        op = self.base.from_sciop(_op_sciop)
        func = getattr(op, _data_from_sciop["mode"])
        out = func(**_data_from_sciop["in_"])
        out_gt = _data_from_sciop["out"]
        assert out.shape == out_gt.shape
        assert self._metric(out, out_gt, as_dtype=out_gt.dtype)

    def test_backend_from_sciop(self, _op_sciop, _data_from_sciop):
        self._skip_if_disabled()
        op = self.base.from_sciop(_op_sciop)
        func = getattr(op, _data_from_sciop["mode"])
        out = func(**_data_from_sciop["in_"])
        out_gt = _data_from_sciop["out"]
        assert pycu.get_array_module(out) == pycu.get_array_module(out_gt)

    def test_prec_from_sciop(self, _op_sciop, _data_from_sciop):
        self._skip_if_disabled()
        op = self.base.from_sciop(_op_sciop)
        func = getattr(op, _data_from_sciop["mode"])
        out = func(**_data_from_sciop["in_"])
        out_gt = _data_from_sciop["out"]
        assert out.dtype == out_gt.dtype


class LinFuncT(ProxDiffFuncT, LinOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.LinFunc
    interface = frozenset(ProxDiffFuncT.interface | LinOpT.interface)
    disable_test = frozenset(ProxDiffFuncT.disable_test | LinOpT.disable_test)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def data_grad(self, op) -> DataLike:
        # We know for linfuncs that op.grad(x) = op.asarray()
        x = self._random_array((op.dim,))
        y = op.asarray()
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
        tau = np.abs(self._random_array((1,)))[0]
        y = x - tau * g
        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )

    # Tests -------------------------------------------------------------------
    @pytest.mark.parametrize("k", [1])
    @pytest.mark.parametrize("which", ["SM", "LM"])
    def test_value1D_svdvals(self, op, xp, _gpu, _op_svd, k, which):
        self._skip_if_disabled()
        super().test_value1D_svdvals(op, xp, _gpu, _op_svd, k, which)

    @pytest.mark.skip("Notion of loss undefined for LinFuncs.")
    def test_interface_asloss(self, op, xp, width):
        super().test_interface_asloss(op, xp, width)


class SquareOpT(LinOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.SquareOp
    interface = frozenset(LinOpT.interface | {"trace"})

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def _op_trace(self, op) -> float:
        # True trace
        tr = op.asarray().trace()
        return float(tr)

    # Tests -------------------------------------------------------------------
    def test_square(self, op):
        self._skip_if_disabled()
        assert op.dim == op.codim

    def test_interface_trace(self, op):
        assert isinstance(op.trace(), float)

    def test_value_trace(self, op, _op_trace):
        # Ensure computed trace (w/ default parameter values) satisfies statistical property stated
        # in hutchpp() docstring, i.e.: estimation error smaller than 1e-2 w/ probability 0.9
        N_trial = 100
        tr = np.array([op.trace() for _ in range(N_trial)])
        N_pass = sum(np.abs(tr - _op_trace) <= 1e-2)
        assert N_pass >= 0.9 * N_trial


class NormalOpT(SquareOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.NormalOp
    interface = frozenset(SquareOpT.interface | {"eigvals"})

    # Internal Helpers --------------------------------------------------------
    eigvals_unsupported_on_gpu = pytest.mark.parametrize(
        "_gpu",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    True,
                    reason="eigvals unsupported by CuPy.",
                    strict=True,
                ),
            ),
        ],
    )

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def _op_eig(self, op) -> np.ndarray:
        # compute all eigenvalues, sorted in ascending magnitude order.
        D = np.linalg.eigvals(op.asarray())
        D = D[np.argsort(np.abs(D))]
        return D

    # Tests -------------------------------------------------------------------
    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("which", ["SM", "LM"])
    @eigvals_unsupported_on_gpu
    def test_value1D_eigvals(self, op, xp, _gpu, _op_eig, k, which):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        data = dict(k=k, which=which, gpu=_gpu)
        self._check_value1D_vals(op.eigvals, data, _op_eig)

    @eigvals_unsupported_on_gpu
    def test_backend_eigvals(self, op, xp, _gpu):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        self._check_backend_vals(op.eigvals, _gpu)

    @eigvals_unsupported_on_gpu
    def test_precCM_eigvals(self, op, xp, _gpu, width):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        data = dict(in_=dict(k=1, gpu=_gpu))
        self._check_precCM(op.eigvals, data, (width.complex,))  # <<===
        # We use the complex-valued types since .eigvals() should return complex.
        # (Exception: SelfAdjointOp)

    def test_math_normality(self, op, xp, width):
        # AA^{*} = A^{*}A
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, op.dim), xp=xp, width=width)

        lhs = op.apply(op.adjoint(x))
        rhs = op.adjoint(op.apply(x))
        assert allclose(lhs, rhs, as_dtype=width.value)


class UnitOpT(NormalOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.UnitOp

    # Internal helpers --------------------------------------------------------
    @classmethod
    def _check_identity(cls, operator, xp, width):
        x = cls._random_array((30, operator.dim), xp=xp, width=width)
        assert allclose(operator.apply(x), x, as_dtype=width.value)
        assert allclose(operator.adjoint(x), x, as_dtype=width.value)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def _op_svd(self, op) -> np.ndarray:
        D = np.ones(op.dim)
        return D

    # Tests -------------------------------------------------------------------
    def test_math_eig(self, _op_eig):
        # |\lambda| == 1
        assert np.allclose(np.abs(_op_eig), 1)

    def test_math_gram(self, op, xp, width):
        # op_g == I
        self._skip_if_disabled()
        self._check_identity(op.gram(), xp, width)

    def test_math_cogram(self, op, xp, width):
        # op_cg == I
        self._skip_if_disabled()
        self._check_identity(op.cogram(), xp, width)

    def test_math_norm(self, op, xp, width):
        # \norm{U x} = \norm{U^{*} x} = \norm{x}
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, op.dim), xp=xp, width=width)

        lhs1 = pylinalg.norm(op.apply(x), axis=-1)
        lhs2 = pylinalg.norm(op.adjoint(x), axis=-1)
        rhs = pylinalg.norm(x, axis=-1)

        assert allclose(lhs1, lhs2, as_dtype=width.value)
        assert allclose(lhs1, rhs, as_dtype=width.value)

    # local override of this fixture: no GPU limitations
    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("which", ["SM", "LM"])
    def test_value1D_svdvals(self, op, xp, _gpu, _op_svd, k, which):
        self._skip_if_disabled()
        super().test_value1D_svdvals(op, xp, _gpu, _op_svd, k, which)


class SelfAdjointOpT(NormalOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.SelfAdjointOp

    # Tests -------------------------------------------------------------------
    def test_math_eig(self, _op_eig):
        self._skip_if_disabled()
        assert pycuc._is_real(_op_eig)

    # local override of this fixture: back to standard _gpu rules
    @pytest.mark.parametrize("k", [1, 2])
    @LinOpT.coupled_gpu_which
    def test_value1D_eigvals(self, op, xp, _gpu, _op_eig, k, which):
        self._skip_if_disabled()
        super().test_value1D_eigvals(op, xp, _gpu, _op_eig, k, which)

    # local override of this fixture: back to standard _gpu rules
    def test_backend_eigvals(self, op, xp, _gpu):
        self._skip_if_disabled()
        super().test_backend_eigvals(op, xp, _gpu)

    def test_precCM_eigvals(self, op, xp, _gpu, width):
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(xp, _gpu)
        data = dict(in_=dict(k=1, gpu=_gpu))
        self._check_precCM(op.eigvals, data, (width,))  # <<===
        # We revert back to real-valued types since .eigvals() should return real.

    def test_math_selfadjoint(self, op, xp, width):
        # A = A^{*}
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, op.dim), xp=xp, width=width)

        lhs = op.apply(x)
        rhs = op.adjoint(x)
        assert allclose(lhs, rhs, as_dtype=width.value)


class PosDefOpT(SelfAdjointOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.PosDefOp

    # Tests -------------------------------------------------------------------
    def test_math_eig(self, _op_eig):
        self._skip_if_disabled()
        assert pycuc._is_real(_op_eig)
        assert np.all(_op_eig > 0)

    def test_math_posdef(self, op, xp, width):
        # <Ax,x> > 0
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, op.dim), xp=xp, width=width)

        ip = lambda a, b: (a * b).sum(axis=-1)  # (N, Q) * (N, Q) -> (N,)
        assert np.all(less_equal(0, ip(op.apply(x), x), as_dtype=width.value))


class ProjOpT(SquareOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.ProjOp

    # Fixtures ----------------------------------------------------------------
    def test_math_idempotent(self, op, xp, width):
        self._skip_if_disabled()
        N = 30
        x = self._random_array((N, op.dim), xp=xp, width=width)
        y = op.apply(x)
        z = op.apply(y)

        assert allclose(y, z, as_dtype=width.value)


class OrthProjOpT(ProjOpT, SelfAdjointOpT):
    # Class Properties --------------------------------------------------------
    base = pyca.OrthProjOp
    interface = frozenset(ProjOpT.interface | SelfAdjointOpT.interface)
    disable_test = frozenset(ProjOpT.disable_test | SelfAdjointOpT.disable_test)
