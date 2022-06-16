import collections.abc as cabc
import copy
import inspect
import itertools
import math
import types
import typing as typ

import numpy as np
import numpy.random as npr
import pytest
import scipy.linalg as splinalg

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def isclose(a: np.ndarray, b: np.ndarray, as_dtype: np.dtype) -> np.ndarray:
    """
    Equivalent of `np.isclose`, but where atol is automatically chosen based on `as_dtype`.
    """
    atol = {
        pycrt.Width.HALF.value: 3e-2,
        pycrt.Width.SINGLE.value: 2e-4,
        pycrt.Width.DOUBLE.value: 1e-8,
        pycrt.Width.QUAD.value: 1e-16,
    }
    # Numbers obtained by:
    # * \sum_{k >= (p+1)//2} 2^{-k}, where p=<number of mantissa bits>; then
    # * round up value to 3 significant decimal digits.
    # N_mantissa = [10, 23, 52, 112] for [half, single, double, quad] respectively.

    if (prec := atol.get(as_dtype)) is None:
        # should occur for integer types only
        prec = atol[pycrt.Width.QUAD.value]
    cast = lambda x: x.astype(as_dtype)
    eq = np.isclose(cast(a), cast(b), atol=prec)
    return eq


def allclose(a: np.ndarray, b: np.ndarray, as_dtype: np.dtype) -> bool:
    """
    Equivalent of `all.isclose`, but where atol is automatically chosen based on `as_dtype`.
    """
    return np.all(isclose(a, b, as_dtype))


def less_equal(a: np.ndarray, b: np.ndarray, as_dtype: np.dtype) -> np.ndarray:
    """
    Equivalent of `a <= b`, but where equality tests are done at a chosen numerical precision.
    """
    x = a <= b
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
# * test_[value,backend,prec,precCM]_<method>(op, ...)
#       Verify that <method>, returns
#       * value: right output values
#       * backend: right output type
#       * prec: input/output have same precision
#       * precCM: output respects context-manager choice
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
    disable_test: cabc.Set[str] = frozenset()
    interface: cabc.Set[str] = frozenset(
        {
            "shape",
            "dim",
            "codim",
            "apply",
            "__call__",
            "lipschitz",
            "squeeze",
            "specialize",
        }
    )

    # Internal helpers --------------------------------------------------------
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
    def _check_has_interface(op: pyco.Map, klass: "MapT"):
        # Verify `op` has the public interface of `klass`.
        assert klass.interface <= frozenset(dir(op))

    @staticmethod
    def _check_value1D(func, data):
        out_gt = data["out"]

        in_ = data["in_"]
        out = pycu.compute(func(**in_))

        assert out.ndim == in_["arr"].ndim
        assert allclose(out, out_gt, as_dtype=in_["arr"].dtype)

    @staticmethod
    def _check_valueND(func, data):
        sh_extra = (2, 1)  # prepend input/output shape by this amount.

        out_gt = data["out"]
        out_gt = np.broadcast_to(out_gt, (*sh_extra, *out_gt.shape))

        in_ = data["in_"]
        arr = in_["arr"]
        xp = pycu.get_array_module(arr)
        arr = xp.broadcast_to(arr, (*sh_extra, *arr.shape))
        in_.update(arr=arr)
        out = pycu.compute(func(**in_))

        assert out.ndim == in_["arr"].ndim
        assert allclose(out, out_gt, as_dtype=in_["arr"].dtype)

    @staticmethod
    def _check_backend(func, data):
        in_ = data["in_"]
        out = func(**in_)

        assert type(out) == type(in_["arr"])

    @staticmethod
    def _check_prec(func, data):
        in_ = data["in_"]
        with pycrt.EnforcePrecision(False):
            out = func(**in_)
            assert out.dtype == in_["arr"].dtype

    @staticmethod
    def _check_precCM(
        func,
        data,
        widths: cabc.Collection[pycrt.Width] = frozenset(pycrt.Width),
    ):
        in_ = data["in_"]
        stats = []
        for width in widths:
            with pycrt.Precision(width):
                out = func(**in_)
            stats.append(out.dtype == width.value)

        assert all(stats)

    @staticmethod
    def _random_array(shape: tuple[int], seed: int = 0):
        rng = npr.default_rng(seed)
        x = rng.normal(size=shape)
        return x

    @staticmethod
    def _sanitize(x, default):
        if x is not None:
            return x
        else:
            return default

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def op(self) -> pyco.Map:
        # override in subclass to instantiate the object to test.
        raise NotImplementedError

    @pytest.fixture(params=pycd.supported_array_modules())
    def xp(self, request) -> types.ModuleType:
        # override in subclass if numeric methods are to be tested on a subset of array backends.
        return request.param

    @pytest.fixture(params=list(pycrt.Width))
    def width(self, request) -> pycrt.Width:
        # override in subclass if numeric methods are to be tested on a subset of precisions.
        return request.param

    @pytest.fixture(
        params=[
            pyco.Map,
            pyco.DiffMap,
            pyco.Func,
            pyco.DiffFunc,
            pyco.ProxFunc,
            pyco.ProxDiffFunc,
            pyco.LinOp,
            pyco.LinFunc,
            pyco.SquareOp,
            pyco.NormalOp,
            pyco.SelfAdjointOp,
            pyco.PosDefOp,
            pyco.UnitOp,
            pyco.ProjOp,
            pyco.OrthProjOp,
        ]
    )
    def _klass(self, request) -> pyco.Map:
        # Returns some operator in the pyco.Map hierarchy.
        # Do not override in subclass: for internal use only to test `op.specialize()`.
        return request.param

    @pytest.fixture
    def data_shape(self) -> pyct.Shape:
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
        # Arrays should be NumPy-only.
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

    @pytest.fixture
    def _data_apply_argshift(self, _data_apply) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argshift().apply()`.
        in_ = copy.deepcopy(_data_apply["in_"])

        xp = pycu.get_array_module(in_["arr"])
        shift = self._random_array((in_["arr"].size,))
        shift = xp.array(shift, dtype=in_["arr"].dtype)
        in_.update(arr=in_["arr"] + shift)

        data = dict(
            in_=in_,
            out=_data_apply["out"],
            shift=shift,  # for _op_argshift()
        )
        return data

    @pytest.fixture
    def _op_argshift(self, op, _data_apply_argshift) -> pyco.Map:
        shift = _data_apply_argshift["shift"]
        return op.argshift(-shift)

    @pytest.fixture
    def _data_apply_argscale(self, _data_apply) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argscale().apply()`.
        in_ = copy.deepcopy(_data_apply["in_"])

        scale = self._random_array((1,)).item()
        in_["arr"] *= scale

        data = dict(
            in_=in_,
            out=_data_apply["out"],
            scale=scale,  # for _op_argscale()
        )
        return data

    @pytest.fixture
    def _op_argscale(self, op, _data_apply_argscale) -> pyco.Map:
        scale = _data_apply_argscale["scale"]
        return op.argscale(1 / scale)

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

    def test_lipschitz(self, op):
        # Ensure:
        # * _lipschitz matches .lipschitz() after being called once;
        self._skip_if_disabled()
        L_computed = op.lipschitz()
        L_memoized = op._lipschitz
        assert np.isclose(L_computed, L_memoized)

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

    def test_interface_argshift(self, op):
        # Must be of same class (subclass if needed)
        self._skip_if_disabled()
        N_dim = self._sanitize(op.dim, 3)
        shift = self._random_array((N_dim,))
        op_s = op.argshift(shift)
        self._check_has_interface(op_s, self.__class__)

    def test_value1D_apply_argshift(self, _op_argshift, _data_apply_argshift):
        self._skip_if_disabled()
        self._check_value1D(_op_argshift.apply, _data_apply_argshift)

    def test_valueND_apply_argshift(self, _op_argshift, _data_apply_argshift):
        self._skip_if_disabled()
        self._check_valueND(_op_argshift.apply, _data_apply_argshift)

    def test_backend_apply_argshift(self, _op_argshift, _data_apply_argshift):
        self._skip_if_disabled()
        self._check_backend(_op_argshift.apply, _data_apply_argshift)

    def test_prec_apply_argshift(self, _op_argshift, _data_apply_argshift):
        self._skip_if_disabled()
        self._check_prec(_op_argshift.apply, _data_apply_argshift)

    def test_precCM_apply_argshift(self, _op_argshift, _data_apply_argshift):
        self._skip_if_disabled()
        self._check_precCM(_op_argshift.apply, _data_apply_argshift)

    def test_interface_argscale(self, op):
        # Must be of same class
        self._skip_if_disabled()
        scale = self._random_array((1,)).item()
        op_s = op.argscale(scale)
        self._check_has_interface(op_s, self.__class__)

    def test_value1D_apply_argscale(self, _op_argscale, _data_apply_argscale):
        self._skip_if_disabled()
        self._check_value1D(_op_argscale.apply, _data_apply_argscale)

    def test_valueND_apply_argscale(self, _op_argscale, _data_apply_argscale):
        self._skip_if_disabled()
        self._check_valueND(_op_argscale.apply, _data_apply_argscale)

    def test_backend_apply_argscale(self, _op_argscale, _data_apply_argscale):
        self._skip_if_disabled()
        self._check_backend(_op_argscale.apply, _data_apply_argscale)

    def test_prec_apply_argscale(self, _op_argscale, _data_apply_argscale):
        self._skip_if_disabled()
        self._check_prec(_op_argscale.apply, _data_apply_argscale)

    def test_precCM_apply_argscale(self, _op_argscale, _data_apply_argscale):
        self._skip_if_disabled()
        self._check_precCM(_op_argscale.apply, _data_apply_argscale)

    def test_math_lipschitz(self, op, data_math_lipschitz):
        # \norm{f(x) - f(y)}{2} \le L * \norm{x - y}{2}
        self._skip_if_disabled()
        L = op.lipschitz()

        stats = []
        for x, y in itertools.combinations(data_math_lipschitz, 2):
            lhs = np.linalg.norm(op.apply(x) - op.apply(y))
            rhs = L * np.linalg.norm(x - y)
            stats.append(less_equal(lhs, rhs, as_dtype=data_math_lipschitz.dtype))

        assert all(stats)

    def test_squeeze(self, op):
        # op.squeeze() sub-classes to Func for scalar outputs, and is transparent otherwise.
        self._skip_if_disabled()
        if op.codim == 1:
            self._check_has_interface(op.squeeze(), FuncT)
        else:
            assert op.squeeze() is op

    @pytest.mark.skip(reason="Requires some scaffolding first.")
    def test_specialize(self, op, _klass):
        self._skip_if_disabled()
        # def map_cmp(a, b):
        #     if a above b:
        #         return -1
        #     elif a == b:
        #         return 0
        #     else:
        #         return 1

        #     test_specialize: needed fixture: op, klass
        #         for every class lower in the hierarchy:
        #             verify op.specialize(klass) has correct class interface
        #         assert op.specialize(op.__class__) is op
        #         for every class upper in the hierarchy:
        #             verify op.specialize() fails
        pass


class FuncT(MapT):
    # Class Properties --------------------------------------------------------
    interface = frozenset(MapT.interface | {"asloss"})

    # Tests -------------------------------------------------------------------
    def test_codim(self, op, data_shape):
        self._skip_if_disabled()
        assert op.codim == 1

    def test_squeeze(self, op):
        self._skip_if_disabled()
        assert op.squeeze() is op

    def test_interface_asloss(self, op):
        # op.asloss() sub-classes Func if data provided, transparent otherwise.
        # Disable this test if asloss() not defined.
        self._skip_if_disabled()
        assert op.asloss() is op

        N_dim = self._sanitize(op.dim, default=3)
        data = self._random_array((N_dim,))
        self._check_has_interface(op.asloss(data), self.__class__)


class DiffMapT(MapT):
    # Class Properties --------------------------------------------------------
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
        # Arrays should be NumPy-only.
        raise NotImplementedError

    # Tests -------------------------------------------------------------------
    def test_diff_lipschitz(self, op):
        # Ensure:
        # * _diff_lipschitz matches .diff_lipschitz() after being called once.
        self._skip_if_disabled()
        dL_computed = op.diff_lipschitz()
        dL_memoized = op._diff_lipschitz
        assert np.isclose(dL_computed, dL_memoized)

    def test_squeeze(self, op):
        # op.squeeze() sub-classes to DiffFunc for scalar outputs, and is transparent otherwise.
        self._skip_if_disabled()
        if op.codim == 1:
            self._check_has_interface(op.squeeze(), DiffFuncT)
        else:
            assert op.squeeze() is op

    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        self._check_has_interface(J, LinOpT)

    def test_math_diff_lipschitz(self, op, data_math_diff_lipschitz):
        # \norm{J(x) - J(y)}{F} \le diff_L * \norm{x - y}{2}
        self._skip_if_disabled()
        dL = op.diff_lipschitz()
        J = lambda _: op.jacobian(_).asarray().flatten()
        # .flatten() used to consistently compare jacobians via the L2 norm.
        # (Allows one to re-use this test for scalar-valued DiffMaps.)

        stats = []
        for x, y in itertools.combinations(data_math_diff_lipschitz, 2):
            lhs = np.linalg.norm(J(x) - J(y))
            rhs = dL * np.linalg.norm(x - y)
            stats.append(less_equal(lhs, rhs, as_dtype=data_math_diff_lipschitz.dtype))

        assert all(stats)


class DiffFuncT(FuncT, DiffMapT):
    # Class Properties --------------------------------------------------------
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

    @pytest.fixture
    def _data_grad_argshift(self, _data_grad) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argshift().grad()`.
        in_ = copy.deepcopy(_data_grad["in_"])

        xp = pycu.get_array_module(in_["arr"])
        shift = self._random_array((in_["arr"].size,))
        shift = xp.array(shift, dtype=in_["arr"].dtype)
        in_.update(arr=in_["arr"] + shift)

        data = dict(
            in_=in_,
            out=_data_grad["out"],
            shift=shift,  # for _op_argshift()
        )
        return data

    @pytest.fixture
    def _data_grad_argscale(self, _data_grad) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argscale().grad()`.
        in_ = copy.deepcopy(_data_grad["in_"])
        out = copy.deepcopy(_data_grad["out"])

        scale = self._random_array((1,)).item()
        in_["arr"] *= scale
        out = out / scale  # potential dtype change doesn't matter: see _data_grad()

        data = dict(
            in_=in_,
            out=out,
            scale=scale,  # for _op_argscale()
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

    def test_value1D_grad_argshift(self, _op_argshift, _data_grad_argshift):
        self._skip_if_disabled()
        self._check_value1D(_op_argshift.grad, _data_grad_argshift)

    def test_valueND_grad_argshift(self, _op_argshift, _data_grad_argshift):
        self._skip_if_disabled()
        self._check_valueND(_op_argshift.grad, _data_grad_argshift)

    def test_backend_grad_argshift(self, _op_argshift, _data_grad_argshift):
        self._skip_if_disabled()
        self._check_backend(_op_argshift.grad, _data_grad_argshift)

    def test_prec_grad_argshift(self, _op_argshift, _data_grad_argshift):
        self._skip_if_disabled()
        self._check_prec(_op_argshift.grad, _data_grad_argshift)

    def test_precCM_grad_argshift(self, _op_argshift, _data_grad_argshift):
        self._skip_if_disabled()
        self._check_precCM(_op_argshift.grad, _data_grad_argshift)

    def test_value1D_grad_argscale(self, _op_argscale, _data_grad_argscale):
        self._skip_if_disabled()
        self._check_value1D(_op_argscale.grad, _data_grad_argscale)

    def test_valueND_grad_argscale(self, _op_argscale, _data_grad_argscale):
        self._skip_if_disabled()
        self._check_valueND(_op_argscale.grad, _data_grad_argscale)

    def test_backend_grad_argscale(self, _op_argscale, _data_grad_argscale):
        self._skip_if_disabled()
        self._check_backend(_op_argscale.grad, _data_grad_argscale)

    def test_prec_grad_argscale(self, _op_argscale, _data_grad_argscale):
        self._skip_if_disabled()
        self._check_prec(_op_argscale.grad, _data_grad_argscale)

    def test_precCM_grad_argscale(self, _op_argscale, _data_grad_argscale):
        self._skip_if_disabled()
        self._check_precCM(_op_argscale.grad, _data_grad_argscale)

    def test_math1_grad(self, op, data_grad):
        # .jacobian/.grad outputs are consistent.
        self._skip_if_disabled()
        arr = data_grad["in_"]["arr"]
        J = op.jacobian(arr).asarray()
        g = op.grad(arr)

        assert J.size == g.size
        assert allclose(J.squeeze(), g, as_dtype=arr.dtype)

    def test_math2_grad(self, op):
        # f(x - \frac{1}{L} \grad_{f}(x)) <= f(x)
        self._skip_if_disabled()
        L = op.lipschitz()

        N_test, N_dim = 5, self._sanitize(op.dim, default=3)
        rhs = self._random_array((N_test, N_dim))
        lhs = rhs - op.grad(rhs) / L

        assert np.all(less_equal(op.apply(lhs), op.apply(rhs), as_dtype=lhs.dtype))


class ProxFuncT(FuncT):
    # Class Properties --------------------------------------------------------
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
    def _op_m(self, op) -> tuple[float, pyco.DiffFunc]:
        mu = 1.1
        return mu, op.moreau_envelope(mu)

    @pytest.fixture
    def data_prox(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.prox().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

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
    def _data_prox_argshift(self, _data_prox) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argshift().prox()`.
        in_ = copy.deepcopy(_data_prox["in_"])

        xp = pycu.get_array_module(in_["arr"])
        shift = self._random_array((in_["arr"].size,))
        shift = xp.array(shift, dtype=in_["arr"].dtype)
        in_.update(arr=in_["arr"] + shift)
        out = pycu.compute(_data_prox["out"] + shift)

        data = dict(
            in_=in_,
            out=out,
            shift=shift,  # for _op_argshift()
        )
        return data

    @pytest.fixture
    def _data_prox_argscale(self, _data_prox) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argscale().prox()`.
        in_ = copy.deepcopy(_data_prox["in_"])
        out = copy.deepcopy(_data_prox["out"])

        scale = self._random_array((1,)).item()
        in_["arr"] *= scale
        in_["tau"] *= scale**2
        out = out * scale

        data = dict(
            in_=in_,
            out=out,
            scale=scale,  # for _op_argscale()
        )
        return data

    @pytest.fixture
    def _data_fenchel_prox(self, _data_prox) -> DataLike:
        # Generate fenchel_prox values from prox ground-truth. (All precision/backends.)
        # Do not override in subclass: for internal use only to test `op.fenchel_prox()`.
        p_arr = _data_prox["in_"]["arr"]
        p_tau = _data_prox["in_"]["tau"]
        p_out = _data_prox["out"]
        data = dict(
            in_=dict(
                arr=p_arr / p_tau,
                sigma=1 / p_tau,
            ),
            out=(p_arr - p_out) / p_tau,
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

    def test_value1D_prox_argshift(self, _op_argshift, _data_prox_argshift):
        self._skip_if_disabled()
        self._check_value1D(_op_argshift.prox, _data_prox_argshift)

    def test_valueND_prox_argshift(self, _op_argshift, _data_prox_argshift):
        self._skip_if_disabled()
        self._check_valueND(_op_argshift.prox, _data_prox_argshift)

    def test_backend_prox_argshift(self, _op_argshift, _data_prox_argshift):
        self._skip_if_disabled()
        self._check_backend(_op_argshift.prox, _data_prox_argshift)

    def test_prec_prox_argshift(self, _op_argshift, _data_prox_argshift):
        self._skip_if_disabled()
        self._check_prec(_op_argshift.prox, _data_prox_argshift)

    def test_precCM_prox_argshift(self, _op_argshift, _data_prox_argshift):
        self._skip_if_disabled()
        self._check_precCM(_op_argshift.prox, _data_prox_argshift)

    def test_value1D_prox_argscale(self, _op_argscale, _data_prox_argscale):
        self._skip_if_disabled()
        self._check_value1D(_op_argscale.prox, _data_prox_argscale)

    def test_valueND_prox_argscale(self, _op_argscale, _data_prox_argscale):
        self._skip_if_disabled()
        self._check_valueND(_op_argscale.prox, _data_prox_argscale)

    def test_backend_prox_argscale(self, _op_argscale, _data_prox_argscale):
        self._skip_if_disabled()
        self._check_backend(_op_argscale.prox, _data_prox_argscale)

    def test_prec_prox_argscale(self, _op_argscale, _data_prox_argscale):
        self._skip_if_disabled()
        self._check_prec(_op_argscale.prox, _data_prox_argscale)

    def test_precCM_prox_argscale(self, _op_argscale, _data_prox_argscale):
        self._skip_if_disabled()
        self._check_precCM(_op_argscale.prox, _data_prox_argscale)

    def test_math_prox(self, op, data_prox):
        # Ensure y = prox_{tau f}(x) minimizes:
        # 2\tau f(z) - \norm{z - x}{2}^{2}, for any z \in \bR^{N}
        self._skip_if_disabled()
        in_ = data_prox["in_"]
        y = op.prox(**in_)

        N_test, N_dim = 5, y.shape[-1]
        x = self._random_array((N_test, N_dim)) + in_["arr"]

        def g(x):
            a = 2 * in_["tau"] * op.apply(x)
            b = np.linalg.norm(in_["arr"] - x, axis=-1, keepdims=True) ** 2
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

    def test_interface_moreau_envelope(self, _op_m):
        self._skip_if_disabled()
        _, op_m = _op_m
        self._check_has_interface(op_m, DiffFuncT)

    def test_math1_moreau_envelope(self, op, _op_m, data_apply):
        # op_m.apply() lower-bounds op.apply()
        self._skip_if_disabled()
        _, op_m = _op_m
        arr = data_apply["in_"]["arr"]
        lhs = op_m.apply(arr)
        rhs = op.apply(arr)

        assert less_equal(lhs, rhs, as_dtype=rhs.dtype)

    def test_math2_moreau_envelope(self, op, _op_m, data_apply):
        # op_m.grad(x) * mu = x - op.prox(x, mu)
        self._skip_if_disabled()
        mu, op_m = _op_m
        arr = data_apply["in_"]["arr"]
        lhs = op_m.grad(arr) * mu
        rhs = arr - op.prox(arr, mu)

        assert allclose(lhs, rhs, as_dtype=arr.dtype)


class ProxDiffFuncT(ProxFuncT, DiffFuncT):
    # Class Properties --------------------------------------------------------
    interface = frozenset(ProxFuncT.interface | DiffFuncT.interface)


class LinOpT(DiffMapT):
    # Class Properties --------------------------------------------------------
    interface = frozenset(
        DiffMapT.interface
        | {
            "__array__",
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
    def _data_adjoint_argscale(self, op, _data_apply_argscale) -> DataLike:
        # Do not override in subclass: for internal use only to test `op.argscale().adjoint()`.
        N_test = 30
        arr = self._random_array((N_test, op.codim))
        out = op.adjoint(arr)
        _data_adjoint = dict(in_=dict(arr=arr), out=out)

        in_ = copy.deepcopy(_data_adjoint["in_"])
        scale = _data_apply_argscale["scale"]
        out /= scale

        data = dict(
            in_=in_,
            out=out,
            scale=scale,  # for _op_argscale()
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

    @pytest.fixture(
        params=[
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not pycd.CUPY_ENABLED,
                    reason="GPU missing",
                ),
            ),
        ]
    )
    def _gpu(self, request) -> bool:
        # Do not override in subclass: for use only to test methods taking a `gpu` parameter.
        return request.param

    @pytest.fixture(params=[None, 1])
    def _damp(self, request) -> typ.Optional[float]:
        # candidate dampening factors for .pinv() & .dagger()
        return request.param

    @pytest.fixture
    def data_pinv(self, op, _damp, data_apply) -> DataLike:
        # override in subclass with 1D input/outputs of op.pinv().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        #
        # Default implementation: auto-computes pinv() at output points specified to test op.apply().
        A = op.gram().asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        if _damp is not None:
            for i in range(op.dim):
                A[i, i] += _damp

        arr = data_apply["out"]
        out = splinalg.solve(A, op.adjoint(arr), assume_a="pos")
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
    def _op_dagger(self, op, _damp) -> pyco.LinOp:
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
        A = op.gram().asarray(xp=np, dtype=pycrt.Width.DOUBLE.value)
        if _damp is not None:
            for i in range(op.dim):
                A[i, i] += _damp

        arr = data_apply["in_"]["arr"]
        out = op.apply(splinalg.solve(A, arr, assume_a="pos"))
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

    def test_math_adjoint(self, op):
        # <op.adjoint(x), y> = <x, op.apply(y)>
        self._skip_if_disabled()
        N = 20
        x = self._random_array((N, op.codim))
        y = self._random_array((N, op.dim))

        ip = lambda a, b: (a * b).sum(axis=-1)  # (N, Q) * (N, Q) -> (N,)
        lhs = ip(op.adjoint(x), y)
        rhs = ip(x, op.apply(y))

        assert allclose(lhs, rhs, lhs.dtype)

    def test_interface_argshift(self, op):
        self._skip_if_disabled()
        shift = self._random_array((op.dim,))
        op_s = op.argshift(shift)
        self._check_has_interface(op_s, DiffMapT)

    def test_value1D_adjoint_argscale(self, _op_argscale, _data_adjoint_argscale):
        self._skip_if_disabled()
        self._check_value1D(_op_argscale.adjoint, _data_adjoint_argscale)

    def test_valueND_adjoint_argscale(self, _op_argscale, _data_adjoint_argscale):
        self._skip_if_disabled()
        self._check_valueND(_op_argscale.adjoint, _data_adjoint_argscale)

    def test_backend_adjoint_argscale(self, _op_argscale, _data_adjoint_argscale):
        self._skip_if_disabled()
        self._check_backend(_op_argscale.adjoint, _data_adjoint_argscale)

    def test_prec_adjoint_argscale(self, _op_argscale, _data_adjoint_argscale):
        self._skip_if_disabled()
        self._check_prec(_op_argscale.adjoint, _data_adjoint_argscale)

    def test_precCM_adjoint_argscale(self, _op_argscale, _data_adjoint_argscale):
        self._skip_if_disabled()
        self._check_precCM(_op_argscale.adjoint, _data_adjoint_argscale)

    def test_math2_lipschitz(self, op):
        # op.lipschitz('fro') upper bounds op.lipschitz('svds')
        self._skip_if_disabled()
        L_svds = op.lipschitz(recompute=True, algo="svds")
        L_fro = op.lipschitz(recompute=True, algo="fro", enable_warnings=False)
        assert L_svds <= L_fro

    def test_math3_lipschitz(self, op, _op_svd):
        # op.lipschitz() computes the optimal Lipschitz constant.
        self._skip_if_disabled()
        assert np.isclose(op.lipschitz(), _op_svd.max())

    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        arr = _data_apply["in_"]["arr"]
        J = op.jacobian(arr)
        assert J is op

    def test_squeeze(self, op):
        # op.squeeze() sub-classes to LinFunc for scalar outputs, and is transparent otherwise.
        self._skip_if_disabled()
        if op.codim == 1:
            self._check_has_interface(op.squeeze(), LinFuncT)
        else:
            assert op.squeeze() is op

    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("which", ["SM", "LM"])
    def test_value1D_svdvals(self, op, _op_svd, k, which):
        self._skip_if_disabled()
        out = op.svdvals(k=k, which=which)
        assert out.size == k  # obtain N_svals asked for,
        assert allclose(np.sort(out), out, out.dtype)  # output is sorted,

        # and output is correct
        idx = np.argsort(np.abs(out))
        idx_gt = np.argsort(np.abs(_op_svd))
        if which == "SM":
            out = out[idx][:k]
            gt = _op_svd[idx_gt][:k]
        else:  # LM
            out = out[idx][-k:]
            gt = _op_svd[idx_gt][-k:]
        assert allclose(out, gt, out.dtype)

    def test_backend_svdvals(self, op, _gpu):
        self._skip_if_disabled()
        data = dict(k=1, gpu=_gpu)
        out = op.svdvals(**data)

        if _gpu:
            import cupy as xp_truth
        else:
            xp_truth = np
        assert pycu.get_array_module(out) == xp_truth

    @pytest.mark.parametrize(
        "width",  # local override of this fixture
        [
            pytest.param(
                pycrt.Width.HALF,
                marks=pytest.mark.xfail(reason="Unsupported by ARPACK/PROPACK/LOBPCG."),
            ),
            pycrt.Width.SINGLE,
            pycrt.Width.DOUBLE,
            pytest.param(
                pycrt.Width.QUAD,
                marks=pytest.mark.xfail(reason="Unsupported by ARPACK/PROPACK/LOBPCG."),
            ),
        ],
    )
    def test_precCM_svdvals(self, op, _gpu, width):
        self._skip_if_disabled()
        data = dict(in_=dict(k=1, gpu=_gpu))
        self._check_precCM(op.svdvals, data, (width,))

    def test_value1D_pinv(self, op, _data_pinv):
        self._skip_if_disabled()
        self._check_value1D(op.pinv, _data_pinv)

    def test_valueND_pinv(self, op, _data_pinv):
        self._skip_if_disabled()
        self._check_valueND(op.pinv, _data_pinv)

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
        self._check_has_interface(_op_dagger, LinOpT)

    def test_value1D_call_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.__call__, _data_apply_dagger)

    def test_valueND_call_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.__call__, _data_apply_dagger)

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
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.apply, _data_apply_dagger)

    def test_valueND_apply_dagger(self, _op_dagger, _data_apply_dagger):
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.apply, _data_apply_dagger)

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
        self._skip_if_disabled()
        self._check_value1D(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_valueND_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_valueND(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_backend_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_backend(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_prec_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_prec(_op_dagger.adjoint, _data_adjoint_dagger)

    def test_precCM_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        self._skip_if_disabled()
        self._check_precCM(_op_dagger.adjoint, _data_adjoint_dagger)


class LinFuncT(ProxDiffFuncT, LinOpT):
    # Class Properties --------------------------------------------------------
    interface = frozenset(ProxDiffFuncT.interface | LinOpT.interface)
    disable_test = frozenset(ProxDiffFuncT.disable_test | LinOpT.disable_test)

    # Fixtures ----------------------------------------------------------------

    # Tests -------------------------------------------------------------------
