import collections.abc as cabc
import copy
import inspect
import itertools
import types
import typing as typ

import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def func_name() -> str:
    """
    Returns
    -------
    up_fname: str
        Name of the function which called `func_name()`.

    Example
    -------
    >>> def f() -> str:
    ...     return func_name()
    ...
    ... f()  # -> 'f'
    """
    my_frame = inspect.currentframe()
    up_frame = inspect.getouterframes(my_frame)[1].frame
    up_finfo = inspect.getframeinfo(up_frame)
    up_fname = up_finfo.function
    return up_fname


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


def has_interface(op: pyco.Map, face: type) -> bool:
    """
    Parameters
    ----------
    op: pyco.Map
    face: MapT or subclasses

    Returns
    -------
    has: bool
        True if `op` has the public interface of `face`.
    """
    return face.interface <= frozenset(dir(op))


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
# * test_[value,backend,precision]_<method>(op, ...)
#       Verify that <method>, returns
#       * value: right output values
#       * backend: right output type
#       * precision: right output precision
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
            "lipschitz",
            "squeeze",
            "specialize",
        }
    )

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
    def data_lipschitz(self) -> DataLike:
        # override in subclass with the Lipschitz constant of op.
        # Don't return `op.lipschitz()`: hard-code what you are expecting.
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

    # Tests -------------------------------------------------------------------
    def test_interface(self, op):
        if func_name() not in self.disable_test:
            assert has_interface(op, self.__class__)

    def test_shape(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.shape == data_shape

    def test_dim(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.dim == data_shape[1]

    def test_codim(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.codim == data_shape[0]

    def test_lipschitz(self, op, data_lipschitz):
        if func_name() not in self.disable_test:
            in_ = op.lipschitz(**data_lipschitz["in_"])
            out = data_lipschitz["out"]
            assert np.isclose(in_, out)

    def test_value1D_apply(self, op, _data_apply):
        if func_name() not in self.disable_test:
            out_gt = _data_apply["out"]

            in_ = _data_apply["in_"]
            out = pycu.compute(op.apply(**in_))

            assert out.ndim == in_["arr"].ndim
            assert allclose(out, out_gt, as_dtype=in_["arr"].dtype)

    def test_valueND_apply(self, op, _data_apply):
        if func_name() not in self.disable_test:
            sh_extra = (2, 1)  # prepend input/output shape by this amount.

            out_gt = _data_apply["out"]
            out_gt = np.broadcast_to(out_gt, (*sh_extra, *out_gt.shape))

            in_ = _data_apply["in_"]
            arr = in_["arr"]
            xp = pycu.get_array_module(arr)
            arr = xp.broadcast_to(arr, (*sh_extra, *arr.shape))
            in_.update(arr=arr)
            out = pycu.compute(op.apply(**in_))

            assert out.ndim == in_["arr"].ndim
            assert allclose(out, out_gt, as_dtype=in_["arr"].dtype)

    def test_backend_apply(self, op, _data_apply):
        if func_name() not in self.disable_test:
            in_ = _data_apply["in_"]
            out = op.apply(**in_)

            assert type(out) == type(in_["arr"])

    def test_precision_apply(self, op, _data_apply):
        if func_name() not in self.disable_test:
            in_ = _data_apply["in_"]
            stats = []
            for width in pycrt.Width:
                with pycrt.Precision(width):
                    out = op.apply(**in_)
                stats.append(out.dtype == width.value)

            assert all(stats)

    def test_math_lipschitz(self, op, data_lipschitz, data_math_lipschitz):
        # \norm{f(x) - f(y)}{2} \le L * \norm{x - y}{2}
        if func_name() not in self.disable_test:
            L = op.lipschitz(**data_lipschitz["in_"])

            stats = []
            for x, y in itertools.combinations(data_math_lipschitz, 2):
                lhs = np.linalg.norm(op.apply(x) - op.apply(y))
                rhs = L * np.linalg.norm(x - y)
                stats.append(lhs <= rhs)

            assert all(stats)

    def test_squeeze(self, op):
        # op.squeeze() sub-classes to Func for scalar outputs, and is transparent otherwise.
        if func_name() not in self.disable_test:
            if op.codim == 1:
                assert isinstance(op.squeeze(), pyco.Func)
            else:
                assert op.squeeze() is op

    @pytest.mark.skip(reason="Requires some scaffolding first.")
    def test_specialize(self, op, _klass):
        if func_name() not in self.disable_test:
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
    def test_codim(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.codim == 1

    def test_squeeze(self, op):
        if func_name() not in self.disable_test:
            assert op.squeeze() is op
