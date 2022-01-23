import collections.abc as cabc
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


# Naming conventions
# ------------------
#
# * data_<property>()
#       Return expected object of `op.<property>`.
#
# * test_<property>(op, data_<property>):
#       Verify property values.
#
# * data_<method>()
#       Return mappings of the form dict(in_=dict(), out=Any), where:
#         * in_ are kwargs to `op.<method>()`;
#         * out denotes the output of `op.method(**data[in_])`.
#
# * test_ioXX_<method>(op, data_<method>)
#       Verify that <method>, when fed with `**data['in_']`, satisfies certain input/output
#       relationships.
#
# * test_prec_<method>(op, data_<method>)
#       Verify that input/output precision can be controlled by `pycsou.runtime.Precision()`.
#
# * data_math_<method>()
#       Special test data for mathematical identities.
#
# * test_mathXX_YY(op, ...)
#       Verify some mathematical property.
DataLike = cabc.Mapping[str, typ.Any]


class MapT:
    disable_test: cabc.Collection[str] = frozenset()

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

    # -------------------------------------------------------------------------
    @pytest.fixture
    def data_shape(self) -> pyct.Shape:
        # override in subclass with the shape of op.
        # Don't return `op.shape`: hard-code what you are expecting.
        raise NotImplementedError

    def test_io_shape(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.shape == data_shape

    def test_io_dim(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.dim == data_shape[1]

    def test_io_codim(self, op, data_shape):
        if func_name() not in self.disable_test:
            assert op.codim == data_shape[0]

    # -------------------------------------------------------------------------
    @pytest.fixture
    def data_lipschitz(self) -> DataLike:
        # override in subclass with the Lipschitz constant of op.
        # Don't return `op.lipschitz()`: hard-code what you are expecting.
        raise NotImplementedError

    def test_io_lipschitz(self, op, data_lipschitz):
        if func_name() not in self.disable_test:
            in_ = op.lipschitz(**data_lipschitz["in_"])
            out = data_lipschitz["out"]
            assert np.isclose(in_, out)

    # -------------------------------------------------------------------------
    @pytest.fixture
    def data_apply(self) -> DataLike:
        # override in subclass with 1D input/outputs of op.apply().
        # Arrays should be NumPy-only. (Internal machinery will transform to different
        # backend/precisions as needed.)
        raise NotImplementedError

    @pytest.fixture
    def _data_apply(self, data_apply, xp, width) -> DataLike:
        # generate all combinations of inputs. (For internal use only.)
        # outputs are left unchanged: different tests should transform them as required.
        arr = xp.array(data_apply["in_"]["arr"], dtype=width.value)
        data = dict(
            in_=dict(arr=arr),
            out=data_apply["out"],
        )
        return data

    def test_io1D_apply(self, op, _data_apply):
        # works on 1D inputs -> output has right shape + values
        if func_name() not in self.disable_test:
            out_gt = _data_apply["out"]

            in_ = _data_apply["in_"]
            out = pycu.compute(op.apply(**in_))

            assert out.ndim == in_["arr"].ndim
            assert np.allclose(out, out_gt)

    def test_ioND_apply(self, op, _data_apply):
        # works on ND inputs -> output has right shape + values
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
            assert np.allclose(out, out_gt)

    def test_ioBackend_apply(self, op, _data_apply):
        # op.apply() preserves array backend
        if func_name() not in self.disable_test:
            in_ = _data_apply["in_"]
            out = op.apply(**in_)

            assert type(out) == type(in_["arr"])

    def test_prec_apply(self, op, _data_apply):
        # op.apply() respects environment precision
        if func_name() not in self.disable_test:
            in_ = _data_apply["in_"]
            stats = []
            for width in pycrt.Width:
                with pycrt.Precision(width):
                    out = op.apply(**in_)
                stats.append(out.dtype == width.value)

            assert all(stats)

    # -------------------------------------------------------------------------
    # Mathematical identities to verify
    @pytest.fixture
    def data_math_lipschitz(self) -> cabc.Collection[np.ndarray]:
        # override in subclass with at least 2 evaluation points for .apply()
        # Used to verify if .apply() satisfies the Lipschitz condition.
        # Arrays should be NumPy-only.
        raise NotImplementedError

    def test_math_lipschitz(self, op, data_lipschitz, data_math_lipschitz):
        # op.apply() satisfies Lipschitz condition:
        #   \norm{f(x) - f(y)}{2} \le L * \norm{x - y}{2}
        if func_name not in self.disable_test:
            L = op.lipschitz(**data_lipschitz["in_"])

            stats = []
            for x, y in itertools.combinations(data_math_lipschitz, 2):
                lhs = np.linalg.norm(op.apply(x) - op.apply(y))
                rhs = L * np.linalg.norm(x - y)
                stats.append(lhs <= rhs)

            assert all(stats)

    # -------------------------------------------------------------------------
    # TODO: test squeeze()
    # TODO: test specialize()
