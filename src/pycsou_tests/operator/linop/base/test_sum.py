import itertools

import numpy as np
import pytest

import pycsou.operator.linop.base as pycob
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class SumMixin:
    @pytest.fixture
    def arg_shape(self, ndim):
        return (6,) * ndim

    @pytest.fixture
    def adjoint_shape(self, arg_shape, axis):
        axis = tuple(np.arange(len(arg_shape))) if axis is None else axis
        axis = (axis,) if not isinstance(axis, tuple) else axis
        axis_ = tuple([ax if ax != -1 else len(arg_shape) - 1 for ax in axis])
        return tuple([dim for ax, dim in enumerate(arg_shape) if ax not in axis_])

    @pytest.fixture
    def data_shape(self, ndim, axis):
        axis = tuple(np.arange(ndim)) if axis is None else axis
        axis = (axis,) if not isinstance(axis, tuple) else axis
        return 6 ** (ndim - len(axis)), 6**ndim

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, arg_shape, axis, request):
        op = pycob.Sum(arg_shape=arg_shape, axis=axis)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, arg_shape, axis):
        arr = self._random_array(arg_shape)
        out = arr.sum(axis=axis)
        return dict(
            in_=dict(arr=arr.ravel()),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_adjoint(self, arg_shape, axis, adjoint_shape):
        arr = self._random_array(adjoint_shape)
        ##
        # Copied from Pylops
        axis = tuple(np.arange(len(arg_shape))) if axis is None else axis
        axis = (axis,) if not isinstance(axis, tuple) else axis
        tile = np.ones(len(arg_shape), dtype=int)
        tile[list(axis)] = np.array(arg_shape)[list(axis)]
        out = np.expand_dims(arr, axis)
        out = np.tile(out, tile)
        ##
        return dict(
            in_=dict(arr=arr.ravel()),
            out=out.ravel(),
        )


# We disable PrecisionWarnings since DiagonalOp() is not precision-agnostic, but the outputs
# computed must still be valid.
class TestSumLinOp(SumMixin, conftest.LinOpT):
    @pytest.fixture(params=[(2, (0,)), (2, (-1,)), (3, (0, -1))])
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def ndim(self, _spec):
        return _spec[0]

    @pytest.fixture
    def axis(self, _spec):
        return _spec[1]


class TestSumLinFunc(SumMixin, conftest.LinFuncT):
    @pytest.fixture(params=[(1, 0), (1, -1), (3, None)])
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def ndim(self, _spec):
        return _spec[0]

    @pytest.fixture
    def axis(self, _spec):
        return _spec[1]
