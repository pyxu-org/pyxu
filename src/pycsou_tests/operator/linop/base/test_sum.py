import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pycsou.info.deps as pycd
import pycsou.info.ptype as pyct
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou_tests.operator.conftest as conftest


class SumMixin:
    @pytest.fixture
    def _spec(self, request) -> tuple[pyct.NDArrayShape, pyct.NDArrayAxis]:
        # (arg_shape, axis) raw user input [specified in sub-classes]
        raise NotImplementedError

    @pytest.fixture
    def arg_shape(self, _spec) -> pyct.NDArrayShape:
        arg_shape, _ = _spec  # raw user input
        if not isinstance(arg_shape, cabc.Sequence):
            arg_shape = (arg_shape,)
        return arg_shape  # canonical form

    @pytest.fixture
    def axis(self, arg_shape, _spec) -> pyct.NDArrayAxis:
        _, axis = _spec  # raw user input
        if axis is None:
            axis = tuple(range(len(arg_shape)))
        elif not isinstance(axis, cabc.Sequence):
            axis = (axis,)
        return axis  # canonical form

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(
        self,
        _spec,  # used instead of arg_shape/axis fixtures to get raw user inputs
        request,
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        arg_shape, axis = _spec
        op = pycl.Sum(arg_shape=arg_shape, axis=axis)
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape, axis) -> pyct.OpShape:
        arg_shape = np.array(arg_shape, dtype=int)
        axis = np.array(axis, dtype=int)

        sum_shape = arg_shape.copy()
        sum_shape[axis] = 1

        dim = arg_shape.prod()
        codim = sum_shape.prod()
        return (codim, dim)

    @pytest.fixture
    def data_apply(self, arg_shape, axis) -> conftest.DataLike:
        rng = np.random.default_rng(51)  # for reproducibility
        arr = rng.normal(size=arg_shape)
        out = arr.sum(axis=axis)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestSumLinOp(SumMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            ((5, 3), 0),
            ((5, 3), 1),
            ((5, 3), -1),
            ((5, 3, 4), (1, 2)),  # multiple axes
            ((5, 3, 4), (0, 2)),  # multiple axes
            ((5, 3, 4), (1, 2)),  # multiple axes
        ]
    )
    def _spec(self, request) -> tuple[pyct.NDArrayShape, pyct.NDArrayAxis]:
        return request.param


class TestSumLinFunc(SumMixin, conftest.LinFuncT):
    @pytest.fixture(
        params=[
            (5, None),
            ((5,), None),
            ((5, 3, 4), None),
            (5, 0),
            ((5,), (0,)),
            ((5, 3, 4), (0, 1, 2)),
        ]
    )
    def _spec(self, request) -> tuple[pyct.NDArrayShape, pyct.NDArrayAxis]:
        return request.param
