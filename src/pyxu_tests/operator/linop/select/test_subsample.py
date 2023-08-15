import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class SubSampleMixin:
    @pytest.fixture
    def arg_shape(self, request):  # anything which is understood as shape parameter of np.zeros()
        raise NotImplementedError

    @pytest.fixture
    def indices(self, request):  # anything which can index an NDArray
        raise NotImplementedError

    @pytest.fixture
    def sub_shape(self, arg_shape, indices) -> tuple[int]:
        sh = np.zeros(arg_shape)[indices].shape
        return sh

    @pytest.fixture
    def data_shape(self, arg_shape, sub_shape):
        return (np.prod(sub_shape), np.prod(arg_shape))

    @pytest.fixture
    def data_apply(self, arg_shape, indices):
        arr = np.arange(np.prod(arg_shape)).reshape(arg_shape)
        out = arr[indices].reshape(-1)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out,
        )


class TestSubSample(SubSampleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            10,
            (10,),
            (10, 5, 2),
        ]
    )
    def arg_shape(self, request):
        return request.param

    @pytest.fixture(
        params=[
            0,  # integer indexing
            [1],  # list indexing
            slice(0, None, 2),
            np.r_[1],  # using an NDArray
            [i % 2 == 0 for i in range(10)],  # boolean mask
        ]
    )
    def indices(self, request):
        return (request.param,)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, arg_shape, indices, request):
        ndi, width = request.param
        op = pxl.SubSample(arg_shape, *indices)
        return op, ndi, width


class TestTrim(SubSampleMixin, conftest.LinOpT):
    @pytest.fixture(
        params=[
            (
                (10, 5, 11),
                0,  # no trimming
                (slice(None), slice(None), slice(None)),
            ),
            (
                (10, 5, 11),
                1,  # equal trim/dimension
                (slice(1, 9), slice(1, 4), slice(1, 10)),
            ),
            (
                (10, 5, 11),
                (1, 2, 3),  # different trim/dimension
                (slice(1, 9), slice(2, 3), slice(3, 8)),
            ),
            (
                (10, 5, 11),
                ((0, 0), (1, 0), (2, 3)),  # different head/tail/dimension
                (slice(None), slice(1, 5), slice(2, 8)),
            ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, trim_width, index_spec-equivalent) configs to test
        arg_shape, trim_width, index_spec = request.param
        return arg_shape, trim_width, index_spec

    @pytest.fixture
    def arg_shape(self, _spec):
        arg_shape, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def trim_width(self, _spec):
        _, trim_width, _ = _spec
        return trim_width

    @pytest.fixture
    def indices(self, _spec):
        _, _, index = _spec
        return index

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, arg_shape, trim_width, request):
        ndi, width = request.param
        op = pxl.Trim(arg_shape, trim_width)
        return op, ndi, width
