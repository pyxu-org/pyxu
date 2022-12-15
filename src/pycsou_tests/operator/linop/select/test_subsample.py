import itertools

import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


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
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, arg_shape, indices, request):
        ndi, width = request.param
        op = pycl.SubSample(arg_shape, *indices)
        return op, ndi, width

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
