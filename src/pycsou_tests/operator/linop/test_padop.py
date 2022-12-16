import numpy as np
import pytest

import pycsou.operator as pycob
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


class TestPadOp(conftest.LinOpT):
    @pytest.fixture(params=[1, 3])
    def ndim(self, request):
        return request.param

    @pytest.fixture(params=[(1, 0), (0, 2), (2, 1)])
    def pad_width(self, request, ndim):
        return (request.param,) * ndim

    @pytest.fixture
    def arg_shape(self, ndim):
        return (6,) * ndim

    @pytest.fixture
    def out_shape(self, arg_shape, pad_width):
        return tuple([s + np.sum(pad_width[i]) for i, s in enumerate(arg_shape)])

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture(params=["constant", "wrap", "reflect", "symmetric", "edge"])
    def pad_mode(self, request):
        return request.param

    @pytest.fixture
    def spec(self, arg_shape, pad_width, pad_mode, ndi, width) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        op = pycob.PadOp(arg_shape=arg_shape, pad_width=pad_width, mode=pad_mode)
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, op, arg_shape, pad_width, pad_mode) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility
        out = np.pad(arr.reshape(arg_shape), pad_width=pad_width, mode=pad_mode)

        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_adjoint(self, op, out_shape) -> conftest.DataLike:
        arr = self._random_array((op.codim,), seed=20)  # random seed for reproducibility
        out = (op.asarray().T @ arr.T).T

        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_shape(self, arg_shape, pad_width) -> pyct.OpShape:
        size_in = np.prod(arg_shape).item()
        size_out = np.prod([s + np.sum(pad_width[i]) for i, s in enumerate(arg_shape)]).item()
        sh = (size_out, size_in)
        return sh
