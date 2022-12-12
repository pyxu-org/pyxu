import numpy as np
import pytest

import pycsou.operator as pycob
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


class TestTrimOp(conftest.LinOpT):
    @pytest.fixture(params=[1, 3])
    def ndim(self, request):
        return request.param

    @pytest.fixture(params=[(1, 0), (0, 2), (2, 1)])
    def widths(self, request, ndim):
        return (request.param,) * ndim

    @pytest.fixture
    def arg_shape(self, ndim):
        return (6,) * ndim

    @pytest.fixture
    def out_shape(self, arg_shape, widths):
        return tuple([s - np.sum(widths[i]) for i, s in enumerate(arg_shape)])

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture
    def spec(self, arg_shape, widths, ndi, width) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        op = pycob.TrimOp(arg_shape=arg_shape, trim_width=widths)
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, op, arg_shape, widths) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility
        out = arr.reshape(-1, *arg_shape)
        for i in range(len(arg_shape)):
            # unpad in the i-th axis
            slices = tuple(
                [
                    slice(None),
                ]
                + [
                    slice(None) if i != j else slice(widths[i][0], arg_shape[i] - widths[i][1])
                    for j in range(len(arg_shape))
                ]
            )
            out = out[slices]

        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_adjoint(self, op, out_shape, widths) -> conftest.DataLike:
        arr = self._random_array((op.codim,), seed=20)  # random seed for reproducibility
        out = arr.reshape(-1, *out_shape)
        for i in range(len(widths)):
            # pad in the i-th axis
            width = ((0, 0),) + tuple([(0, 0) if i != j else width for j, width in enumerate(widths)])
            out = np.pad(array=out, pad_width=width, mode="constant", constant_values=0.0)
        return dict(
            in_=dict(arr=arr),
            out=out.ravel(),
        )

    @pytest.fixture
    def data_shape(self, arg_shape, widths) -> pyct.OpShape:
        size_in = np.prod(arg_shape).item()
        size_out = np.prod([s - np.sum(widths[i]) for i, s in enumerate(arg_shape)]).item()
        sh = (size_out, size_in)
        return sh
