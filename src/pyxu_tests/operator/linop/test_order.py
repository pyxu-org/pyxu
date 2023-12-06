import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestTransposeAxes(conftest.UnitOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     arg_shape (user-specified),
            #     arg_shape (canonical),
            #     axes (user-specified),
            #     axes (canonical).
            # 1D cases --------------------
            (5, (5,), None, (0,)),
            ((5,), (5,), None, (0,)),
            ((5,), (5,), 0, (0,)),
            # 2D cases --------------------
            ((5, 3), (5, 3), None, (1, 0)),
            ((5, 3), (5, 3), (1, 0), (1, 0)),
            ((5, 3), (5, 3), (0, 1), (0, 1)),
            # 3D cases --------------------
            ((5, 3, 4), (5, 3, 4), None, (2, 1, 0)),
            ((5, 3, 4), (5, 3, 4), (0, 1, 2), (0, 1, 2)),
            ((5, 3, 4), (5, 3, 4), (1, 0, 2), (1, 0, 2)),
            ((5, 3, 4), (5, 3, 4), (2, 0, 1), (2, 0, 1)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical arg_shape
        return _spec[1]

    @pytest.fixture
    def axes(self, _spec) -> pxt.NDArrayAxis:
        # canonical axes
        return _spec[3]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        arg_shape, axes = _spec[0], _spec[2]  # user-specified version
        ndi, width = request.param

        op = pxo.TransposeAxes(
            arg_shape=arg_shape,
            axes=axes,
        )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape) -> pxt.OpShape:
        codim = dim = np.prod(arg_shape)
        return (codim, dim)

    @pytest.fixture
    def data_apply(self, arg_shape, axes) -> conftest.DataLike:
        arr_gt = np.arange(np.prod(arg_shape))
        arr = arr_gt.reshape(arg_shape)
        out = arr.transpose(axes)
        out_gt = out.reshape(-1)

        return dict(
            in_=dict(arr=arr_gt),
            out=out_gt,
        )
