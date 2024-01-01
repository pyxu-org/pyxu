import itertools

import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestSum(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     axes (user-specified),
            #     axes (canonical).
            # 1D cases --------------------
            (5, (5,), None, (0,)),
            (5, (5,), 0, (0,)),
            (5, (5,), -1, (0,)),
            # 2D cases --------------------
            ((5, 3), (5, 3), None, (0, 1)),
            ((5, 3), (5, 3), 0, (0,)),
            ((5, 3), (5, 3), 1, (1,)),
            ((5, 3), (5, 3), (0, -1), (0, 1)),
            # 3D cases --------------------
            ((5, 3, 4), (5, 3, 4), None, (0, 1, 2)),
            ((5, 3, 4), (5, 3, 4), -2, (1,)),
            ((5, 3, 4), (5, 3, 4), (0, -1), (0, 2)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def axis(self, _spec) -> pxt.NDArrayAxis:
        # canonical axes
        return _spec[3]

    @pytest.fixture
    def codim_shape(self, dim_shape, axis) -> pxt.NDArrayShape:
        sh = list(dim_shape)
        for ax in axis:
            sh[ax] = 1
        return tuple(sh)

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        dim_shape, axis = _spec[0], _spec[2]  # user-specified version
        ndi, width = request.param

        op = pxo.Sum(
            dim_shape=dim_shape,
            axis=axis,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape, axis) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x.sum(axis=axis, keepdims=True)
        return dict(
            in_=dict(arr=x),
            out=y,
        )
