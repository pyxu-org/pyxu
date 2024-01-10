import itertools

import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ScaleDown(pxa.OrthProjOp):
    # Drop the last component of an NDArray
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        assert self.dim_shape[-1] > 1, "Not a orth-projection."

    @pxrt.enforce_precision("arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr.copy()
        out[..., -1] = 0
        return out


class TestScaleDown(conftest.OrthProjOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = ScaleDown(dim_shape)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)

        y = x.copy()
        idx = dim_shape[-1] - 1
        y[..., idx] = 0

        return dict(
            in_=dict(arr=x),
            out=y,
        )
