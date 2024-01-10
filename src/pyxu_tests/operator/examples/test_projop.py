import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class Oblique(pxa.ProjOp):
    # f: \bR^{M1,...,MD} -> \bR^{M1,...,MD}
    #      x             -> (\alpha E_{N,1} + E_{N, N}) x
    #                       [along last axis]
    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        alpha: float,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self._alpha = float(alpha)
        assert self.dim_shape[-1] > 1, "Not a projection."

    @pxrt.enforce_precision("arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = np.zeros_like(arr)
        out[..., -1] = (self._alpha * arr[..., 0]) + arr[..., -1]
        return out

    @pxrt.enforce_precision("arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = np.zeros_like(arr)
        out[..., 0] = self._alpha * arr[..., -1]
        out[..., -1] = arr[..., -1]
        return out


class TestOblique(conftest.ProjOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, alpha, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = Oblique(dim_shape, alpha)
        return op, ndi, width

    @pytest.fixture(params=[0.8, 3.1])
    def alpha(self, request) -> float:
        return request.param

    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[0, 19, 103])
    def data_apply(self, dim_shape, alpha, request):
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = np.zeros_like(x)
        y[..., -1] = alpha * x[..., 0] + x[..., -1]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
