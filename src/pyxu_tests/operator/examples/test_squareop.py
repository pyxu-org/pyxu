import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class CumSum(pxa.SquareOp):
    # f: \bR^{M1,...,MD} -> \bR^{M1,...,MD}
    #      x             -> x.cumsum(axis=-1)
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )

        N = self.dim_shape[-1]
        self.lipschitz = np.sqrt(N * (N + 1) / 2)  # Frobenius norm

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = arr.cumsum(axis=-1)
        return y

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        y = arr[..., ::-1].cumsum(axis=-1)[..., ::-1]
        return y


class TestCumSum(conftest.SquareOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = CumSum(dim_shape)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = x.copy()
        for i in range(1, dim_shape[-1]):
            y[..., i] += y[..., i - 1]

        return dict(
            in_=dict(arr=x),
            out=y,
        )
