import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Sum(pxa.LinOp):
    # f: \bR^{M1,...,MD} -> \bR^{M1,...,M(D-1)}
    #      x             -> x.sum(axis=-1)
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,  # temporary; just to canonicalize shapes
        )

        if self.dim_rank == 1:
            self._codim_shape = (1,)
        else:
            self._codim_shape = self.dim_shape[:-1]

        self.lipschitz = np.sqrt(self.codim_shape[-1])

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        if self.dim_rank == 1:
            y = arr.sum(axis=-1, keepdims=True)
        else:
            y = arr.sum(axis=-1)
        return y

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        sh = arr.shape[: -self.codim_rank]

        if self.dim_rank == 1:
            y = xp.broadcast_to(arr, (*sh, *self.dim_shape))
        else:
            y = xp.broadcast_to(arr[..., np.newaxis], (*sh, *self.dim_shape))
        return y


class TestSum(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = Sum(dim_shape)
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
        if len(dim_shape) == 1:
            return (1,)
        else:
            return dim_shape[:-1]

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = x.sum(axis=-1, keepdims=True)
        if len(dim_shape) > 1:
            y = y[..., 0]

        return dict(
            in_=dict(arr=x),
            out=y,
        )
