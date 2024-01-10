import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Sum(pxa.LinFunc):
    # f: \bR^{M1,...,MD} -> \bR
    #      x             -> x.sum()
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        self.lipschitz = np.sqrt(np.prod(self.dim_shape))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        axis = tuple(range(-self.dim_rank, 0))
        y = arr.sum(axis=axis)[..., np.newaxis]
        return y

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.codim_rank]
        extend = (np.newaxis,) * (self.dim_rank - 1)

        xp = pxu.get_array_module(arr)
        y = xp.broadcast_to(
            arr[..., *extend],
            (*sh, *self.dim_shape),
        )
        return y


class TestSum(conftest.LinFuncT):
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

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = np.array([x.sum()])

        return dict(
            in_=dict(arr=x),
            out=y,
        )
