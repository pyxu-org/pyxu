import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Sin(pxa.DiffMap):
    # f: \bR^{M1,...,MD} -> \bR^{M1,...,MD}
    #      x             -> sin(x)
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )
        self.lipschitz = 1
        self.diff_lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        y = xp.sin(arr)
        return y

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        xp = pxu.get_array_module(arr)
        op = pxo.DiagonalOp(
            vec=xp.cos(arr),
            dim_shape=self.dim_shape,
        )
        return op


class TestSin(conftest.DiffMapT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = Sin(dim_shape)
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
        y = np.sin(x)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x
