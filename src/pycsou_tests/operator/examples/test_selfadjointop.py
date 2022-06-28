import dask.array as da
import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class CDO2(pyco.SelfAdjointOp):
    # Central Difference of Order 2
    # f: \bR^{N} -> \bR^{N}
    #      x     -> [ -2x[0]+x[1]
    #                     ...
    #                x[k-1]-2x[k]+x[k+1]
    #                     ...
    #                x[N-2]-2x[N-1] ]
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    def _apply(self, arr):
        xp = pycu.get_array_module(arr)
        h = xp.array([1, -2, 1], dtype=arr.dtype)
        out = xp.convolve(arr, h)[1:-1]
        return out

    def _apply_dask(self, arr):
        out = da.map_overlap(
            self._apply,
            arr,
            depth=2,
            boundary=0,
            trim=True,
            dtype=arr.dtype,
        )
        return out

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    @pycu.redirect("arr", DASK=_apply_dask)
    def apply(self, arr):
        return self._apply(arr)


class TestCDO2(conftest.SelfAdjointOpT):
    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def op(self, dim):
        return CDO2(dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        x = self._random_array((dim,))
        y = np.zeros_like(x)
        y[0] = -2 * x[0] + x[1]
        for i in range(1, dim - 1):
            y[i] = x[i - 1] - 2 * x[i] + x[i + 1]
        y[-1] = x[-2] - 2 * x[-1]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
