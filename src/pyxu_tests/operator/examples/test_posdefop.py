import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


def filterF(M: int) -> np.ndarray:
    hF = np.r_[1.0, np.r_[1 : (M - 1) // 2 + 1], -np.r_[-(M // 2) : 0]]
    hF /= np.abs(hF).max()
    return hF


class PSDConvolution(pxa.PosDefOp):
    # Convolution (along last axis) where filter coefficients imply operator is positive-definite.
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )

        M = self.dim_shape[-1]
        assert M % 2 == 1, "Even-length filters are unsupported."

        self.lipschitz = np.inf
        self._hF = filterF(M)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # DASK.FFT() only works along non-chunked axes, so we work around the problem
        ndi = pxd.NDArrayInfo.from_obj(arr)
        if using_dask := ndi == pxd.NDArrayInfo.DASK:
            xp = pxu.get_array_module(arr._meta)
            _arr = pxu.compute(arr)
        else:
            xp = ndi.module()
            _arr = arr

        fw = lambda _: xp.fft.fft(_, axis=-1)
        bw = lambda _: xp.fft.ifft(_, axis=-1)
        hF = xp.array(self._hF, dtype=arr.dtype)
        out = bw(hF * fw(_arr)).real

        if using_dask:
            xp = ndi.module()
            out = xp.from_array(out, chunks=arr.chunks)
        return out.astype(arr.dtype, copy=False)


class TestPSDConvolution(conftest.PosDefOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = PSDConvolution(dim_shape)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 7),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(
        self,
        dim_shape,
        request,
    ) -> conftest.DataLike:
        seed = request.param

        M = dim_shape[-1]
        conv_filter = np.fft.ifft(filterF(M)).real

        x = self._random_array(dim_shape, seed=seed)
        y = np.zeros(dim_shape)
        for n in range(M):
            for k in range(M):
                y[..., n] += x[..., k] * conv_filter[n - k % M]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
