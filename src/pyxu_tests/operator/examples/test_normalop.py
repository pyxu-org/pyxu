import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class CircularConvolution(pxa.NormalOp):
    # f_{h}: \bR^{N} -> \bR^{N}
    #          x     -> h \circ x, h \in \bR^{N}
    def __init__(self, h: pxt.NDArray):
        N = h.size
        super().__init__(shape=(N, N))
        self.lipschitz = N * (h**2).sum()  # Frobenius norm

        self._h_fw = h.reshape(-1)
        self._h_bw = self._h_fw[[0, *np.arange(1, N)[::-1]]]

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        return self._circ_convolve(self._h_fw, arr)

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return self._circ_convolve(self._h_bw, arr)

    @staticmethod
    def _circ_convolve(h, x):
        xp = pxu.get_array_module(x)
        fw = lambda _: xp.fft.fft(_, axis=-1)
        bw = lambda _: xp.fft.ifft(_, axis=-1)
        h = xp.array(h, dtype=x.dtype)
        out = bw(fw(h) * fw(x)).real
        return out.astype(x.dtype, copy=False)


class TestCircularConvolution(conftest.NormalOpT):
    filter = conftest.NormalOpT._random_array((10,), seed=2)

    @pytest.fixture(
        params=itertools.product(
            (CircularConvolution(h=filter),),
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self):
        return (self.filter.size, self.filter.size)

    @pytest.fixture
    def data_apply(self):
        F = self.filter
        N = F.size
        arr = self._random_array((N,))
        out = np.zeros((N,))
        for n in range(N):
            for k in range(N):
                out[n] += arr[k] * F[n - k % N]
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
