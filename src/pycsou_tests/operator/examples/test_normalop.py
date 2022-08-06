import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


class CircularConvolution(pyca.NormalOp):
    # f_{h}: \bR^{N} -> \bR^{N}
    #          x     -> h \circ x, h \in \bR^{N}
    def __init__(self, h: pyct.NDArray):
        N = h.size
        super().__init__(shape=(N, N))
        self._lipschitz = N * (h**2).sum()  # Frobenius norm

        self._h_fw = h.reshape(-1)
        self._h_bw = self._h_fw[[0, *np.arange(1, N)[::-1]]]

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return self._circ_convolve(self._h_fw, arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return self._circ_convolve(self._h_bw, arr)

    @staticmethod
    def _circ_convolve(h, x):
        xp = pycu.get_array_module(x)
        fw = lambda _: xp.fft.fft(_, axis=-1)
        bw = lambda _: xp.fft.ifft(_, axis=-1)
        h = xp.array(h, dtype=x.dtype)
        out = bw(fw(h) * fw(x)).real
        return out.astype(x.dtype, copy=False)


class TestCircularConvolution(conftest.NormalOpT):
    @pytest.fixture
    def filter(self):
        return self._random_array((5,), seed=2)

    @pytest.fixture
    def op(self, filter):
        return CircularConvolution(h=filter)

    @pytest.fixture
    def data_shape(self, filter):
        return (filter.size, filter.size)

    @pytest.fixture
    def data_apply(self, filter):
        N = filter.size
        arr = self._random_array((N,))
        out = np.zeros((N,))
        for n in range(N):
            for k in range(N):
                out[n] += arr[k] * filter[n - k % N]
        return dict(
            in_=dict(arr=arr),
            out=out,
        )
