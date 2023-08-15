import itertools

import numpy as np
import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


def filterF(N: int) -> np.ndarray:
    hF = np.r_[1.0, np.r_[1 : (N - 1) // 2 + 1], -np.r_[-(N // 2) : 0]]
    hF /= np.abs(hF).max()
    return hF


class PSDConvolution(pxa.PosDefOp):
    # Convolution where the filter coefficients imply operator is positive-definite.
    def __init__(self, N: int):
        assert N % 2 == 1, "Even-length filters are unsupported."
        super().__init__(shape=(N, N))
        self.lipschitz = np.inf
        self._hF = filterF(N)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pxu.get_array_module(arr)
        fw = lambda _: xp.fft.fft(_, axis=-1)
        bw = lambda _: xp.fft.ifft(_, axis=-1)
        hF = xp.array(self._hF, dtype=arr.dtype)
        out = bw(hF * fw(arr)).real
        return out.astype(arr.dtype, copy=False)


class TestPSDConvolution(conftest.PosDefOpT):
    @pytest.fixture(
        params=itertools.product(
            ((11, PSDConvolution(N=11)),),  # dim, op
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        F = np.fft.ifft(filterF(dim)).real
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
