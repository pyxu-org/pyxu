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
    # f_{h}: \bR^{M1,...,MD} -> \bR^{M1,...,MD}
    #          x             -> h \circ x, h \in \bR^{MD}
    #                           [circ-conv along last axis.]
    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        h: pxt.NDArray,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )

        M = h.size
        assert M == self.dim_shape[-1], "Filter must have same length as last axis."

        self.lipschitz = M * (h**2).sum()  # Frobenius norm
        self._h_fw = h.reshape(-1)
        self._h_bw = self._h_fw[[0, *np.arange(1, M)[::-1]]]

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self._circ_convolve(self._h_fw, arr)

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self._circ_convolve(self._h_bw, arr)

    @staticmethod
    def _circ_convolve(
        h: pxt.NDArray,  # (MD,)
        x: pxt.NDArray,  # (..., M1,...,MD)
    ) -> pxt.NDArray:
        # DASK.FFT() only works along non-chunked axes, so we work around the problem
        ndi = pxd.NDArrayInfo.from_obj(x)
        if using_dask := ndi == pxd.NDArrayInfo.DASK:
            xp = pxu.get_array_module(x._meta)
            _x = pxu.compute(x)
        else:
            xp = ndi.module()
            _x = x

        fw = lambda _: xp.fft.fft(_, axis=-1)
        bw = lambda _: xp.fft.ifft(_, axis=-1)
        h = xp.array(h, dtype=x.dtype)
        out = bw(fw(h) * fw(_x)).real

        if using_dask:
            xp = ndi.module()
            out = xp.from_array(out, chunks=x.chunks)
        return out.astype(x.dtype, copy=False)


class TestCircularConvolution(conftest.NormalOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, conv_filter, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = CircularConvolution(dim_shape, conv_filter)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture
    def conv_filter(self, dim_shape) -> np.ndarray:
        M = dim_shape[-1]
        h = self._random_array((M,), seed=53)
        return h

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(
        self,
        dim_shape,
        conv_filter,
        request,
    ) -> conftest.DataLike:
        seed = request.param

        M = dim_shape[-1]
        x = self._random_array(dim_shape, seed=seed)
        y = np.zeros(dim_shape)
        for n in range(M):
            for k in range(M):
                y[..., n] += x[..., k] * conv_filter[n - k % M]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
