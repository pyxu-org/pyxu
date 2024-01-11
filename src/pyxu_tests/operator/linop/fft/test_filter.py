import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.linop.test_stencil as test_stencil


# FFTCorrelate has the exact same interface as Stencil(), so use those tests as-is.
class TestFFTCorrelate(test_stencil.TestStencil):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        dim_shape, kernel, center, mode = _spec[0]  # user-provided form

        # transform kernel to right backend
        xp = ndi.module()
        try:
            pxd.NDArrayInfo.from_obj(kernel)  # passes if array object
            kernel = xp.array(kernel, dtype=width.value)
        except Exception:
            kernel = [xp.array(k, dtype=width.value) for k in kernel]

        with pxrt.Precision(width):
            op = pxo.FFTCorrelate(
                dim_shape=dim_shape,
                kernel=kernel,
                center=center,
                mode=mode,
                enable_warnings=False,
            )
        return op, ndi, width


class TestFFTConvolve:
    # A convolution corresponds to a correlation with reversed kernel/center.
    # Since FFTConvolve() inherits from FFTCorrelate(), it is sufficient to test that convolution with
    # shifted Diracs (in 1D) gives the right results.
    #
    # Adapted from TestConvolve().
    @pytest.mark.parametrize(
        ["kernel", "center", "shift"],
        [
            (np.r_[1, 0], (0,), 0),  # k[n] = \delta[n]
            (np.r_[0, 1], (0,), -1),  # k[n] = \delta[n - 1]
            (np.r_[0, 0, 1, 0], (1,), -1),  # k[n] = \delta[n - 1]
            (np.r_[1, 0], (1,), 1),  # k[n] = \delta[n + 1]
            (np.r_[1, 0, 0], (1,), 1),  # k[n] = \delta[n + 1]
            (np.r_[0, 0, 1, 0, 0, 0], (3,), 1),  # k[n] = \delta[n + 1]
        ],
    )
    def test_value1D_apply(self, kernel, center, shift):
        N = 10
        arr = np.arange(N)

        # Compute ground-truth
        if shift == 0:
            out_gt = arr.copy()
        elif shift > 0:  # anti-causal filter
            out_gt = np.pad(arr, pad_width=(0, abs(shift)))[-N:]
        else:  # causal filter
            out_gt = np.pad(arr, pad_width=(abs(shift), 0))[:N]

        # Compute Stencil-based solution
        op = pxo.FFTConvolve(
            dim_shape=arr.shape,
            kernel=kernel,
            center=center,
            mode="constant",
        )
        out = op.apply(arr)

        assert TestFFTCorrelate._metric(out, out_gt, as_dtype=out.dtype)
