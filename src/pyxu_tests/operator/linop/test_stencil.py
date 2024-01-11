import itertools

import numpy as np
import pytest
import scipy.ndimage as snd

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


# We disable NumbaPerformanceWarnings due to solving small-scale problems at test time.
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
class TestStencil(conftest.SquareOpT):
    @pytest.fixture(
        params=[
            # 1D, random center/mode ------------------------
            (
                (10, np.arange(1, 7), (0,), "constant"),
                ((10,), (np.arange(1, 7),), ((0,),), ("constant",)),
            ),
            (
                (10, np.arange(1, 7), (1,), "edge"),
                ((10,), (np.arange(1, 7),), ((1,),), ("edge",)),
            ),
            (
                (10, np.arange(1, 7), (2,), "wrap"),
                ((10,), (np.arange(1, 7),), ((2,),), ("wrap",)),
            ),
            (
                (10, np.arange(1, 7), (3,), "reflect"),
                ((10,), (np.arange(1, 7),), ((3,),), ("reflect",)),
            ),
            (
                (10, np.arange(1, 7), (4,), "symmetric"),
                ((10,), (np.arange(1, 7),), ((4,),), ("symmetric",)),
            ),
            # ND, random center/mode ------------------------
            (
                ((10, 11), np.arange(1, 9).reshape(2, 4), (0, 3), "constant"),
                ((10, 11), (np.arange(1, 9).reshape(2, 4),), ((0, 3),), ("constant", "constant")),
            ),
            (
                ((10, 11), np.arange(1, 9).reshape(2, 4), (1, 2), ("wrap", "reflect")),
                ((10, 11), (np.arange(1, 9).reshape(2, 4),), ((1, 2),), ("wrap", "reflect")),
            ),
            (
                ((10, 11), np.arange(1, 9).reshape(2, 4), (1, 1), ("edge", "symmetric")),
                ((10, 11), (np.arange(1, 9).reshape(2, 4),), ((1, 1),), ("edge", "symmetric")),
            ),
            # ND seperable, random center/mode --------------
            (
                ((10, 11), (np.arange(1, 7), np.arange(2, 5)), (3, 0), "constant"),
                (
                    (10, 11),
                    (np.arange(1, 7).reshape(-1, 1), np.arange(2, 5).reshape(1, -1)),
                    ((3, 0), (0, 0)),
                    ("constant", "constant"),
                ),
            ),
            (
                ((10, 11), (np.arange(1, 7), np.arange(2, 5)), (2, 1), ("edge", "wrap")),
                (
                    (10, 11),
                    (np.arange(1, 7).reshape(-1, 1), np.arange(2, 5).reshape(1, -1)),
                    ((2, 0), (0, 1)),
                    ("edge", "wrap"),
                ),
            ),
            (
                ((10, 11), (np.arange(1, 7), np.arange(2, 5)), (3, 2), ("reflect", "symmetric")),
                (
                    (10, 11),
                    (np.arange(1, 7).reshape(-1, 1), np.arange(2, 5).reshape(1, -1)),
                    ((3, 0), (0, 2)),
                    ("reflect", "symmetric"),
                ),
            ),
        ]
    )
    def _spec(self, request):
        # (dim_shape, kernel, center, mode) configs to test
        # * `request.param[0]` corresponds to raw inputs users provide to Stencil().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical representation
        dim_shape, _, _, _ = _spec[1]
        return dim_shape

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        # canonical representation
        return dim_shape

    @pytest.fixture
    def kernel(self, _spec):
        # canonical representation (NumPy)
        _, kernel, _, _ = _spec[1]
        return kernel

    @pytest.fixture
    def center(self, _spec):
        # canonical representation
        _, _, center, _ = _spec[1]
        return center

    @pytest.fixture
    def mode(self, _spec):
        # canonical representation
        _, _, _, mode = _spec[1]
        return mode

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
            op = pxo.Stencil(
                dim_shape=dim_shape,
                kernel=kernel,
                center=center,
                mode=mode,
                enable_warnings=False,
            )
        return op, ndi, width

    @pytest.fixture(params=[0, 1, 2])  # different seeds to test robustness
    def data_apply(
        self,
        dim_shape,
        kernel,
        center,
        mode,
        request,
    ) -> conftest.DataLike:
        seed = request.param
        x = self._random_array(dim_shape, seed=seed)

        # Pad input in excess of what is stricly required (using Pad(); assumed correct)
        if len(kernel) == 1:  # non-seperable filter
            pad_width = [(w, w) for w in kernel[0].shape]
        else:  # seperable filter(s)
            pad_width = [(k.shape[i],) * 2 for (i, k) in enumerate(kernel)]
        pad = pxo.Pad(
            dim_shape=dim_shape,
            pad_width=pad_width,
            mode=mode,
        )
        corr_in = pad.apply(x)

        # perform correlation via scipy.ndimage.correlate
        corr_out = corr_in.copy()
        for k, c in zip(kernel, center):
            origin = [cc - (n // 2) for (cc, n) in zip(c, k.shape)]
            corr_out = snd.correlate(
                input=corr_out,
                weights=k,
                mode="constant",
                cval=0,
                origin=origin,
            )

        # Trim fat off (using Trim(); assumed correct)
        trim = pxo.Trim(
            dim_shape=corr_out.shape,
            trim_width=pad_width,
        )
        y = trim.apply(corr_out)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestConvolve:
    # A convolution corresponds to a stencil with reversed kernel/center.
    # Since Convolve() inherits from Stencil(), it is sufficient to test that convolution with
    # shifted Diracs (in 1D) gives the right results.
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
        op = pxo.Convolve(
            dim_shape=arr.shape,
            kernel=kernel,
            center=center,
            mode="constant",
        )
        out = op.apply(arr)

        assert TestStencil._metric(out, out_gt, as_dtype=out.dtype)
