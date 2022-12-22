import enum
import itertools

import numpy as np
import pytest
import scipy.ndimage as scimage

import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


@enum.unique
class Modes(enum.Enum):

    NONE = "constant"
    WRAP = "wrap"
    # SYMMETRIC = "symmetric"
    # EDGE = "edge"
    # REFLECT = "reflect"


# We disable PrecisionWarnings since Stencil() is not precision-agnostic, but the outputs
# computed must still be valid. We also disable NumbaPerformanceWarnings, as these appear
# due to using the overkill use of GPU for very small test input arrays.
@pytest.mark.filterwarnings(
    "ignore::pycsou.util.warning.PrecisionWarning", "ignore::numba.core.errors.NumbaPerformanceWarning"
)
class MixinStencil(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            [1, 3], [0, 2]  # (1) 1D and (3) 3D kernels  # [0, 2]  # right or left centered kernels (wrt origin=1)
        )
    )
    def stencil_params(self, request):
        ndim, pos = request.param
        stencil_coefs = np.expand_dims(self._random_array((3,), seed=20), axis=list(np.arange(ndim - 1)))
        center = np.zeros(ndim, dtype=int)
        center[-1] = pos
        return stencil_coefs, center

    @pytest.fixture
    def stencil_coefs(self, stencil_params):
        return stencil_params[0]

    @pytest.fixture
    def center(self, stencil_params):
        return stencil_params[1]

    @pytest.fixture
    def arg_shape(self, stencil_params):
        return (8,) * stencil_params[0].ndim

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.OpShape:
        size = np.prod(arg_shape).item()
        sh = (size, size)
        return sh

    @pytest.fixture(params=Modes)
    def mode(self, request):
        return request.param

    @pytest.fixture
    def spec(
        self, stencil_coefs, center, arg_shape, ndi, width, mode, correlate_or_convolve
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        cls = {"correlate": pycl.Correlation, "convolve": pycl.Convolution}[correlate_or_convolve]
        op = cls(
            stencil_coefs=ndi.module().asarray(stencil_coefs, dtype=width.value),
            center=center,
            arg_shape=arg_shape,
            mode=mode.value,
            enable_warnings=False,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, op, center, arg_shape, mode, correlate_or_convolve) -> conftest.DataLike:

        kernel, center = op.stencil_coefs, op.center
        xp = pycu.get_array_module(kernel)
        kernel = kernel.get() if xp.__name__ == "cupy" else kernel
        width_right = np.atleast_1d(kernel.shape) - center - 1
        widths = tuple([(max(center[i].item(), width_right[i].item()),) * 2 for i in range(len(arg_shape))])

        arr = self._random_array(arg_shape)
        mode = op._mode

        if len(set(mode)) == 1:  # uni-mode
            out = np.pad(
                array=arr,
                pad_width=widths,
                mode=mode[0],
            )
        else:  # multi-mode
            N_dim = len(arg_shape)
            out = arr
            for i in range(N_dim):
                p = [(0, 0)] * N_dim
                p[i] = widths[i]
                out = np.pad(
                    array=out,
                    pad_width=p,
                    mode=mode[i],
                )
        origin = center - (np.array(kernel.shape) // 2)

        f = {"correlate": scimage.correlate, "convolve": scimage.convolve}[correlate_or_convolve]
        out = f(
            out,
            kernel,
            origin=origin,
            mode="constant",
            cval=np.nan,
        )

        selector = [slice(lhs, N - rhs) for N, (lhs, rhs) in zip(out.shape, widths)]
        out = out[tuple(selector)]

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestCorrelation(MixinStencil):
    @pytest.fixture
    def correlate_or_convolve(self):
        return "correlate"


class TestConvolution(MixinStencil):
    @pytest.fixture
    def correlate_or_convolve(self):
        return "convolve"
