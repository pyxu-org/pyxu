import itertools

import numpy as np
import pytest
import scipy.ndimage as scimage
import skimage

import pycsou.operator.linop.filter as pycfilt
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
class FilterMixin(conftest.SquareOpT):
    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.OpShape:
        dim = np.prod(arg_shape).item()
        return (dim, dim)

    @pytest.fixture
    def mode(self):
        return "constant"

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(
        self, arg_shape, mode, filter_klass, filter_kwargs, request
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        with pycrt.Precision(width):
            op = filter_klass(
                arg_shape=arg_shape,
                mode=mode,
                gpu=ndi.name == "CUPY",
                dtype=width.value,
                **filter_kwargs,
            )
        return op, ndi, width


class TestMovingAverage(FilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.MovingAverage

    @pytest.fixture(
        params=[
            # arg_shape, size, center, origin (for scipy)
            ((5,), 4, (0,), -2),
            ((5, 3, 4), 3, (0, 1, 2), (-1, 0, 1)),
            ((5, 3, 4), (5, 1, 3), None, 0),
        ]
    )
    def _spec(self, request):
        # (arg_shape, size, center, origin) configs to test.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        arg_shape, _, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def size(self, _spec):
        _, size, _, _ = _spec
        return size

    @pytest.fixture
    def center(self, _spec):
        _, _, center, _ = _spec
        return center

    @pytest.fixture
    def origin(self, _spec):
        _, _, _, origin = _spec
        return origin

    @pytest.fixture
    def filter_kwargs(self, size, center):
        return {"size": size, "center": center}

    @pytest.fixture
    def data_apply(self, _spec, mode) -> conftest.DataLike:
        arg_shape, size, _, origin = _spec
        arr = self._random_array(arg_shape)
        out = scimage.uniform_filter(arr, size=size, mode=mode, origin=origin)

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestGaussian(FilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.Gaussian

    @pytest.fixture(
        params=[
            # arg_shape, sigma, order, truncate
            ((8,), 3, 0, 1),
            ((4, 4, 4), 3, (0, 1, 2), 1),
            ((8, 8, 8), (1, 2, 3), 1, 2),
        ]
    )
    def _spec(self, request):
        # (arg_shape, sigma, order, truncate) configs to test.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        arg_shape, _, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def filter_kwargs(self, _spec):
        _, sigma, order, truncate = _spec
        return {"sigma": sigma, "truncate": truncate, "order": order}

    @pytest.fixture
    def data_apply(self, _spec, mode) -> conftest.DataLike:

        arg_shape, sigma, order, truncate = _spec
        arr = self._random_array(arg_shape)
        out = scimage.gaussian_filter(arr, sigma=sigma, truncate=truncate, order=order, mode=mode)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestDoG(FilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.DifferenceOfGaussians

    @pytest.fixture(
        params=[
            # arg_shape, low_sigma, high_sigma, low_truncate, high_truncate
            ((8,), 2, 3, 1, 1),
            ((4, 4, 4), 1, (2, 1.5, 3), 1, 1),
            ((8, 8, 8), (1, 2, 3), 4, 1, 1),
        ]
    )
    def _spec(self, request):
        # (arg_shape, low_sigma, high_sigma, low_truncate, high_truncate) configs to test.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        arg_shape, _, _, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def filter_kwargs(self, _spec):
        _, low_sigma, high_sigma, low_truncate, high_truncate = _spec
        return {
            "low_sigma": low_sigma,
            "high_sigma": high_sigma,
            "low_truncate": low_truncate,
            "high_truncate": high_truncate,
        }

    @pytest.fixture
    def data_apply(self, _spec, mode) -> conftest.DataLike:

        arg_shape, low_sigma, high_sigma, low_truncate, high_truncate = _spec
        arr = self._random_array(arg_shape)
        out_low = scimage.gaussian_filter(arr, sigma=low_sigma, truncate=low_truncate, order=0, mode=mode)
        out_high = scimage.gaussian_filter(arr, sigma=high_sigma, truncate=high_truncate, order=0, mode=mode)
        out = out_low - out_high
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestLaplace(FilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.Laplace

    @pytest.fixture(params=[(8,), (4, 4, 4)])
    def arg_shape(self, request):
        return request.param

    @pytest.fixture
    def filter_kwargs(self):
        return dict()

    @pytest.fixture
    def data_apply(self, arg_shape, mode) -> conftest.DataLike:
        arr = self._random_array(arg_shape)
        out = scimage.laplace(arr, mode=mode)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
class EdgeFilterMixin(conftest.DiffMapT):
    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.OpShape:
        dim = np.prod(arg_shape).item()
        return (dim, dim)

    @pytest.fixture
    def mode(self):
        return "constant"

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(
        self, arg_shape, mode, filter_klass, filter_kwargs, request
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        with pycrt.Precision(width):
            op = filter_klass(
                arg_shape=arg_shape,
                mode=mode,
                gpu=ndi.name == "CUPY",
                dtype=width.value,
                **filter_kwargs,
            )
        return op, ndi, width

    @pytest.fixture(
        params=[
            # arg_shape, axis, axis_scipy
            ((4,), (0,), (0,)),
            ((4,), None, (0,)),
            ((4, 4, 4), (0, 2), (0, 2)),
            ((4, 4, 4), None, (0, 1, 2)),
        ]
    )
    def _spec(self, request):
        # arg_shape, axis, axis_scipy
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        arg_shape, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def axis(self, _spec):
        _, axis, _ = _spec
        return axis

    @pytest.fixture
    def axis_skimage(self, _spec):
        _, _, axis_skimage = _spec
        return axis_skimage

    @pytest.fixture
    def filter_kwargs(self, axis):
        return {"axis": axis}

    @pytest.fixture
    def data_apply(self, arg_shape, filter_skimage, axis_skimage, mode) -> conftest.DataLike:
        arr = self._random_array(arg_shape)
        out = filter_skimage(arr, mode=mode, axis=axis_skimage)

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestSobel(EdgeFilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.Sobel

    @pytest.fixture
    def filter_skimage(self):
        return skimage.filters.sobel


class TestPrewitt(EdgeFilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.Prewitt

    @pytest.fixture
    def filter_skimage(self):
        return skimage.filters.prewitt


class TestScharr(EdgeFilterMixin):
    @pytest.fixture
    def filter_klass(self):
        return pycfilt.Scharr

    @pytest.fixture
    def filter_skimage(self):
        return skimage.filters.scharr


class TestStructureTensor(conftest.DiffMapT):
    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.OpShape:
        ndim = len(arg_shape)
        dim = np.prod(arg_shape).item()
        if ndim > 1:
            codim = (ndim * (ndim + 1) / 2) * dim
        else:
            codim = dim
        return (codim, dim)

    @pytest.fixture
    def mode(self):
        return "constant"

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, arg_shape, mode, filter_kwargs, request) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        ndi, width = request.param
        with pycrt.Precision(width):
            op = pycfilt.StructureTensor(
                arg_shape=arg_shape,
                mode=mode,
                gpu=ndi.name == "CUPY",
                dtype=width.value,
                **filter_kwargs,
            )
        return op, ndi, width

    @pytest.fixture(params=[(8,), (4, 4), (4, 4, 4)])
    def arg_shape(self, request):
        return request.param

    @pytest.fixture
    def diff_method(self):
        return "fd"

    @pytest.fixture
    def smooth_sigma(self):
        return 2.0

    @pytest.fixture
    def smooth_truncate(self):
        return 2.0

    @pytest.fixture
    def diff_params(self):
        return {"diff_type": "central", "accuracy": 1}

    @pytest.fixture
    def filter_kwargs(self, diff_method, smooth_sigma, smooth_truncate, sampling, parallel, diff_params):
        return {
            "diff_method": diff_method,
            "smooth_sigma": smooth_sigma,
            "smooth_truncate": smooth_truncate,
            "sampling": sampling,
            "parallel": parallel,
            "diff_params": diff_params,
        }

    @pytest.fixture(params=[1.5])
    def sampling(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def parallel(self, request):
        return request.param

    @pytest.fixture
    def data_apply(self, arg_shape, mode, filter_kwargs) -> conftest.DataLike:
        arr = self._random_array(arg_shape)

        def _compute_derivatives(arr, filter_kwargs, mode):
            directions = np.arange(arr.ndim)
            x_np = np.pad(arr, ((1, 1),) * len(arg_shape), mode=mode)
            slices = (slice(None, None),) + (slice(1, -1, None),) * len(arg_shape)
            out = np.gradient(x_np, filter_kwargs["sampling"], edge_order=2, axis=directions)
            if len(directions) == 1:
                out = [
                    out,
                ]
            return np.stack(out)[slices]

        # Cannot use skimage structure tensor as it uses the sobel filter as derivative.
        derivatives = _compute_derivatives(arr, filter_kwargs, mode)
        out = np.concatenate(
            [
                skimage.filters.gaussian(
                    der0 * der1,
                    sigma=filter_kwargs["smooth_sigma"],
                    mode=mode,
                    truncate=filter_kwargs["smooth_truncate"],
                )
                for der0, der1 in itertools.combinations_with_replacement(derivatives, 2)
            ]
        )

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
