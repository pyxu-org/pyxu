import collections.abc as cabc
import warnings

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct


class ViewAs:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def func(self) -> cabc.Callable:
        raise NotImplementedError

    @pytest.fixture
    def width_in_out(self) -> tuple[pxrt.Width, pxrt.Width]:
        raise NotImplementedError

    @pytest.fixture(
        params=[
            1,
            (),
            None,
            np.array([1])[0],
        ]
    )
    def non_array_input(self, request):
        return request.param

    @pytest.fixture(
        params=[
            np.bool_,
            np.byte,
            np.ubyte,
            np.short,
            np.ushort,
            np.intc,
            np.uintc,
            np.int_,
            np.uint,
            np.longlong,
            np.ulonglong,
        ]
    )
    def unrecognized_dtype(self, request) -> np.dtype:
        return request.param

    @pytest.fixture
    def no_op_dtype(self) -> np.dtype:
        raise NotImplementedError

    @pytest.fixture
    def valid_data(self) -> tuple[np.ndarray, np.ndarray]:  # input -> output
        # No stacking dimensions.
        # Tests needing stacked inputs should augment accordingly.
        raise NotImplementedError

    @pytest.fixture
    def _valid_data(self, valid_data, xp, width_in_out):
        # Same as valid_data(), but with right backend/width.
        return (
            xp.array(valid_data[0], dtype=width_in_out[0].value),
            xp.array(valid_data[1], dtype=width_in_out[1].value),
        )

    # Tests -------------------------------------------------------------------
    def test_fail_non_array_input(self, func, non_array_input):
        with pytest.raises(Exception):
            func(non_array_input)

    def test_fail_unrecognized_dtype(
        self,
        func,
        valid_data,
        unrecognized_dtype,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            array = valid_data[0].astype(unrecognized_dtype)
        with pytest.raises(Exception):
            func(array)

    def test_no_op_dtype(self, func, _valid_data, no_op_dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            x = _valid_data[0].astype(no_op_dtype)
        assert x is func(x)

    def test_value1D(self, func, _valid_data):
        in_ = _valid_data[0]
        out_gt = _valid_data[1]
        out = pxu.compute(func(in_))

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_valueND(self, func, _valid_data):
        sh_extra = (2, 1, 3)  # prepend input/output shape by this amount.

        in_ = _valid_data[0]
        in_ = np.broadcast_to(in_, (*sh_extra, *in_.shape))
        out_gt = _valid_data[1]
        out_gt = np.broadcast_to(out_gt, (*sh_extra, *out_gt.shape))
        out = pxu.compute(func(in_))

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_backend(self, func, _valid_data):
        out = func(_valid_data[0])
        assert type(out) == type(_valid_data[0])  # noqa: E721

    def test_prec(self, func, _valid_data, width_in_out):
        in_ = _valid_data[0]
        out = func(in_)

        w_in = type(width_in_out[0])
        w_out = type(width_in_out[1])

        assert w_in(in_.dtype).name == w_out(out.dtype).name


class TestViewAsComplex(ViewAs):
    @pytest.fixture
    def func(self) -> cabc.Callable:
        return pxu.view_as_complex

    @pytest.fixture(params=[_.name for _ in pxrt.CWidth])
    def width_in_out(self, request):
        w_in = pxrt.Width[request.param]
        w_out = pxrt.CWidth[request.param]
        return w_in, w_out

    @pytest.fixture
    def valid_data(self):
        N = 5
        in_ = np.arange(2 * N).reshape(N, 2)
        out = in_[:, 0] + 1j * in_[:, 1]
        return in_, out

    @pytest.fixture(params=[_.value for _ in list(pxrt.CWidth)])
    def no_op_dtype(self, request):
        return request.param

    # Tests -----------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (5,),
            (5, 1, 4),
            (5, 10, 4),
        ],
    )
    def test_chunk(self, shape, width):
        # DASK-only: verify chunk structure:
        # * unchanged in batch dimensions.
        ndi = pxd.NDArrayInfo.DASK
        xp = ndi.module()

        rng = xp.random.default_rng()
        x = rng.standard_normal(size=(*shape, 2), dtype=width.value)
        x = ct.chunk_array(x, complex_view=True)

        y = pxu.view_as_complex(x)

        assert y.shape == shape
        assert y.dtype == width.complex.value
        assert y.chunks == x.chunks[:-1]
        assert xp.allclose(y.real, x[..., 0]).compute()
        assert xp.allclose(y.imag, x[..., 1]).compute()


class TestViewAsReal(ViewAs):
    @pytest.fixture
    def func(self) -> cabc.Callable:
        return pxu.view_as_real

    @pytest.fixture(params=[_.name for _ in pxrt.CWidth])
    def width_in_out(self, request):
        w_in = pxrt.CWidth[request.param]
        w_out = pxrt.Width[request.param]
        return w_in, w_out

    @pytest.fixture
    def valid_data(self):
        N = 5
        in_ = np.arange(N) + 1j * np.arange(N, 2 * N)
        out = np.stack([np.arange(N), np.arange(N, 2 * N)], axis=-1)
        return in_, out

    @pytest.fixture(params=[_.value for _ in list(pxrt.Width)])
    def no_op_dtype(self, request):
        return request.param

    # Tests -----------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (5,),
            (5, 1, 4),
            (5, 10, 4),
        ],
    )
    def test_chunk(self, shape, width):
        # DASK-only: verify chunk structure:
        # * unchanged in batch dimensions;
        # * no chunks in virtual dimension.
        ndi = pxd.NDArrayInfo.DASK
        xp = ndi.module()

        rng = xp.random.default_rng()
        xR = rng.standard_normal(size=shape, dtype=width.value)
        xI = rng.standard_normal(size=shape, dtype=width.value)
        x = (xR + 1j * xI).astype(width.complex.value)
        x = ct.chunk_array(x, complex_view=False)

        y = pxu.view_as_real(x)

        assert y.shape == (*shape, 2)
        assert y.dtype == width.value
        assert y.chunks[:-1] == x.chunks
        assert y.chunks[-1] == (2,)
        assert xp.allclose(y[..., 0], x.real).compute()
        assert xp.allclose(y[..., 1], x.imag).compute()
