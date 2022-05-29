import collections.abc as cabc

import numpy as np
import pytest

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.complex as pycuc


class ViewAs:
    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def func(self) -> cabc.Callable:
        raise NotImplementedError

    @pytest.fixture
    def width_in_out(self) -> tuple["Width", "Width"]:
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

    @pytest.fixture
    def unrecognized_dtype(self) -> np.dtype:
        raise NotImplementedError

    @pytest.fixture
    def valid_data(self) -> tuple[np.ndarray, np.ndarray]:  # input -> output
        # 1D inputs only. Tests needing ND inputs should augment accordingly.
        raise NotImplementedError

    @pytest.fixture
    def _valid_data(self, valid_data, xp, width_in_out):
        return (
            xp.array(valid_data[0], dtype=width_in_out[0].value),
            xp.array(valid_data[1], dtype=width_in_out[1].value),
        )

    # Tests -------------------------------------------------------------------
    def test_fail_non_array_input(self, func, non_array_input):
        with pytest.raises(Exception):
            func(non_array_input)

    def test_fail_unrecognized_dtype(self, func, unrecognized_dtype):
        array = np.arange(50).astype(unrecognized_dtype)
        with pytest.raises(Exception):
            func(array)

    def test_value1D(self, func, _valid_data):
        in_ = _valid_data[0]
        out_gt = _valid_data[1]
        out = pycu.compute(func(in_))

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_valueND(self, func, _valid_data):
        sh_extra = (2, 1)  # prepend input/output shape by this amount.

        in_ = _valid_data[0]
        in_ = np.broadcast_to(in_, (*sh_extra, *in_.shape)).copy()
        out_gt = _valid_data[1]
        out_gt = np.broadcast_to(out_gt, (*sh_extra, *out_gt.shape)).copy()
        out = pycu.compute(func(in_))

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_backend(self, func, _valid_data):
        out = func(_valid_data[0])
        assert type(out) == type(_valid_data[0])

    def test_prec(self, func, _valid_data, width_in_out):
        in_ = _valid_data[0]
        out = func(in_)

        w_in = type(width_in_out[0])
        w_out = type(width_in_out[1])

        assert w_in(in_.dtype).name == w_out(out.dtype).name


class TestViewAsComplex(ViewAs):
    @pytest.fixture
    def func(self) -> cabc.Callable:
        return pycu.view_as_complex

    @pytest.fixture(params=[_.name for _ in pycuc._CWidth])
    def width_in_out(self, request):
        w_in = pycrt.Width[request.param]
        w_out = pycuc._CWidth[request.param]
        return w_in, w_out

    @pytest.fixture(
        params=[
            *[np.uint8, np.int64],  # integer types
            *[_.value for _ in pycuc._CWidth],  # complex-valued types
        ]
    )
    def unrecognized_dtype(self, request) -> np.dtype:
        return request.param

    @pytest.fixture
    def valid_data(self):
        in_ = np.arange(6)
        out = np.r_[0, 2, 4] + 1j * np.r_[1, 3, 5]
        return in_, out


class TestViewAsReal(ViewAs):
    @pytest.fixture
    def func(self) -> cabc.Callable:
        return pycu.view_as_real

    @pytest.fixture(params=[_.name for _ in pycuc._CWidth])
    def width_in_out(self, request):
        w_in = pycuc._CWidth[request.param]
        w_out = pycrt.Width[request.param]
        return w_in, w_out

    @pytest.fixture(
        params=[
            *[np.uint8, np.int64],  # integer types
            *[_.value for _ in pycrt.Width],  # real-valued types
        ]
    )
    def unrecognized_dtype(self, request) -> np.dtype:
        return request.param

    @pytest.fixture
    def valid_data(self):
        in_ = np.r_[0, 2, 4] + 1j * np.r_[1, 3, 5]
        out = np.arange(6)
        return in_, out
