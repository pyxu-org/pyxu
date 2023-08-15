import collections.abc as cabc
import warnings

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu


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
        sh_extra = (2, 1)  # prepend input/output shape by this amount.

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
        in_ = np.arange(6)
        out = np.r_[0, 2, 4] + 1j * np.r_[1, 3, 5]
        return in_, out

    @pytest.fixture(params=[_.value for _ in list(pxrt.CWidth)])
    def no_op_dtype(self, request):
        return request.param


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
        in_ = np.r_[0, 2, 4] + 1j * np.r_[1, 3, 5]
        out = np.arange(6)
        return in_, out

    @pytest.fixture(params=[_.value for _ in list(pxrt.Width)])
    def no_op_dtype(self, request):
        return request.param


class TestViewAsMat:
    @pytest.fixture(
        params=[
            np.reshape(np.r_[:6] + 1j * np.r_[2:8], (2, 3)),
        ]
    )
    def input(self, request) -> np.ndarray:
        # (M,N) NumPy input to test. Will be transformed to different backend/precisions via _input().
        return request.param

    @pytest.fixture(params=pxd.supported_array_modules())
    def xp(self, request) -> pxt.ArrayModule:
        return request.param

    @pytest.fixture(params=pxrt.CWidth)
    def cwidth(self, request) -> pxrt.CWidth:
        return request.param

    @pytest.fixture
    def width(self, cwidth) -> pxrt.Width:
        return cwidth.real

    @pytest.fixture
    def _input(self, input, xp, cwidth) -> pxt.NDArray:
        in_ = xp.array(input, dtype=cwidth.value)
        return in_

    @pytest.fixture(params=[True, False])
    def real_input(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True, False])
    def real_output(self, request) -> bool:
        return request.param

    def test_backend_asmat(self, _input, real_input, real_output):
        out = pxu.view_as_real_mat(
            _input,
            real_input=real_input,
            real_output=real_output,
        )
        assert type(out) == type(_input)  # noqa: E721

    def test_prec_asmat(self, _input, real_input, real_output):
        out = pxu.view_as_real_mat(
            _input,
            real_input=real_input,
            real_output=real_output,
        )
        in_prec = pxrt.CWidth(_input.dtype)
        out_prec = pxrt.Width(out.dtype).complex
        assert in_prec == out_prec

    def test_shape_asmat(self, _input, real_input, real_output):
        if real_input and real_output:
            sh_gt = _input.shape
        elif (not real_input) and real_output:
            sh_gt = (_input.shape[0], 2 * _input.shape[1])
        elif real_input and (not real_output):
            sh_gt = (2 * _input.shape[0], _input.shape[1])
        else:  # (not real_input) and (not real_output)
            sh_gt = (2 * _input.shape[0], 2 * _input.shape[1])

        out = pxu.view_as_real_mat(
            _input,
            real_input=real_input,
            real_output=real_output,
        )
        assert out.shape == sh_gt

    def test_math_asmat(self, xp, cwidth, real_input, real_output):
        # view_as_real(A @ x) = view_as_real_mat(A) @ view_as_real(x)
        rng = np.random.default_rng(seed=3)  # random seed for reproducibility

        M, N = 5, 3
        A = xp.array(rng.normal(size=(M, N)) + 1j * rng.normal(size=(M, N)), dtype=cwidth.value)
        x = xp.array(rng.normal(size=(N,)) + 1j * rng.normal(size=(N,)), dtype=cwidth.value)
        if real_input:
            x = x.real

        lhs = pxu.view_as_real(A @ x)
        if real_output:
            lhs = lhs[::2]

        Ar = pxu.view_as_real_mat(A, real_input=real_input, real_output=real_output)
        rhs = Ar @ pxu.view_as_real(x)
        assert np.allclose(lhs, rhs)

    def test_round_trip(self, _input, real_input, real_output):
        # view_as_complex \compose view_as_real = Id
        x = _input
        y = pxu.view_as_real_mat(
            x,
            real_input=real_input,
            real_output=real_output,
        )
        z = pxu.view_as_complex_mat(
            y,
            real_input=real_input,
            real_output=real_output,
        )

        assert np.allclose(x.real, z.real)
        if real_input and real_output:
            assert np.allclose(z.imag, 0)
        else:
            assert np.allclose(x.imag, z.imag)
