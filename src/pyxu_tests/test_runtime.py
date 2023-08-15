import numpy as np
import pytest

import pyxu.runtime as pxrt


class TestPrecisionContextManager:
    @pytest.mark.parametrize("w", pxrt.Width)
    def test_contextManager(self, w: pxrt.Width):
        with pxrt.Precision(w):
            assert pxrt.getPrecision() == w


class TestEnforcePrecisionContextManager:
    @pytest.mark.parametrize("state", [True, False])
    def test_contextManager(self, state: bool):
        assert pxrt.getCoerceState() is True  # default
        with pxrt.EnforcePrecision(state):
            assert pxrt.getCoerceState() == state


class TestEnforcePrecisionDecorator:
    def right_dtype(self, x):
        return x.dtype == pxrt.getPrecision().value

    def test_CM_noOp(self):
        @pxrt.enforce_precision("x")
        def f(x):
            return x

        with pxrt.EnforcePrecision(False):
            x = 1
            assert x is f(x)

    @pytest.mark.parametrize(
        "value",
        [
            0.1,  # Python float
            1,  # Python int
            np.float128(1),  # NumPy dtypes
            np.int8(-1),
            np.uint8(1),
            True,  # Python T/F
            False,
        ],
    )
    def test_valid_scalar_io(self, value):
        @pxrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x + 1

        with pxrt.Precision(pxrt.Width.SINGLE):
            assert self.right_dtype(f(value))

    @pytest.mark.parametrize(
        "value",
        [
            1j,  # Python complex
            np.complex64(2),  # NumPy complex dtype
        ],
    )
    def test_invalid_scalar_io(self, value):
        @pxrt.enforce_precision("xx")
        def f(xx):  # explicitly test multi-character strings
            assert self.right_dtype(xx)
            return xx + 1

        with pytest.raises(TypeError):
            f(value)

    def test_multi_i(self):
        @pxrt.enforce_precision(["x", "y"])
        def f(x, y, z):
            assert self.right_dtype(x)
            assert self.right_dtype(y)
            return x + y

        with pxrt.Precision(pxrt.Width.SINGLE):
            f(1, 1.0, True)

    def test_valid_name_i(self):
        @pxrt.enforce_precision("y")
        def f(x):
            return x

        with pytest.raises(ValueError):
            f(1)

    def test_valid_array_io(self, xp):
        @pxrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x

        with pxrt.Precision(pxrt.Width.SINGLE):
            x = xp.arange(5)
            assert self.right_dtype(f(x))

    def test_invalid_array_io(self, xp):
        @pxrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x

        with pxrt.Precision(pxrt.Width.SINGLE):
            x = xp.arange(5) + 1j * xp.arange(5)
            with pytest.raises(TypeError):
                f(x)

    @pytest.mark.parametrize("allow_None", [True, False])
    def test_None_input(self, allow_None):
        @pxrt.enforce_precision("x", allow_None=allow_None)
        def f(x):
            return x

        if allow_None:
            assert f(None) is None
        else:
            with pytest.raises(ValueError):
                f(None)


class TestCoerce:
    def right_dtype(self, x):
        return x.dtype == pxrt.getPrecision().value

    @pytest.mark.parametrize(
        "value",
        [
            0.1,  # Python float
            1,  # Python int
            np.float128(1),  # NumPy dtypes
            np.int8(-1),
            np.uint8(1),
            True,  # Python T/F
            False,
        ],
    )
    def test_valid_scalar(self, value):
        with pxrt.Precision(pxrt.Width.SINGLE):
            assert self.right_dtype(pxrt.coerce(value))

    @pytest.mark.parametrize(
        "value",
        [
            1j,  # Python complex
            np.complex64(2),  # NumPy complex dtype
        ],
    )
    def test_invalid_scalar(self, value):
        with pytest.raises(TypeError):
            pxrt.coerce(value)

    def test_valid_array(self, xp):
        with pxrt.Precision(pxrt.Width.SINGLE):
            x = xp.arange(5)
            assert self.right_dtype(pxrt.coerce(x))

    def test_invalid_array(self, xp):
        with pxrt.Precision(pxrt.Width.SINGLE):
            x = xp.arange(5) + 1j * xp.arange(5)
            with pytest.raises(TypeError):
                pxrt.coerce(x)
