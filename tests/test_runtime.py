import numpy as np
import pytest

import pycsou.runtime as pycrt


class TestPrecision:
    @pytest.mark.parametrize("w", pycrt.Width)
    def test_contextManager(self, w: pycrt.Width):
        with pycrt.Precision(w):
            assert pycrt.getPrecision() == w


class TestEnforcePrecision:
    def right_dtype(self, x):
        return x.dtype == pycrt.getPrecision().value

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
        @pycrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x + 1

        with pycrt.Precision(pycrt.Width.HALF):
            assert self.right_dtype(f(value))

    @pytest.mark.parametrize(
        "value",
        [
            1j,  # Python complex
            np.complex64(2),  # NumPy complex dtype
        ],
    )
    def test_invalid_scalar_io(self, value):
        @pycrt.enforce_precision("xx")
        def f(xx):  # explicitly test multi-character strings
            assert self.right_dtype(xx)
            return xx + 1

        with pytest.raises(TypeError):
            f(value)

    def test_multi_i(self):
        @pycrt.enforce_precision(["x", "y"])
        def f(x, y, z):
            assert self.right_dtype(x)
            assert self.right_dtype(y)
            return x + y

        with pycrt.Precision(pycrt.Width.HALF):
            f(1, 1.0, True)

    def test_valid_name_i(self):
        @pycrt.enforce_precision("y")
        def f(x):
            return x

        with pytest.raises(ValueError):
            f(1)

    def test_valid_array_io(self, xp):
        @pycrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x

        with pycrt.Precision(pycrt.Width.HALF):
            x = xp.arange(5)
            assert self.right_dtype(f(x))

    def test_invalid_array_io(self, xp):
        @pycrt.enforce_precision("x")
        def f(x):
            assert self.right_dtype(x)
            return x

        with pycrt.Precision(pycrt.Width.HALF):
            x = xp.arange(5) + 1j * xp.arange(5)
            with pytest.raises(TypeError):
                f(x)


class TestCoerce:
    def right_dtype(self, x):
        return x.dtype == pycrt.getPrecision().value

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
        with pycrt.Precision(pycrt.Width.HALF):
            assert self.right_dtype(pycrt.coerce(value))

    @pytest.mark.parametrize(
        "value",
        [
            1j,  # Python complex
            np.complex64(2),  # NumPy complex dtype
        ],
    )
    def test_invalid_scalar(self, value):
        with pytest.raises(TypeError):
            pycrt.coerce(value)

    def test_valid_array(self, xp):
        with pycrt.Precision(pycrt.Width.HALF):
            x = xp.arange(5)
            assert self.right_dtype(pycrt.coerce(x))

    def test_invalid_array(self, xp):
        with pycrt.Precision(pycrt.Width.HALF):
            x = xp.arange(5) + 1j * xp.arange(5)
            with pytest.raises(TypeError):
                pycrt.coerce(x)
