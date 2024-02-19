import functools

import numpy as np
import pytest

import pyxu.math as pxm
import pyxu.util as pxu
import pyxu_tests.conftest as ct


class TestHadamardOuter:
    def test_value(self, x, args, y_gt, complex_input, xp, width):
        # hadamard_outer(x, *args) == y_gt
        # width(x) == width
        # xp(x) == xp(y)
        dtype = width.value if (not complex_input) else width.complex.value
        cast = lambda _: xp.array(_, dtype=dtype)

        _x = cast(x)
        _args = [cast(A) for A in args]
        _y = pxm.hadamard_outer(_x, *_args)

        assert _y.dtype == dtype
        assert xp == pxu.get_array_module(_y)
        assert ct.allclose(y_gt, _y, as_dtype=dtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3, 4])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture(params=[True, False])
    def complex_input(self, request) -> bool:
        return request.param

    @pytest.fixture
    def y_shape(self, space_dim) -> tuple[int]:
        rng = np.random.default_rng()
        sh = rng.integers(low=2, high=15, size=space_dim)
        return tuple(sh)

    @pytest.fixture
    def x(self, y_shape, complex_input) -> np.ndarray:
        rng = np.random.default_rng()
        _x = rng.standard_normal(y_shape)
        if complex_input:
            _x = _x + 1j * rng.standard_normal(y_shape)
        return _x

    @pytest.fixture
    def args(self, y_shape, complex_input) -> list[np.ndarray]:
        rng = np.random.default_rng()
        _args = [rng.standard_normal(size) for size in y_shape]
        if complex_input:
            _args = [A + 1j * rng.standard_normal(A.size) for A in _args]
        return _args

    @pytest.fixture
    def y_gt(self, x, args) -> np.ndarray:
        A = functools.reduce(np.multiply.outer, args)  # (N1,...,ND)
        y = x * A  # (..., N1,...,ND)
        return y
