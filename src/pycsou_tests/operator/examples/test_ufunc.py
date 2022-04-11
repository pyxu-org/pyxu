import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.operator.linop.base as pyclb
import pycsou_tests.operator.conftest as conftest


# Trigonometric Functions

class Sin(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Sin, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.sin(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(xp.cos(arr)))

class TestSin(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Sin(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2*np.pi, data_shape[0])
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))


class Cos(pyca.DiffMap):
    def __init__(self, shape: pyct.Shape):
        super(Cos, self).__init__(shape)
        self._lipschitz = self._diff_lipschitz = 1

    @pycrt.enforce_precision(i='arr', o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.cos(arr)

    def jacobian(self, arr: pyct.NDArray):
        xp = pycu.get_array_module(arr)
        return pyclb.ExplicitLinOp(xp.diag(-xp.sin(arr)))

class TestCos(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 40

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def op(self, data_shape):
        return Cos(shape=data_shape)

    @pytest.fixture
    def data_apply(self, data_shape):
        A = np.linspace(0, 2*np.pi, data_shape[0])
        B = np.cos(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        N_test = 5
        return self._random_array((N_test, data_shape[0]))

