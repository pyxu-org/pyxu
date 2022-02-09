import numpy as np
import numpy.random as npr
import pytest

import pycsou.abc.operator as pyco
import pycsou.linop.base as pyclb
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Sin(pyco.DiffMap):
    # f: \bR^{M} -> \bR^{M}
    #      x     -> sin(x)
    def __init__(self, M: int):
        super().__init__(shape=(M, M))

    def lipschitz(self, **kwargs):
        return 1

    def diff_lipschitz(self, **kwargs):
        return 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.sin(arr)
        return y

    def jacobian(self, arr):
        xp = pycu.get_array_module(arr)
        J = xp.diag(xp.cos(arr))
        return pyclb.ExplicitLinOp(J)


class TestSin(conftest.DiffMapT):
    @pytest.fixture
    def dim(self):
        return 4

    @pytest.fixture
    def op(self, dim):
        return Sin(M=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_lipschitz(self):
        return dict(
            in_=dict(),
            out=1,
        )

    @pytest.fixture
    def data_diff_lipschitz(self):
        return dict(
            in_=dict(),
            out=1,
        )

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, np.pi / 2, dim)
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        rng, N_test = npr.default_rng(seed=3), 5
        return rng.normal(size=(N_test, dim))  # 5 test points

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        rng, N_test = npr.default_rng(seed=4), 5
        return rng.normal(size=(N_test, dim))  # 5 test points
