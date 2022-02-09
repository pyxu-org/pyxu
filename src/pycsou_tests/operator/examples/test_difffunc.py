import numpy as np
import numpy.random as npr
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class SquaredL2Norm(pyco.DiffFunc):
    # f: \bR^{M} -> \bR
    #      x     -> \norm{x}{2}^{2}
    def __init__(self, M: int = None):
        super().__init__(shape=(1, M))

    def lipschitz(self, **kwargs):
        return np.inf

    def diff_lipschitz(self, **kwargs):
        return 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.linalg.norm(arr, axis=-1, keepdims=True)
        y2 = xp.power(y, 2, dtype=arr.dtype)
        return y2

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr):
        return 2 * arr


class TestSquaredL2Norm(conftest.DiffFuncT):
    @pytest.fixture(params=[4, None])
    def dim(self, request):
        return request.param

    @pytest.fixture
    def op(self, dim):
        return SquaredL2Norm(M=dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_lipschitz(self):
        return dict(
            in_=dict(),
            out=np.inf,
        )

    @pytest.fixture
    def data_diff_lipschitz(self):
        return dict(
            in_=dict(),
            out=2,
        )

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((3,))),
                out=np.zeros((1,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 3)),
                out=np.array([19]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        rng, N_test = npr.default_rng(seed=5), 6
        if dim is None:
            dim = 3
        return rng.normal(size=(N_test, dim))  # 6 test points

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        rng, N_test = npr.default_rng(seed=6), 6
        if dim is None:
            dim = 3
        return rng.normal(size=(N_test, dim))  # 6 test points

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((3,))),
                out=np.zeros((3,)),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 3)),
                out=2 * np.arange(-3, 3),
            ),
        ]
    )
    def data_grad(self, request):
        return request.param
