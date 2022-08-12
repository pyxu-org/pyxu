import itertools

import numpy as np
import pytest
import scipy.linalg as splinalg

import pycsou.abc as pyca
import pycsou.operator.func as pycof
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest


def Q(dim: int) -> pyca.PosDefOp:
    import pycsou_tests.operator.examples.test_posdefop as tc_p

    return tc_p.CDO4(N=dim)


def c(dim: int) -> pyca.LinFunc:
    import pycsou_tests.operator.examples.test_linfunc as tc_l

    return tc_l.ScaledSum(N=dim)


def t() -> float:
    return 1


class TestQuadraticFunc(conftest._QuadraticFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, pycof.QuadraticFunc(Q=Q(5), c=c(5), t=t())),),  # dim, op
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec) -> int:
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_apply(self, dim):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = 0.5 * (arr @ Q(dim).apply(arr)) + c(dim).apply(arr) + t()
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, dim):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = Q(dim).apply(arr) + c(dim).asarray().reshape(-1)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, dim):
        dim = self._sanitize(dim, 3)
        arr, tau = self._random_array((dim,)), 2
        out, *_ = splinalg.lstsq(
            Q(dim).asarray() + np.eye(dim) / tau,
            arr / tau - c(dim).asarray().reshape(-1),
        )
        return dict(
            in_=dict(
                arr=arr,
                tau=tau,
            ),
            out=out,
        )
