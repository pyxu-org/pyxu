import numpy as np
import pytest
import scipy.linalg as splinalg

import pycsou.abc as pyca
import pycsou.operator.func as pycof
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest
import pycsou_tests.operator.examples.test_linfunc as tc_l
import pycsou_tests.operator.examples.test_posdefop as tc_p


class TestQuadraticFunc(conftest.ProxDiffFuncT):
    @pytest.fixture
    def dim(self) -> int:
        return 5

    @pytest.fixture
    def Q(self, dim) -> pyca.PosDefOp:
        return tc_p.CDO4(N=dim)

    @pytest.fixture
    def c(self, dim) -> pyca.LinFunc:
        return tc_l.ScaledSum(N=dim)

    @pytest.fixture(params=[0, 1])
    def t(self, request) -> pyct.Real:
        return request.param

    @pytest.fixture
    def op(self, Q, c, t):
        return pycof.QuadraticFunc(Q=Q, c=c, t=t)

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 5, self._sanitize(dim, 3)
        return self._random_array((N_test, dim), seed=5)

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test, dim = 5, self._sanitize(dim, 3)
        return self._random_array((N_test, dim), seed=5)

    @pytest.fixture
    def data_apply(self, dim, Q, c, t):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = 0.5 * (arr @ Q.apply(arr)) + c.apply(arr) + t
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, dim, Q, c):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = Q.apply(arr) + c.asarray().reshape(-1)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, dim, Q, c):
        dim = self._sanitize(dim, 3)
        arr, tau = self._random_array((dim,)), 2
        out, *_ = splinalg.lstsq(
            Q.asarray() + np.eye(dim) / tau,
            arr / tau - c.asarray().reshape(-1),
        )
        return dict(
            in_=dict(
                arr=arr,
                tau=tau,
            ),
            out=out,
        )
