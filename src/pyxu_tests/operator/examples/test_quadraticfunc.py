import itertools

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


def Q(dim: int) -> pxa.PosDefOp:
    import pyxu_tests.operator.examples.test_posdefop as tc_p

    return tc_p.PSDConvolution(N=dim)


def c(dim: int) -> pxa.LinFunc:
    import pyxu_tests.operator.examples.test_linfunc as tc_l

    return tc_l.ScaledSum(N=dim)


def t() -> float:
    return 1


class TestQuadraticFunc(conftest.QuadraticFuncT):
    from pyxu.operator.linop import IdentityOp, NullFunc

    @pytest.fixture(
        params=itertools.product(
            (
                # dim, Q[init], c[init], t[init], Q[ground-truth], c[ground-truth], t[ground-truth]
                (5, None, None, 0, IdentityOp(5), NullFunc(5), 0),
                (5, Q(5), c(5), t(), Q(5), c(5), t()),
            ),
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def Q_param(self, _spec) -> pxt.OpT:
        return _spec[0][4]

    @pytest.fixture
    def c_param(self, _spec) -> pxt.OpT:
        return _spec[0][5]

    @pytest.fixture
    def t_param(self, _spec) -> pxt.Real:
        return _spec[0][6]

    @pytest.fixture
    def spec(self, _spec):
        op = pxa.QuadraticFunc(
            shape=(1, _spec[0][0]),
            Q=_spec[0][1],
            c=_spec[0][2],
            t=_spec[0][3],
        )
        return op, _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec) -> int:
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_apply(self, dim, Q_param, c_param, t_param):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = 0.5 * (arr @ Q_param.apply(arr)) + c_param.apply(arr) + t_param
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_grad(self, dim, Q_param, c_param):
        dim = self._sanitize(dim, 3)
        arr = self._random_array((dim,))
        out = Q_param.apply(arr) + c_param.asarray().reshape(-1)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def data_prox(self, dim, Q_param, c_param):
        dim = self._sanitize(dim, 3)
        arr, tau = self._random_array((dim,)), 2
        out, *_ = splinalg.lstsq(
            Q_param.asarray() + np.eye(dim) / tau,
            arr / tau - c_param.asarray().reshape(-1),
        )
        return dict(
            in_=dict(
                arr=arr,
                tau=tau,
            ),
            out=out,
        )
