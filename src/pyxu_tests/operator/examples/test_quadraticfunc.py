import itertools

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


def Q(dim_shape: pxt.NDArrayShape) -> pxa.PosDefOp:
    import pyxu_tests.operator.examples.test_posdefop as tc_p

    return tc_p.PSDConvolution(dim_shape)


def c(dim_shape: pxt.NDArrayShape) -> pxa.LinFunc:
    import pyxu_tests.operator.examples.test_linfunc as tc_l

    return tc_l.Sum(dim_shape)


def t() -> float:
    return 1


class TestQuadraticFunc(conftest.QuadraticFuncT):
    from pyxu.operator import IdentityOp, NullFunc

    @pytest.fixture(
        params=[
            # [0]    dim_shape,
            # [1--3] Q[init], c[init], t[init],
            # [4--6] Q[ground-truth], c[ground-truth], t[ground-truth]
            ((5,), None, None, 0, IdentityOp(5), NullFunc(5), 0),
            ((5,), Q(5), c(5), t(), Q(5), c(5), t()),
            ((5, 3, 7), None, None, 0, IdentityOp((5, 3, 7)), NullFunc((5, 3, 7)), 0),
            ((5, 3, 7), Q((5, 3, 7)), c((5, 3, 7)), t(), Q((5, 3, 7)), c((5, 3, 7)), t()),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def Q_param(self, _spec) -> pxt.OpT:
        return _spec[4]

    @pytest.fixture
    def c_param(self, _spec) -> pxt.OpT:
        return _spec[5]

    @pytest.fixture
    def t_param(self, _spec) -> pxt.Real:
        return _spec[6]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request):
        ndi, width = request.param
        op = pxa.QuadraticFunc(
            dim_shape=_spec[0],
            codim_shape=1,
            Q=_spec[1],
            c=_spec[2],
            t=_spec[3],
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, Q_param) -> pxt.NDArrayShape:
        return Q_param.dim_shape

    @pytest.fixture(params=[0, 19, 107])
    def data_apply(
        self,
        dim_shape,
        Q_param,
        c_param,
        t_param,
        request,
    ) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = 0.5 * (x * Q_param.apply(x)).sum() + c_param.apply(x) + t_param

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 19, 107])
    def data_grad(
        self,
        dim_shape,
        Q_param,
        c_param,
        request,
    ) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = Q_param.apply(x) + c_param.asarray()[0]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 19, 107])
    def data_prox(
        self,
        dim_shape,
        Q_param,
        c_param,
        request,
    ) -> conftest.DataLike:
        seed = request.param

        x, tau = self._random_array(dim_shape, seed=seed), 2
        N = np.prod(dim_shape)

        A = np.reshape(Q_param.asarray(), (N, N)) + np.eye(N) / tau
        b = np.reshape(x / tau - c_param.asarray(), (N,))

        y, *_ = splinalg.lstsq(A, b)
        y = np.reshape(y, dim_shape)

        return dict(
            in_=dict(
                arr=x,
                tau=tau,
            ),
            out=y,
        )
