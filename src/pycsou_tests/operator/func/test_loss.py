import itertools

import numpy as np
import pytest

import pycsou.operator.func as pycof
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class TestKLDivergence(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            # dim
            (5,),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        dim = request.param[0]
        ndi = request.param[1]
        dtype = request.param[2].value.name
        if ndi is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
            op = None
        else:
            xp = ndi.module()
            op = pycof.KLDivergence(dim, xp.arange(dim, dtype=dtype))
        return (dim, op), ndi, request.param[2]

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(arr=np.zeros((5,))),
                out=np.array([-10.0]),
            ),
            dict(
                in_=dict(arr=np.arange(-3, 2)),
                out=np.array([-9.45482256]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                in_=dict(
                    arr=np.zeros((5,)),
                    tau=1,
                ),
                out=np.array([0.0, 0.61803399, 1.0, 1.30277564, 1.56155281]),
            ),
            dict(
                in_=dict(
                    arr=np.arange(-3, 2),
                    tau=1,
                ),
                out=np.array([0.0, 0.30277564, 0.73205081, 1.30277564, 2.0]),
            ),
        ]
    )
    def data_prox(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))
