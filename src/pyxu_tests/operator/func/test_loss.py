import itertools

import numpy as np
import numpy.random as npr
import pytest

import pyxu.operator.func as pxof
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu_tests.operator.conftest as conftest


class TestKLDivergence(conftest.ProxFuncT):
    @pytest.fixture(
        params=itertools.product(
            # dim
            (5,),
            list(pxd.NDArrayInfo),
            pxrt.Width,
        )
    )
    def _spec(self, request):
        dim = request.param[0]
        ndi = request.param[1]
        dtype = request.param[2].value.name
        data = np.arange(1, dim + 1, dtype=dtype)
        if ndi is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
            op = None
        else:
            xp = ndi.module()
            op = pxof.KLDivergence(dim, xp.array(data))
        return (dim, op), ndi, request.param[2], data

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def kl_data(self, _spec):
        return _spec[3]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture(params=[np.zeros((5,)) + 1e-6, np.arange(1, 6)])  # 2 evaluation points
    def data_apply(self, request, kl_data):
        x = request.param
        y = pxu.to_NUMPY(kl_data)

        H = np.ones_like(x) * np.inf
        ids_1 = (x > 0) * (y > 0)
        ids_2 = (x == 0) * (y > 0)
        H[ids_1] = y[ids_1] * np.log(y[ids_1] / x[ids_1])
        H[ids_2] = 0
        out = np.sum(H - y + x)
        return dict(
            in_=dict(arr=x),
            out=out,
        )

    @pytest.fixture(
        params=[  # 2 evaluation points
            dict(
                arr=np.zeros((5,)) + 1e-6,
                tau=1,
            ),
            dict(
                arr=np.arange(1, 6),
                tau=1,
            ),
        ]
    )
    def data_prox(self, request, kl_data):
        # ground truth prox taken from Pyxu v1.
        in_ = request.param
        arr = in_["arr"]
        tau = in_["tau"]
        out = (arr - tau + np.sqrt((arr - tau) ** 2 + 4 * tau * kl_data)) / 2
        return dict(in_=in_, out=out)

    @staticmethod
    def _random_array(
        shape: pxt.NDArrayShape,
        seed: int = 0,
        xp: pxt.ArrayModule = pxd.NDArrayInfo.NUMPY.module(),
        width: pxrt.Width = pxrt.Width.DOUBLE,
    ):
        # Create only arrays with only positive-valued entries.
        rng = npr.default_rng(seed)
        x = rng.normal(size=shape)
        return xp.abs(xp.array(x, dtype=width.value)) + 1e-6

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test, dim = 10, self._sanitize(dim, 3)
        return self._random_array((N_test, dim))

    @pytest.mark.skip("KLDiv is already a loss.")
    def test_interface_asloss(self, op, xp, width):
        pass
