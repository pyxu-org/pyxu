import collections.abc as cabc
import itertools

import numpy as np
import pytest
import scipy.special as sp

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest
from pyxu_tests.conftest import chunk_array


class TestKLDivergence(conftest.ProxFuncT):
    disable_test = conftest.ProxFuncT.disable_test | {
        # ---------------------------------------------------------------------
        # KLDivergence() has strict chunk-size rules, and test_math_lipschitz()
        # is not built to handle this constraint. We know however that L=\inf
        # for KLDivergence(), so this test is meaningless anyway.
        "test_math_lipschitz",
        # ---------------------------------------------------------------------
    }

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, kl_data, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        data = chunk_array(
            xp.array(
                kl_data,
                dtype=width.value,
            ),
            complex_view=False,
        )

        op = pxo.KLDivergence(data=data)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def kl_data(self, request) -> np.ndarray:
        # The `data` parameter to KLDivergence
        dim_shape = request.param

        x = np.fabs(self._random_array(dim_shape))
        return x

    @pytest.fixture
    def dim_shape(self, kl_data) -> pxt.NDArrayShape:
        return kl_data.shape

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, kl_data, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        y = sp.kl_div(kl_data, x).sum()[np.newaxis]

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture(params=[0, 17, 93])
    def data_prox(self, dim_shape, kl_data, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = 0.5 * ((x - t) + np.sqrt((x - t) ** 2 + (4 * kl_data * t)))

        return dict(
            in_=dict(
                arr=x,
                tau=t,
            ),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        x = np.fabs(x)
        return x
