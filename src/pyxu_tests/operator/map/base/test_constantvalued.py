import collections.abc as cabc
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ConstantValueMixin:
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, codim_shape, cst, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.ConstantValued(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
            cst=cst,
        )
        return op, ndi, width

    @pytest.fixture(
        params=[
            -2.1,
            -1,
            0,
            1,
            2.1,
        ]
    )
    def cst(self, request) -> pxt.Real:
        return request.param

    @pytest.fixture
    def data_apply(self, dim_shape, codim_shape, cst) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.full(codim_shape, fill_value=cst)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture
    def data_grad(self, dim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.zeros(dim_shape)

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim_shape) -> cabc.Collection[np.ndarray]:
        N_test = 10
        x = self._random_array(shape=(N_test, *dim_shape))
        return x

    @pytest.fixture
    def data_prox(self, dim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        t = np.fabs(x).min() + 1e-2  # some small positive offset
        y = x.copy()

        return dict(
            in_=dict(
                arr=x,
                tau=t,
            ),
            out=y,
        )


class TestConstantValuedDiffMap(ConstantValueMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def codim_shape(self, request) -> pxt.NDArrayShape:
        return request.param


class TestConstantValuedProxDiffFunc(ConstantValueMixin, conftest.ProxDiffFuncT):
    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param
