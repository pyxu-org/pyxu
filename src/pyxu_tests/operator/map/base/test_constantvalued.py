import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class ConstantValueMixin:
    @pytest.fixture
    def _spec(self, request):
        # (cst, shape, backend, width) config
        raise NotImplementedError

    @pytest.fixture
    def spec(self, _spec):
        op = pxo.ConstantValued(shape=_spec[1], cst=_spec[0])
        ndi, width = _spec[2:]
        return op, ndi, width

    @pytest.fixture
    def cst(self, _spec) -> float:
        return _spec[0]

    @pytest.fixture
    def data_shape(self, _spec) -> pxt.OpShape:
        return _spec[1]

    @pytest.fixture
    def data_apply(self, data_shape, cst):
        codim, dim = data_shape
        return dict(
            in_=dict(arr=np.zeros(dim)),
            out=np.full(codim, fill_value=cst),
        )

    @pytest.fixture
    def data_grad(self, data_shape):
        _, dim = data_shape
        arr = conftest.MapT._random_array((dim,))
        return dict(
            in_=dict(arr=arr),
            out=np.zeros(dim),
        )

    @pytest.fixture
    def data_prox(self, data_shape):
        _, dim = data_shape
        arr = conftest.MapT._random_array((dim,))
        return dict(
            in_=dict(arr=arr, tau=2.1),
            out=arr,
        )

    @pytest.fixture
    def data_math_lipschitz(self, data_shape):
        _, dim = data_shape
        N_test = 5
        return conftest.MapT._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, data_shape):
        _, dim = data_shape
        N_test = 5
        return conftest.MapT._random_array((N_test, dim))


class TestConstantValuedMap(ConstantValueMixin, conftest.DiffMapT):
    @pytest.fixture(
        params=itertools.product(
            [-3.14, 0, 2],  # cst
            [(2, 4), (3, 1)],  # shape
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param


class TestConstantValuedFunc(ConstantValueMixin, conftest.ProxDiffFuncT):
    disable_test = frozenset(
        conftest.ProxDiffFuncT.disable_test
        | {
            "test_interface_asloss",  # does not make sense for ConstantValued().
        }
    )

    @pytest.fixture(
        params=itertools.product(
            [-3.14, 0, 2],  # cst
            [(1, 4), (1, 1)],  # shape
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def _spec(self, request):
        return request.param
