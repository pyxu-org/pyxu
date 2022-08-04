import numpy as np
import pytest

import pycsou.operator.linop as pycl
import pycsou.operator.map as pycm
import pycsou_tests.operator.conftest as conftest


class ConstantValueMixin:
    @pytest.fixture(params=[-3.14, 0, 2])
    def cst(self, request) -> float:
        return request.param

    @pytest.fixture
    def data_apply(self, data_shape, cst):
        codim, dim = data_shape
        N_test = 5
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
    @pytest.fixture
    def op(self, data_shape, cst):
        return pycm.ConstantValued(
            shape=data_shape,
            cst=cst,
        )

    @pytest.fixture
    def data_shape(self):
        return (3, 4)

    def test_interface_nullfunc(self, op, cst):
        self._skip_if_disabled()
        if np.isclose(cst, 0):
            assert isinstance(op, pycl.NullOp)


class TestConstantValuedFunc(ConstantValueMixin, conftest.ProxDiffFuncT):
    disable_test = frozenset(
        conftest.ProxDiffFuncT.disable_test
        | {
            "test_math2_grad",  # trivially correct, but raises warning since L=0
        }
    )

    @pytest.fixture
    def op(self, data_shape, cst):
        return pycm.ConstantValued(
            shape=data_shape,
            cst=cst,
        )

    @pytest.fixture
    def data_shape(self):
        return (1, 4)
