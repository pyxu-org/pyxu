import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class CabinetProjection(pyca.ProjOp):
    # https://en.wikipedia.org/wiki/Oblique_projection#Cabinet_projection
    def __init__(self, angle: float):
        super().__init__(shape=(3, 3))
        self._A = np.array(
            [
                [1, 0, np.cos(angle) / 2],
                [0, 1, np.sin(angle) / 2],
                [0, 0, 0],
            ]
        )

    @staticmethod
    def _bcast_apply(A, b):
        xp = pycu.get_array_module(A)
        A = xp.array(A, dtype=b.dtype)
        y = xp.tensordot(A, b, axes=[[1], [-1]])
        return xp.moveaxis(y, 0, -1)

    @pycrt.enforce_precision("arr")
    def apply(self, arr):
        return self._bcast_apply(self._A, arr)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr):
        return self._bcast_apply(self._A.T, arr)


class TestCabinetProjection(conftest.ProjOpT):
    @pytest.fixture
    def angle(self):
        return np.pi / 4

    @pytest.fixture
    def op(self, angle):
        return CabinetProjection(angle)

    @pytest.fixture
    def data_shape(self):
        return (3, 3)

    @pytest.fixture
    def data_apply(self, angle):
        x = self._random_array((3,))
        y = np.array(
            [
                x[0] + 0.5 * np.cos(angle) * x[2],
                x[1] + 0.5 * np.sin(angle) * x[2],
                0,
            ]
        )
        return dict(
            in_=dict(arr=x),
            out=y,
        )
