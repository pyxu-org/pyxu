import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class Rotation(pyco.UnitOp):
    # f: \bR^{N} -> \bR^{N}
    #      x     -> Rz(aZ) Ry(aY) Rx(aX) x
    #               Rk(aK) \in \bR^{N x N} = rotation of `ak[rad]` around axis `k`.
    def __init__(self, ax: float, ay: float, az: float):
        super().__init__(shape=(3, 3))
        self._ax = ax
        self._ay = ay
        self._az = az

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return self._compose(self.Rz @ self.Ry @ self.Rx, arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        return self._compose(self.Rx.T @ self.Ry.T @ self.Rz.T, arr)
        return out

    @staticmethod
    def _compose(R, x):
        xp = pycu.get_array_module(x)
        R = xp.array(R, dtype=x.dtype)
        out = xp.tensordot(R, x, axes=[[1], [-1]])
        return xp.moveaxis(out, 0, -1)

    @property
    def Rx(self) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(self._ax), -np.sin(self._ax)],
                [0, np.sin(self._ax), np.cos(self._ax)],
            ]
        )

    @property
    def Ry(self) -> np.ndarray:
        return np.array(
            [
                [np.cos(self._ay), 0, np.sin(self._ay)],
                [0, 1, 0],
                [-np.sin(self._ay), 0, np.cos(self._ay)],
            ]
        )

    @property
    def Rz(self) -> np.ndarray:
        return np.array(
            [
                [np.cos(self._az), -np.sin(self._az), 0],
                [np.sin(self._az), np.cos(self._az), 0],
                [0, 0, 1],
            ]
        )


class TestRotation(conftest.UnitOpT):
    @pytest.fixture
    def angleZ(self):  # rotation around Z-axis
        return np.pi / 3

    @pytest.fixture
    def op(self, angleZ):
        return Rotation(ax=0, ay=0, az=angleZ)

    @pytest.fixture
    def data_shape(self):
        return (3, 3)

    @pytest.fixture
    def data_apply(self, angleZ):
        x = np.r_[1, 0, 0.5]
        c, s = np.cos(angleZ), np.sin(angleZ)
        y = np.r_[
            c * x[0] - s * x[1],
            s * x[0] + c * x[1],
            x[2],
        ]

        return dict(
            in_=dict(arr=x),
            out=y,
        )
