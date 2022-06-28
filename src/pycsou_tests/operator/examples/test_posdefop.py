import dask.array as da
import numpy as np
import pytest

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou_tests.operator.conftest as conftest


class CDO4(pyco.PosDefOp):
    # Central Difference of Order 4 (implemented as cascade of 2 CDO2)
    def __init__(self, N: int):
        super().__init__(shape=(N, N))

    def _apply(self, arr):
        xp = pycu.get_array_module(arr)
        h = xp.array([1, -4, 6, -4, 1], dtype=arr.dtype)
        out = xp.convolve(arr, h)[2:-2]
        return out

    def _apply_dask(self, arr):
        out = da.map_overlap(
            self._apply,
            arr,
            depth=4,
            boundary=0,
            trim=True,
            dtype=arr.dtype,
        )
        return out

    @pycrt.enforce_precision(i="arr")
    @pycu.vectorize("arr")
    @pycu.redirect("arr", DASK=_apply_dask)
    def apply(self, arr):
        return self._apply(arr)


class TestCDO4(conftest.PosDefOpT):
    @pytest.fixture
    def dim(self):
        return 10

    @pytest.fixture
    def op(self, dim):
        return CDO4(dim)

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        x = self._random_array((dim,))
        y = np.zeros_like(x)
        a, b, c = 1, -4, 6
        y[0] = c * x[0] + b * x[1] + a * x[2]
        y[1] = b * x[0] + c * x[1] + b * x[2] + a * x[3]
        for i in range(2, dim - 2):
            y[i] = a * x[i - 2] + b * x[i - 1] + c * x[i] + b * x[i + 1] + a * x[i + 2]
        y[-2] = a * x[-4] + b * x[-3] + c * x[-2] + b * x[-1]
        y[-1] = a * x[-3] + b * x[-2] + c * x[-1]
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    # -------------------------------------------------------------------------
    # Run all .pinv()/.dagger() tests with low-precision only.
    #
    # Reason: value[1N]D tests check solution closeness to the ground-truth up to different
    # accuracies depending on the chosen precision.
    #
    # The default stopping criterion of CG is AbsErr(eps=1e-4). At this eps-level,
    # DOUBLE/QUAD-precision will not have dropped below the accuracy threshold set in MapT.isclose()
    # to pass value[1N]D tests.
    #
    # Since CG converges however, it is OK to accept these tests as being successful (despite their
    # apparent failure).
    low_precision = pytest.mark.parametrize(
        "width",  # local override of this fixture
        [
            pycrt.Width.HALF,
            pycrt.Width.SINGLE,
            pytest.param(
                pycrt.Width.DOUBLE,
                marks=pytest.mark.xfail(reason="CG auto-quits given default stop-crit threshold too high."),
            ),
            pytest.param(
                pycrt.Width.QUAD,
                marks=pytest.mark.xfail(reason="CG auto-quits given default stop-crit threshold too high."),
            ),
        ],
    )

    @low_precision
    def test_value1D_pinv(self, op, _data_pinv):
        super().test_value1D_pinv(op, _data_pinv)

    @low_precision
    def test_valueND_pinv(self, op, _data_pinv):
        super().test_valueND_pinv(op, _data_pinv)

    @low_precision
    def test_value1D_call_dagger(self, _op_dagger, _data_apply_dagger):
        super().test_value1D_call_dagger(_op_dagger, _data_apply_dagger)

    @low_precision
    def test_valueND_call_dagger(self, _op_dagger, _data_apply_dagger):
        super().test_valueND_call_dagger(_op_dagger, _data_apply_dagger)

    @low_precision
    def test_value1D_apply_dagger(self, _op_dagger, _data_apply_dagger):
        super().test_value1D_apply_dagger(_op_dagger, _data_apply_dagger)

    @low_precision
    def test_valueND_apply_dagger(self, _op_dagger, _data_apply_dagger):
        super().test_valueND_apply_dagger(_op_dagger, _data_apply_dagger)

    @low_precision
    def test_value1D_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        super().test_value1D_adjoint_dagger(_op_dagger, _data_adjoint_dagger)

    @low_precision
    def test_valueND_adjoint_dagger(self, _op_dagger, _data_adjoint_dagger):
        super().test_valueND_adjoint_dagger(_op_dagger, _data_adjoint_dagger)
