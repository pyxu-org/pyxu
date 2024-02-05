import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest


class TestFFT(conftest.NormalOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     axes (user-specified),
            #     axes (canonical),
            # 1D transforms ---------------------------------------------------
            (7, (7,), None, (0,)),
            (7, (7,), 0, (0,)),
            (7, (7,), -1, (0,)),
            # 2D transforms ---------------------------------------------------
            ((7, 8), (7, 8), None, (0, 1)),
            ((7, 8), (7, 8), 0, (0,)),
            ((7, 8), (7, 8), 1, (1,)),
            ((7, 8), (7, 8), (-1, 0), (0, 1)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.CWidth,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        dim_shape, axes = _spec[0], _spec[2]  # user-specified form
        op = pxo.FFT(
            dim_shape=dim_shape,
            axes=axes,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        sh = _spec[1]
        return (*sh, 2)

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture
    def axes(self, _spec) -> pxt.NDArrayAxis:
        return _spec[3]

    @pytest.fixture
    def data_apply(
        self,
        dim_shape,
        axes,
    ) -> conftest.DataLike:
        sh = dim_shape[:-1]  # complex-valued inputs have this shape

        x = self._random_array(sh) + 1j * self._random_array(sh)
        y = np.fft.fftn(x, axes=axes).copy(order="C")  # guarantee C-order

        return dict(
            in_=dict(arr=pxu.view_as_real(x)),
            out=pxu.view_as_real(y),
        )

    # Tests -------------------------------------------------------------------
    def test_value_asarray(self, op, _op_array, ndi):
        # Fixture[_op_array] is computed by calling .apply() on canonical sequences e_{k}.
        # These calls are done at all (precision, backend) pairs given in Fixture[spec].
        # If backend=DASK, FFT.apply() uses the CZT method to evaluate the FFT in chunks.
        # Some FP accumulation error is inevitable here, hence Fixture[_op_array] is NOT
        # the ultimate ground-truth it is supposed to denote.
        # Due to the above this test may fail (by a razor's edge) in the DASK-case.
        self._skip_if_disabled()
        ct.flaky(
            func=super().test_value_asarray,
            args=dict(
                op=op,
                _op_array=_op_array,
            ),
            condition=ndi == pxd.NDArrayInfo.DASK,
            reason="asarray() ground-truth [_op_array()] is approximate in DASK case.",
        )
