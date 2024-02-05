import itertools

import pytest
import scipy.signal as signal

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest


class TestCZT(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # Specification:
            # [0] dim_shape, axes, M, A, W  (user-specified)
            # [1] dim_shape, axes, M, A, W  (canonical)
            # 1D transforms ---------------------------------------------------
            [
                (5, None, 6, 1, 1j),
                ((5,), (0,), (6,), (1,), (1j,)),
            ],
            [
                (6, None, 5, -1, 1j),
                ((6,), (0,), (5,), (-1,), (1j,)),
            ],
            # 2D transforms ---------------------------------------------------
            [
                ((6, 5), None, 7, -1, -1j),
                ((6, 5), (0, 1), (7, 7), (-1, -1), (-1j, -1j)),
            ],
            [
                ((6, 5), None, (7, 6), (-1, 1j), (-1j, 1j)),
                ((6, 5), (0, 1), (7, 6), (-1, 1j), (-1j, 1j)),
            ],
            [
                ((6, 5), 0, 7, -1, 1j),
                ((6, 5), (0,), (7,), (-1,), (1j,)),
            ],
            [
                ((6, 5), -1, 7, -1, 1j),
                ((6, 5), (1,), (7,), (-1,), (1j,)),
            ],
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
        dim_shape, axes, M, A, W = _spec[0]  # user-specified form
        op = pxo.CZT(
            dim_shape=dim_shape,
            axes=axes,
            M=M,
            A=A,
            W=W,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        dim_shape, _, _, _, _ = _spec[1]
        return (*dim_shape, 2)

    @pytest.fixture
    def codim_shape(self, dim_shape, axes, M) -> pxt.NDArrayShape:
        sh = list(dim_shape[:-1])
        for _M, ax in zip(M, axes):
            sh[ax] = _M
        return (*sh, 2)

    @pytest.fixture
    def axes(self, _spec) -> pxt.NDArrayAxis:
        _, axes, _, _, _ = _spec[1]
        return axes

    @pytest.fixture
    def M(self, _spec) -> tuple[int]:
        _, _, M, _, _ = _spec[1]
        return M

    @pytest.fixture
    def A(self, _spec) -> tuple[complex]:
        _, _, _, A, _ = _spec[1]
        return A

    @pytest.fixture
    def W(self, _spec) -> tuple[complex]:
        _, _, _, _, W = _spec[1]
        return W

    @pytest.fixture
    def data_apply(self, dim_shape, axes, M, A, W) -> conftest.DataLike:
        sh = dim_shape[:-1]  # complex-valued inputs have this shape

        x = self._random_array(sh) + 1j * self._random_array(sh)
        y = x.copy()
        for _M, _A, _W, ax in zip(M, A, W, axes):
            y = signal.czt(
                x=y,
                m=_M,
                w=_W,
                a=_A,
                axis=ax,
            )
        y = y.copy(order="C")  # guarantee C-order

        return dict(
            in_=dict(arr=pxu.view_as_real(x)),
            out=pxu.view_as_real(y),
        )

    # Tests -------------------------------------------------------------------
    def test_value_asarray(self, op, _op_array, ndi):
        # Fixture[_op_array] is computed by calling .apply() on canonical sequences e_{k}.
        # These calls are done at all (precision, backend) pairs given in Fixture[spec].
        # If backend=DASK, CZT.apply() uses the chunked-FFT method to evaluate the CZT in chunks.
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
