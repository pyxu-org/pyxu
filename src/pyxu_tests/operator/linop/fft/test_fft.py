import itertools
import typing as typ

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class TestFFT(conftest.LinOpT):
    # Internal Helpers ----------------------------------------------------------------------------
    @staticmethod
    def spec_data() -> (
        list[
            tuple[
                pxd.NDArrayInfo,
                pxrt.Width,
                typ.Any,  # (arg_shape, axes), user-provided to FFT.__init__()
                typ.Any,  # (arg_shape, axes), ground-truth it corresponds to. (Canonical form.)
                dict,  # kwargs
            ]
        ]
    ):
        N = pxd.NDArrayInfo
        W = pxrt.Width
        data = []

        # (arg_shape, axes) configs to test
        # * `spec[k][0]` corresponds to raw inputs users provide to FFT().
        # * `spec[k][1]` corresponds to their ground-truth canonical parameterization.
        spec = [
            # 1D transforms ---------------------------------------------------
            (
                (5, None),
                ((5,), (0,)),
            ),
            (
                ((5,), None),
                ((5,), (0,)),
            ),
            (
                (5, 0),
                ((5,), (0,)),
            ),
            # ND transforms ---------------------------------------------------
            (
                ((5, 3, 4), None),
                ((5, 3, 4), (0, 1, 2)),
            ),
            (
                ((5, 3, 4), 0),
                ((5, 3, 4), (0,)),
            ),
            (
                ((5, 3, 4), 1),
                ((5, 3, 4), (1,)),
            ),
            (
                ((5, 3, 4), 2),
                ((5, 3, 4), (2,)),
            ),
            (
                ((5, 3, 4), (0, 1)),  # transform axes sequential
                ((5, 3, 4), (0, 1)),
            ),
            (
                ((5, 3, 4), (0, 2)),  # transform axes non-sequential
                ((5, 3, 4), (0, 2)),
            ),
        ]

        # Test all backend/width combos, with no kwargs -----------------------
        for ndi, width, (init_spec, canonical_spec) in itertools.product(N, W, spec):
            data.append(
                (
                    ndi,
                    width,
                    init_spec,
                    canonical_spec,
                    dict(),  # no kwargs
                )
            )

        # A specific NUMPY/DASK spec, with all kwargs combos ------------------
        kwargs = [
            dict(
                workers=workers,
                auto_align_input=auto_align_input,
                auto_contiguous=auto_contiguous,
            )
            for (workers, auto_align_input, auto_contiguous) in itertools.product(
                [1, 2],  # workers
                [True, False],  # auto_align_input
                [True, False],  # auto_contiguous
            )
        ]
        for ndi, width, (init_spec, canonical_spec), _kwargs in itertools.product(
            [N.NUMPY, N.DASK],
            W,
            [spec[5]],  # ND input, mono-axes transform
            kwargs,
        ):
            data.append(
                (
                    ndi,
                    width,
                    init_spec,
                    canonical_spec,
                    _kwargs,
                )
            )

        return data

    @pytest.fixture(params=spec_data())
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):  # canonical representation
        arg_shape = _spec[3][0]
        return arg_shape

    @pytest.fixture
    def axes(self, _spec):  # canonical representation
        axes = _spec[3][1]
        return axes

    @pytest.fixture(params=[False, True])
    def transform_real(self, request) -> bool:
        return request.param

    @pytest.fixture
    def spec(self, _spec, transform_real) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width, (arg_shape, axes), _, kwargs = _spec  # user-provided (arg_shape, axes).

        op = pxl.FFT(
            arg_shape=arg_shape,
            axes=axes,
            real=transform_real,
            **kwargs,
        )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape, transform_real) -> pxt.OpShape:
        codim = dim = 2 * np.prod(arg_shape)
        if transform_real:
            dim /= 2
        return (codim, dim)

    @pytest.fixture
    def data_apply(self, arg_shape, axes, transform_real) -> conftest.DataLike:
        arr_r, arr_c = self._random_array((2, *arg_shape), seed=26)  # fixed seed for reproducibility
        if transform_real:
            arr = arr_r
            arr_gt = arr.reshape(-1)
        else:
            arr = arr_r + 1j * arr_c
            arr_gt = pxu.view_as_real(arr.reshape(-1))

        out = np.fft.fftn(arr, axes=axes, norm="backward")
        out_gt = pxu.view_as_real(out.reshape(-1))

        return dict(
            in_=dict(arr=arr_gt),
            out=out_gt,
        )

    # Overridden Tests --------------------------------------------------------
    def test_value_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_value_to_sciop(_op_sciop, _data_to_sciop)

    def test_backend_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_backend_to_sciop(_op_sciop, _data_to_sciop)

    def test_prec_to_sciop(self, _op_sciop, _data_to_sciop):
        if _data_to_sciop["mode"] in {"matmat", "rmatmat"}:
            pytest.xfail(reason="Last axis is non-contiguous: apply/adjoint will fail.")
        else:
            super().test_prec_to_sciop(_op_sciop, _data_to_sciop)
