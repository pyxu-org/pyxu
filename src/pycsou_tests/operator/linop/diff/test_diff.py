import enum
import itertools
import math
import typing as typ

import numpy as np
import numpy.linalg as npl
import pytest
import scipy.ndimage as scimage

import pycsou.math.linalg as pylinalg
import pycsou.operator.linop.diff as pycdiff
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest

if pycd.CUPY_ENABLED:
    import cupy.linalg as cpl

import collections.abc as cabc

try:
    import scipy.ndimage._filters as scif
except ImportError:
    import scipy.ndimage.filters as scif


class DiffOpMixin(conftest.LinOpT):
    disable_test = frozenset(
        conftest.SquareOpT.disable_test
        | {
            # Stencil does not support evaluating inputs at different precisions.
            "test_precCM_adjoint",
            "test_precCM_adjoint_dagger",
            "test_precCM_adjoint_T",
            "test_precCM_apply",
            "test_precCM_apply_dagger",
            "test_precCM_apply_T",
            "test_precCM_call",
            "test_precCM_call_dagger",
            "test_precCM_call_T",
            "test_precCM_eigvals",
            "test_precCM_eigvals",
            "test_precCM_pinv",
            "test_precCM_svdvals",
            # from_sciop() tests try round trip Stencil<>to_sciop()<>from_sciop().
            # Compounded effect of approximations make most tests fail.
            # There is no reason to use from_sciop() in Stencil -> safe to disable.
            "test_value_from_sciop",
            "test_prec_from_sciop",
            "test_backend_from_sciop",
        }
    )

    @pytest.fixture
    def arg_shape(self, _spec):  # canonical representation
        arg_shape, _, _, _ = _spec[1]
        return arg_shape

    @pytest.fixture(
        params=[
            #          (arg_shape, order, axis, mode)
            (
                (10, (1,), (0,), "constant"),
                ((10,), (1,), (0,), ("constant",)),
            ),
            # (
            #         ((10, 10), (2, 1), (None,), ("edge", "constant")),
            #         ((10, 10), (2, 1), None, ("edge", "constant")),
            # ),
            # (
            #         ((10, 10), (1, ), (1,), ("constant", "wrap")),
            #         ((10, 10), (1,), (1,), ("constant", "wrap")),
            # ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, order, axis, mode) configs to test
        # * `request.param[0]` corresponds to raw inputs users provide to DiffOp().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture(
        params=[
            # 1,
            1.5,
            # 2
        ]
    )
    def sampling(self, request):
        return request.param

    @pytest.fixture
    def order(self, _spec):  # canonical representation (NumPy)
        _, order, _, _ = _spec[1]
        return order

    @pytest.fixture
    def axis(self, _spec):  # canonical representation
        _, _, axis, _ = _spec[1]
        return axis

    @pytest.fixture
    def mode(self, _spec):  # canonical representation
        _, _, _, mode = _spec[1]
        return mode

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture
    def gt_diffs(self, diff_params):
        # Ground truth diff kernels
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self, op, gt_diffs, order, arg_shape, axis, mode, sampling) -> conftest.DataLike:

        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility

        axis = [i for i in range(len(arg_shape))] if axis is None else axis

        order = [order] if not isinstance(order, typ.Sequence) else order
        axis = [axis] if not isinstance(axis, typ.Sequence) else axis

        out = np.zeros((1, *arg_shape))
        assert len(axis) == len(order)

        for ax, ord in zip(axis, order):
            ax = len(arg_shape) if ax == -1 else ax
            coefs = gt_diffs[ord]["coefs"] / (sampling**ord)
            origin = np.zeros(len(arg_shape) + 1, dtype="int8")
            origin[ax + 1] = gt_diffs[ord]["origin"] - (len(coefs) // 2)
            kernel = np.array(coefs).reshape(*((1,) * len(arg_shape)), -1).swapaxes(ax + 1, -1)

            # Scipy and numpy padding modes have different names.
            _mode = mode[ax]
            _mode = _mode if _mode != "reflect" else "mirror"
            _mode = _mode if _mode != "symmetric" else "reflect"
            _mode = _mode if _mode != "edge" else "nearest"

            out += scimage.correlate(arr.reshape(-1, *arg_shape), kernel, mode=_mode, origin=origin, cval=0.0)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.NDArrayShape:
        size = np.prod(arg_shape)
        sh = (size, size)
        return sh


class TestFiniteDifferences(DiffOpMixin):
    @pytest.fixture(
        params=[
            # diff_typ, acc
            ("forward", 2),
            # ("forward", 4),
            # ("backward", 2),
            # ("central", 2),
            # ("central", 4),
        ]
    )
    def diff_params(self, request):
        return request.param

    @pytest.fixture
    def gt_diffs(self, diff_params):
        gt_diff = {
            # diff type
            "central": {
                # accuracy
                2: {
                    # order
                    1: {
                        "coefs": np.array([-1 / 2, 0, 1 / 2]),
                        "origin": 1,
                    },
                    2: {
                        "coefs": np.array([1, -2, 1]),
                        "origin": 1,
                    },
                },
                4: {
                    1: {
                        "coefs": np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
                        "origin": 2,
                    },
                    2: {
                        "coefs": np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
                        "origin": 2,
                    },
                },
            },
            "forward": {
                2: {
                    1: {
                        "coefs": np.array([-3 / 2, 2, -1 / 2]),
                        "origin": 0,
                    },
                    2: {
                        "coefs": np.array([2, -5, 4, -1]),
                        "origin": 0,
                    },
                },
                4: {
                    1: {
                        "coefs": np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]),
                        "origin": 0,
                    },
                    2: {
                        "coefs": np.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6]),
                        "origin": 0,
                    },
                },
            },
            "backward": {
                2: {
                    1: {
                        "coefs": np.array([1 / 2, -2, 3 / 2]),
                        "origin": 2,
                    },
                    2: {
                        "coefs": np.array([-1, 4, -5, 2]),
                        "origin": 3,
                    },
                }
            },
        }
        return gt_diff[diff_params[0]][diff_params[1]]

    @pytest.fixture
    def spec(
        self, _spec, diff_params, order, arg_shape, axis, ndi, width, mode, sampling
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        with pycrt.Precision(width):
            op = pycdiff.FiniteDifference(
                order=order,
                arg_shape=arg_shape,
                diff_type=diff_params[0],
                axis=axis,
                accuracy=diff_params[1],
                mode=mode,
                gpu=ndi.name == "CUPY",
                dtype=width.value,
                sampling=sampling,
            )
        return op, ndi, width


class TestGaussianDerivative(DiffOpMixin):
    @pytest.fixture(
        params=[
            # sigma, truncate
            (1, 2),
            # ("forward", 4),
            # ("backward", 2),
            # ("central", 2),
            # ("central", 4),
        ]
    )
    def diff_params(self, request):
        return request.param

    @pytest.fixture
    def gt_diffs(self, diff_params):
        # based on https: // en.wikipedia.org / wiki / Finite_difference_coefficient
        sigma = diff_params[0]
        truncate = diff_params[1]
        radius = int(truncate * float(sigma) + 0.5)
        return {
            1: {"coefs": np.flip(scif._gaussian_kernel1d(sigma, 1, radius)), "origin": radius // 2 + 1},
            2: {"coefs": np.flip(scif._gaussian_kernel1d(sigma, 2, radius)), "origin": radius // 2 + 1},
        }

    @pytest.fixture
    def spec(
        self, _spec, diff_params, order, arg_shape, axis, ndi, width, mode, sampling
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:
        with pycrt.Precision(width):
            op = pycdiff.GaussianDerivative(
                order=order,
                arg_shape=arg_shape,
                sigma=diff_params[0],
                axis=axis,
                truncate=diff_params[1],
                mode=mode,
                gpu=ndi.name == "CUPY",
                dtype=width.value,
                sampling=sampling,
            )
        return op, ndi, width
