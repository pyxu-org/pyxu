import typing as typ

import numpy as np
import pytest
import scipy.ndimage as scimage

import pycsou.operator.linop.diff as pycdiff
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou_tests.operator.conftest as conftest

try:
    import scipy.ndimage._filters as scif
except ImportError:
    import scipy.ndimage.filters as scif


def diff_params_fd(diff_type, accuracy):  # Finite Difference
    diff_params = {
        "diff_type": diff_type,
        "accuracy": accuracy,
    }
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
    gt_diffs = gt_diff[diff_type][accuracy]

    return diff_params, gt_diffs


def diff_params_gd(sigma, truncate):  # Gaussian Derivative
    radius = int(truncate * float(sigma) + 0.5)
    diff_params = {"sigma": sigma, "truncate": truncate}
    gt_diffs = {
        1: {"coefs": np.flip(scif._gaussian_kernel1d(sigma, 1, radius)), "origin": radius // 2 + 1},
        2: {"coefs": np.flip(scif._gaussian_kernel1d(sigma, 2, radius)), "origin": radius // 2 + 1},
    }
    return diff_params, gt_diffs


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

    @pytest.fixture(
        params=[
            #  Finite Diff. ,   Gaussian Der.
            # (diff_typ, acc), (sigma, truncate)
            (("forward", 2), (2, 1))
            # (("forward", 4), (, ))
            # (("backward", 2), (, ))
            # (("central", 2), (, ))
            # (("central", 4), (, ))
        ]
    )
    def init_params(self, diff_method, request):
        params_fd, params_gd = request.param
        if diff_method == "fd":
            return diff_params_fd(params_fd[0], params_fd[1])
        elif diff_method == "gd":
            return diff_params_gd(params_gd[0], params_gd[1])

    @pytest.fixture
    def diff_params(self, init_params):
        return init_params[0]

    @pytest.fixture
    def gt_diffs(self, init_params):
        return init_params[1]

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

    @pytest.fixture
    def diff_kwargs(self, order, arg_shape, axis, mode, ndi, width, sampling):
        return {
            "order": order,
            "arg_shape": arg_shape,
            "axis": axis,
            "mode": mode,
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def spec(
        self, _spec, diff_op, diff_params, diff_kwargs, ndi, width
    ) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:

        kwargs = diff_params.copy()
        kwargs.update(diff_kwargs)
        with pycrt.Precision(width):
            op = diff_op(**kwargs)
        return op, ndi, width


class TestFiniteDifferences(DiffOpMixin):
    @pytest.fixture(params=["fd"])
    def diff_method(self, request):
        return request.param

    @pytest.fixture
    def diff_op(self):
        return pycdiff.FiniteDifference


class TestGaussianDerivative(DiffOpMixin):
    @pytest.fixture(params=["gd"])
    def diff_method(self, request):
        return request.param

    @pytest.fixture
    def diff_op(self):
        return pycdiff.GaussianDerivative


class TestPartialDerivative(DiffOpMixin):
    @pytest.fixture(params=["fd", "gd"])
    def diff_method(self, request):
        return request.param

    @pytest.fixture
    def diff_op(self, diff_method):
        if diff_method == "fd":
            return pycdiff.PartialDerivative.finite_difference
        elif diff_method == "gd":
            return pycdiff.PartialDerivative.gaussian_derivative

    @pytest.fixture
    def diff_kwargs(self, order, arg_shape, mode, ndi, width, sampling):
        return {
            "order": order,
            "arg_shape": arg_shape,
            "mode": mode,
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }
