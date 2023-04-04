import collections.abc as cabc
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


def diff_params_fd(diff_type, accuracy, sampling):  # Finite Difference
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
                    "coefs": np.array([-1 / 2, 0, 1 / 2]) / sampling,
                    "origin": 1,
                },
                2: {
                    "coefs": np.array([1, -2, 1]) / (sampling**2),
                    "origin": 1,
                },
            },
            4: {
                1: {
                    "coefs": np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]) / sampling,
                    "origin": 2,
                },
                2: {
                    "coefs": np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]) / (sampling**2),
                    "origin": 2,
                },
            },
        },
        "forward": {
            2: {
                1: {
                    "coefs": np.array([-3 / 2, 2, -1 / 2]) / sampling,
                    "origin": 0,
                },
                2: {
                    "coefs": np.array([2, -5, 4, -1]) / (sampling**2),
                    "origin": 0,
                },
            },
            4: {
                1: {
                    "coefs": np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]) / sampling,
                    "origin": 0,
                },
                2: {
                    "coefs": np.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6]) / (sampling**2),
                    "origin": 0,
                },
            },
        },
        "backward": {
            2: {
                1: {
                    "coefs": np.array([1 / 2, -2, 3 / 2]) / sampling,
                    "origin": 2,
                },
                2: {
                    "coefs": np.array([-1, 4, -5, 2]) / (sampling**2),
                    "origin": 3,
                },
            }
        },
    }
    gt_diffs = gt_diff[diff_type][accuracy]

    return diff_params, gt_diffs


def diff_params_gd(sigma, truncate, sampling):  # Gaussian Derivative
    diff_params = {"sigma": sigma, "truncate": truncate}

    sigma_pix = sigma / sampling  # Sigma rescaled to pixel units
    radius = int(truncate * float(sigma_pix) + 0.5)
    gt_diffs = {
        1: {"coefs": np.flip(scif._gaussian_kernel1d(sigma_pix, 1, radius)) / sampling, "origin": radius // 2 + 1},
        2: {
            "coefs": np.flip(scif._gaussian_kernel1d(sigma_pix, 2, radius)) / (sampling**2),
            "origin": radius // 2 + 1,
        },
    }
    return diff_params, gt_diffs


@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
class DiffOpMixin(conftest.LinOpT):
    disable_test = frozenset(
        conftest.SquareOpT.disable_test
        | {
            # from_sciop() tests try round trip Stencil<>to_sciop()<>from_sciop().
            # Compounded effect of approximations make most tests fail.
            # There is no reason to use from_sciop() in Stencil -> safe to disable.
            "test_value_from_sciop",
            "test_prec_from_sciop",
            "test_backend_from_sciop",
        }
    )

    @pytest.fixture(
        params=[
            # 1,
            1.5,
            # 2
        ]
    )
    def sampling(self, request):
        return request.param

    @pytest.fixture(params=pycd.NDArrayInfo)
    def ndi(self, request):
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request):
        return request.param

    @pytest.fixture
    def spec(self, diff_op, diff_params, diff_kwargs, ndi, width) -> tuple[pyct.OpT, pycd.NDArrayInfo, pycrt.Width]:

        kwargs = diff_params.copy()
        kwargs.update(diff_kwargs)
        with pycrt.Precision(width):
            op = diff_op(**kwargs)
        return op, ndi, width


class TestPartialDerivative(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (arg_shape, order, axis, mode)
            ((10,), (1,), (0,), ("constant",)),
            ((10, 10), (2, 1), None, ("edge", "constant")),
            ((10, 10), (1,), (1,), ("constant", "wrap")),
        ]
    )
    def _spec(self, request):
        # (arg_shape, order, axis, mode) configs to test
        # * `request.param[0]` corresponds to raw inputs users provide to DiffOp().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):  # canonical representation
        arg_shape, _, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def order(self, _spec):  # canonical representation (NumPy)
        _, order, _, _ = _spec
        return order

    @pytest.fixture
    def axis(self, _spec):  # canonical representation
        _, _, axis, _ = _spec
        return axis

    @pytest.fixture
    def mode(self, _spec):  # canonical representation
        _, _, _, mode = _spec
        return mode

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.NDArrayShape:
        size = np.prod(arg_shape).item()
        sh = (size, size)
        return sh

    @pytest.fixture(
        params=[
            #  Finite Diff. ,   Gaussian Der.
            # (diff_typ, acc), (sigma, truncate)
            (("forward", 2), (2.0, 1.0)),
            (("forward", 4), (1.0, 1.0)),
            (("backward", 2), (1.0, 1.0)),
            (("central", 2), (1.0, 1.0)),
            (("central", 4), (1.0, 1.0)),
        ]
    )
    def init_params(self, diff_method, sampling, request):
        params_fd, params_gd = request.param
        if diff_method == "fd":
            return diff_params_fd(params_fd[0], params_fd[1], sampling)
        elif diff_method == "gd":
            return diff_params_gd(params_gd[0], params_gd[1], sampling)

    @pytest.fixture
    def diff_params(self, init_params):
        return init_params[0]

    @pytest.fixture
    def gt_diffs(self, init_params):
        return init_params[1]

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

    @pytest.fixture
    def data_apply(self, op, gt_diffs, order, arg_shape, axis, mode) -> conftest.DataLike:

        arr = self._random_array((op.dim,), seed=20)  # random seed for reproducibility

        axis = [i for i in range(len(arg_shape))] if axis is None else axis

        order = [order] if not isinstance(order, typ.Sequence) else order
        axis = [axis] if not isinstance(axis, typ.Sequence) else axis

        out = np.zeros((1, *arg_shape))
        assert len(axis) == len(order)

        for ax, ord in zip(axis, order):
            ax = len(arg_shape) if ax == -1 else ax
            coefs = gt_diffs[ord]["coefs"]
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

    @pytest.fixture(params=["fd", "gd"])
    def diff_method(self, request):
        return request.param

    @pytest.fixture
    def diff_op(self, diff_method):
        if diff_method == "fd":
            return pycdiff.PartialDerivative.finite_difference
        elif diff_method == "gd":
            return pycdiff.PartialDerivative.gaussian_derivative


class TestGradient(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (arg_shape, directions)
            (
                (5,),
                (0,),
            ),
            (
                (5, 5),
                (0, 1),
            ),
            (
                (5, 5),
                None,
            ),
            (
                (5, 5, 5),
                (0, 2),
            ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        return _spec[0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def data_shape(self, arg_shape, directions) -> pyct.NDArrayShape:
        size = np.prod(arg_shape).item()
        n_derivatives = len(directions) if directions is not None else len(arg_shape)
        sh = (size * n_derivatives, size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pycdiff.Gradient

    @pytest.fixture
    def diff_kwargs(self, arg_shape, directions, ndi, width, sampling):
        return {
            "arg_shape": arg_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def diff_params(self):
        return {"diff_type": "central", "accuracy": 1}

    @pytest.fixture
    def data_apply(self, op, arg_shape, sampling, directions) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)
        directions = np.arange(len(arg_shape)) if directions is None else directions
        x_np = np.pad(arr, ((1, 1),) * len(arg_shape))
        slices = (slice(None, None),) + (slice(1, -1, None),) * len(arg_shape)
        out = np.gradient(x_np, sampling, edge_order=2, axis=directions)
        if len(directions) == 1:
            out = [
                out,
            ]
        out = np.stack(out)[slices]
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestHessian(DiffOpMixin):
    @pytest.fixture(
        params=[
            # (arg_shape, directions)
            ((5,), 0),  # Mode 0 (See Hessian Notes)
            ((5,), (0, 0)),  # Mode 1
            ((5,), ((0, 0),)),  # Mode 2
            ((5,), "all"),  # Mode 3
            ((5, 5, 5), 1),  # Mode 0
            ((5, 5, 5), (0, 2)),  # Mode 1
            ((5, 5, 5), ((1, 1), (1, 2))),  # Mode 2
            ((5, 5, 5), "all"),  # Mode 3
        ]
    )
    def _spec(self, request):
        # (arg_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        return _spec[0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def data_shape(self, arg_shape, directions) -> pyct.NDArrayShape:
        size = np.prod(arg_shape).item()
        if isinstance(directions, int):  # Case 0
            sh = (size, size)
        elif isinstance(directions, cabc.Sequence):
            if isinstance(directions, str):  # case 3
                n_derivatives = len(arg_shape) * (len(arg_shape) + 1) // 2
                sh = (size * n_derivatives, size)
            elif isinstance(directions[0], int):  # Case 1
                sh = (size, size)
            elif isinstance(directions[0], cabc.Sequence):  # Case 2
                sh = (size * len(directions), size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pycdiff.Hessian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, directions, ndi, width, sampling):
        return {
            "arg_shape": arg_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def diff_params(self):
        return {"diff_type": "forward", "accuracy": 1}

    @pytest.fixture
    def data_apply(self, op, arg_shape, sampling, directions) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)  # random seed for reproducibility

        # compute forward finite diffs
        grad = []
        for ax in range(len(arg_shape)):
            arr_pad = np.pad(arr, ((0, 0),) * ax + ((1, 1),) + ((0, 0),) * (len(arg_shape) - ax - 1))
            slices = (
                (slice(None, None),) * ax + (slice(1, None, None),) + (slice(None, None),) * (len(arg_shape) - ax - 1)
            )
            grad.append(np.diff(arr_pad, axis=ax)[slices] / sampling)

        # Canonical form for dimensions
        directions = np.arange(len(arg_shape)) if directions is None else directions
        if isinstance(directions, int):
            directions = ((directions, directions),)
        elif isinstance(directions, str):
            # Directions == "all":
            import itertools

            directions = tuple(
                list(_) for _ in itertools.combinations_with_replacement(np.arange(len(arg_shape)).astype(int), 2)
            )
        elif isinstance(directions, cabc.Sequence):
            if not isinstance(directions[0], cabc.Sequence):
                directions = (directions,)

        out = np.empty((len(directions),) + arg_shape)

        for d, (k, l) in enumerate(directions):
            grad_pad = np.pad(grad[k], ((0, 0),) * l + ((1, 1),) + ((0, 0),) * (len(arg_shape) - l - 1))
            slices = (
                (slice(None, None),) * l + (slice(1, None, None),) + (slice(None, None),) * (len(arg_shape) - l - 1)
            )
            out[d] = np.diff(grad_pad, axis=l)[slices] / sampling
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestJacobian(DiffOpMixin):
    @pytest.fixture(params=[(5,), (5, 5, 5)])
    def arg_shape(self, request):
        return request.param

    @pytest.fixture(params=[1, 3])
    def n_channels(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, arg_shape, n_channels) -> pyct.NDArrayShape:
        size = np.prod(arg_shape).item()
        n_derivatives = len(arg_shape)
        sh = (size * n_derivatives * n_channels, size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pycdiff.Jacobian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, n_channels, ndi, width, sampling):
        return {
            "arg_shape": arg_shape,
            "n_channels": n_channels,
            "mode": "constant",
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def diff_params(self):
        return {"diff_type": "central", "accuracy": 1}

    @pytest.fixture
    def data_apply(self, op, arg_shape, sampling, n_channels) -> conftest.DataLike:
        arr = self._random_array((n_channels,) + arg_shape, seed=20)  # random seed for reproducibility
        out = []
        for ch in range(n_channels):
            x_np = np.pad(arr[ch], ((1, 1),) * len(arg_shape))
            slices = (slice(None, None),) + (slice(1, -1, None),) * len(arg_shape)
            pad_shape = tuple(d + 2 for d in arg_shape)
            out_ = np.gradient(x_np, sampling, edge_order=2)
            if len(arg_shape) > 1:
                out_ = np.concatenate(out_, axis=0)
            out.append(out_.reshape(len(arg_shape), *pad_shape)[slices])
        out = np.concatenate(out)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestLaplacian(DiffOpMixin):
    @pytest.fixture(params=[(5,), (5, 5, 5)])
    def arg_shape(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, arg_shape) -> pyct.NDArrayShape:
        size = np.prod(arg_shape).item()
        sh = (size, size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pycdiff.Laplacian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, ndi, width, sampling):
        return {
            "arg_shape": arg_shape,
            "mode": "constant",
            "gpu": ndi.name == "CUPY",
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def diff_params(self):
        return {"diff_type": "forward", "accuracy": 1}

    @pytest.fixture
    def data_apply(self, op, arg_shape, sampling) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)  # random seed for reproducibility

        # compute forward finite diffs
        out = np.zeros(arg_shape)
        for ax in range(len(arg_shape)):
            arr_pad = np.pad(arr, ((0, 0),) * ax + ((1, 1),) + ((0, 0),) * (len(arg_shape) - ax - 1))
            slices = (
                (slice(None, None),) * ax + (slice(1, None, None),) + (slice(None, None),) * (len(arg_shape) - ax - 1)
            )
            grad = np.diff(arr_pad, axis=ax)[slices] / sampling
            grad_pad = np.pad(grad, ((0, 0),) * ax + ((1, 1),) + ((0, 0),) * (len(arg_shape) - ax - 1))
            out += np.diff(grad_pad, axis=ax)[slices] / sampling

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
