import collections.abc as cabc

import numpy as np
import pytest
import scipy.ndimage as scimage

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop.diff as pxld
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest

try:
    import scipy.ndimage._filters as scif
except ImportError:
    import scipy.ndimage.filters as scif


def diff_params_fd(scheme, accuracy, sampling):  # Finite Difference
    diff_params = {
        "scheme": scheme,
        "accuracy": accuracy,
    }
    gt_diff = {
        # diff type
        "central": {
            # accuracy
            2: {
                # order
                0: {
                    "coefs": np.array([1]),
                    "origin": 0,
                },
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
                0: {
                    "coefs": np.array([1]),
                    "origin": 0,
                },
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
                0: {
                    "coefs": np.array([1]),
                    "origin": 0,
                },
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
                0: {
                    "coefs": np.array([1]),
                    "origin": 0,
                },
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
                0: {
                    "coefs": np.array([1]),
                    "origin": 0,
                },
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
    gt_diffs = gt_diff[scheme][accuracy]

    return diff_params, gt_diffs


def diff_params_gd(sigma, truncate, sampling):  # Gaussian Derivative
    diff_params = {"sigma": sigma, "truncate": truncate}

    sigma_pix = sigma / sampling  # Sigma rescaled to pixel units
    radius = int(truncate * float(sigma_pix) + 0.5)
    gt_diffs = {
        0: {
            "coefs": np.flip(scif._gaussian_kernel1d(sigma_pix, 0, radius)),
            "origin": radius,
        },
        1: {
            "coefs": np.flip(scif._gaussian_kernel1d(sigma_pix, 1, radius)) / sampling,
            "origin": radius,
        },
        2: {
            "coefs": np.flip(scif._gaussian_kernel1d(sigma_pix, 2, radius)) / (sampling**2),
            "origin": radius,
        },
    }
    return diff_params, gt_diffs


def apply_derivative(arr, arg_shape, axis, gt_diffs, order, mode="constant"):
    # Apply derivative based on ground truth kernel

    axis = len(arg_shape) if axis == -1 else axis
    coefs = gt_diffs[order]["coefs"]
    origin = np.zeros(len(arg_shape) + 1, dtype="int8")
    origin[axis + 1] = gt_diffs[order]["origin"] - (len(coefs) // 2)
    kernel = np.array(coefs).reshape(*((1,) * len(arg_shape)), -1).swapaxes(axis + 1, -1)

    # Scipy and numpy padding modes have different names.
    mode = mode if mode != "reflect" else "mirror"
    mode = mode if mode != "symmetric" else "reflect"
    mode = mode if mode != "edge" else "nearest"

    return scimage.correlate(
        arr.reshape(-1, *arg_shape),
        kernel,
        mode=mode,
        origin=origin,
        cval=0.0,
    )


def apply_gradient(arr, arg_shape, gt_diffs, directions, diff_method, mode="constant"):
    if diff_method == "fd":
        pd = [apply_derivative(arr, arg_shape, axis, gt_diffs, 1, mode) for axis in directions]
    else:
        # diff_method == "gd"
        pd = []
        for axis in directions:
            out = apply_derivative(arr, arg_shape, axis, gt_diffs, 1, mode)
            # for smooth_axis in directions:
            for smooth_axis in range(len(arg_shape)):
                if smooth_axis != axis:
                    out = apply_derivative(out, arg_shape, smooth_axis, gt_diffs, 0, mode)
            pd.append(out)
    return np.stack(pd)


def apply_hessian(arr, arg_shape, gt_diffs, directions, diff_method, mode="constant"):
    pd = []
    for ax1, ax2 in directions:
        if ax1 == ax2:
            out = apply_derivative(arr, arg_shape, ax1, gt_diffs, 2, mode)
        else:
            out = apply_derivative(arr, arg_shape, ax1, gt_diffs, 1, mode)
            out = apply_derivative(out, arg_shape, ax2, gt_diffs, 1, mode)

        if diff_method == "gd":
            # for smooth_axis in directions:
            for smooth_axis in range(len(arg_shape)):
                if smooth_axis not in [ax1, ax2]:
                    out = apply_derivative(out, arg_shape, smooth_axis, gt_diffs, 0, mode)
        pd.append(out)
    return np.stack(pd)


@pytest.mark.filterwarnings("ignore::pyxu.info.warning.PrecisionWarning")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
class DiffOpMixin(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # 1,
            1.5,
            # 2
        ]
    )
    def sampling(self, request):
        return request.param

    @pytest.fixture(params=pxd.NDArrayInfo)
    def ndi(self, request):
        # [Sepand] Not inferred from spec(diff_kwargs) [unlike most Pyxu operators] since diff_kwargs() needs this information beforehand.
        ndi_ = request.param
        if ndi_.module() is None:
            pytest.skip(f"{ndi_} unsupported on this machine.")
        return ndi_

    @pytest.fixture(params=pxrt.Width)
    def width(self, request):
        # [Sepand] Not inferred from spec(diff_kwargs) [unlike most Pyxu operators] since diff_kwargs() needs this information beforehand.
        return request.param

    @pytest.fixture(params=["fd", "gd"])
    def diff_method(self, request):
        return request.param

    @pytest.fixture(
        params=[
            #  Finite Diff. ,   Gaussian Der.
            # (diff_typ, acc), (sigma, truncate)
            (("central", 2), (1.0, 1.0)),
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
    def diff_op(self) -> pxt.OpC:
        # To override in subclasses.
        raise NotImplementedError

    @pytest.fixture
    def diff_kwargs(self) -> dict:
        # To override in subclasses.
        raise NotImplementedError

    @pytest.fixture
    def spec(
        self,
        diff_op,
        diff_params,
        diff_kwargs,
        ndi,
        width,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        kwargs = diff_params.copy()
        kwargs.update(diff_kwargs)
        with pxrt.Precision(width):
            op = diff_op(**kwargs)
        return op, ndi, width


class TestPartialDerivative(DiffOpMixin):
    @pytest.fixture(
        params=[
            # (arg_shape, order, mode)
            ((10,), (1,), "constant"),
            ((10, 10), (2, 1), ("edge", "constant")),
            (
                (10, 10),
                (
                    0,
                    1,
                ),
                "edge",
            ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, order, mode) configs to test
        # * `request.param[0]` corresponds to raw inputs users provide to DiffOp().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):  # canonical representation
        arg_shape, _, _ = _spec
        return arg_shape

    @pytest.fixture
    def order(self, _spec):  # canonical representation (NumPy)
        _, order, _ = _spec
        return order

    @pytest.fixture
    def mode(self, _spec):  # canonical representation
        _, _, mode = _spec
        return mode

    @pytest.fixture
    def data_shape(self, arg_shape) -> pxt.NDArrayShape:
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
    def diff_kwargs(self, order, arg_shape, mode, ndi, width, sampling):
        return {
            "order": order,
            "arg_shape": arg_shape,
            "mode": mode,
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def data_apply(self, op, gt_diffs, order, arg_shape, mode) -> conftest.DataLike:
        arr = self._random_array((op.dim,), seed=20)

        order = (order,) if not isinstance(order, tuple) else order
        mode = (mode,) * len(arg_shape) if not isinstance(mode, tuple) else mode

        out = arr.copy()
        for ax, ord_ in enumerate(order):
            out = apply_derivative(out, arg_shape, ax, gt_diffs, ord_, mode[ax])

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )

    @pytest.fixture
    def diff_op(self, diff_method):
        if diff_method == "fd":
            return pxld.PartialDerivative.finite_difference
        elif diff_method == "gd":
            return pxld.PartialDerivative.gaussian_derivative


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
    def data_shape(self, arg_shape, directions) -> pxt.NDArrayShape:
        size = np.prod(arg_shape).item()
        n_derivatives = len(directions) if directions is not None else len(arg_shape)
        sh = (size * n_derivatives, size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxld.Gradient

    @pytest.fixture()
    def diff_kwargs(self, arg_shape, directions, ndi, width, sampling, diff_method):
        return {
            "arg_shape": arg_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, arg_shape, diff_method, gt_diffs, directions) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)
        directions = np.arange(len(arg_shape)) if directions is None else directions
        out = apply_gradient(
            arr,
            arg_shape=arg_shape,
            gt_diffs=gt_diffs,
            directions=directions,
            diff_method=diff_method,
        )
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestHessian(DiffOpMixin):
    @pytest.fixture(
        params=[
            #       (arg_shape, directions)
            #       (canonical directions)
            # 1D
            (((5,), 0), ((0, 0),)),  # Mode 0 (See Hessian Notes)
            (((5,), (0, 0)), ((0, 0),)),  # Mode 1
            (((5,), ((0, 0),)), ((0, 0),)),  # Mode 2
            (((5,), "all"), ((0, 0),)),  # Mode 3
            # ND
            (((5, 5, 5), 1), ((1, 1),)),  # Mode 0
            (((5, 5, 5), (0, 2)), ((0, 2),)),  # Mode 1
            (((5, 5, 5), ((1, 1), (1, 2))), ((1, 1), (1, 2))),  # Mode 2
            (((5, 5, 5), "all"), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))),  # Mode 3
        ]
    )
    def _spec(self, request):
        # (arg_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[0][1]

    @pytest.fixture
    def canonical_directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def data_shape(self, arg_shape, directions) -> pxt.NDArrayShape:
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
        return pxld.Hessian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, directions, ndi, width, sampling, diff_method):
        return {
            "arg_shape": arg_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, arg_shape, diff_method, gt_diffs, canonical_directions) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)  # random seed for reproducibility

        out = apply_hessian(
            arr,
            arg_shape=arg_shape,
            gt_diffs=gt_diffs,
            directions=canonical_directions,
            diff_method=diff_method,
        )
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestJacobian(DiffOpMixin):
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

    @pytest.fixture(params=[1, 3])
    def n_channels(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, arg_shape, n_channels, directions) -> pxt.NDArrayShape:
        size = np.prod(arg_shape).item()
        n_derivatives = len(directions) if directions is not None else len(arg_shape)
        sh = (size * n_derivatives * n_channels, size * n_channels)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxld.Jacobian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, n_channels, directions, ndi, width, sampling, diff_method):
        return {
            "arg_shape": arg_shape,
            "n_channels": n_channels,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, arg_shape, n_channels, diff_method, gt_diffs, directions) -> conftest.DataLike:
        arr = self._random_array((n_channels,) + arg_shape, seed=20)  # random seed for reproducibility
        directions = np.arange(len(arg_shape)) if directions is None else directions
        out = []
        for ch in range(n_channels):
            out.append(
                apply_gradient(
                    arr[ch],
                    arg_shape=arg_shape,
                    gt_diffs=gt_diffs,
                    directions=directions,
                    diff_method=diff_method,
                )
            )
        out = np.concatenate(out)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestDivergence(DiffOpMixin):
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
    def data_shape(self, arg_shape, directions) -> pxt.NDArrayShape:
        size = np.prod(arg_shape).item()
        n_derivatives = len(directions) if directions is not None else len(arg_shape)
        sh = (size, size * n_derivatives)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxld.Divergence

    @pytest.fixture
    def diff_kwargs(self, arg_shape, directions, ndi, width, sampling, diff_method):
        return {
            "arg_shape": arg_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, arg_shape, gt_diffs, directions, diff_method) -> conftest.DataLike:
        directions = np.arange(len(arg_shape)) if directions is None else directions
        arr = self._random_array((len(directions),) + arg_shape, seed=20)
        out = [
            apply_gradient(
                arr[i],
                arg_shape=arg_shape,
                gt_diffs=gt_diffs,
                directions=(ax,),
                diff_method=diff_method,
            )
            for i, ax in enumerate(directions)
        ]
        out = np.stack(out).sum(axis=0)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestLaplacian(DiffOpMixin):
    @pytest.fixture(params=[(5,), (5, 5, 5)])
    def arg_shape(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self, arg_shape) -> pxt.NDArrayShape:
        size = np.prod(arg_shape).item()
        sh = (size, size)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxld.Laplacian

    @pytest.fixture
    def diff_kwargs(self, arg_shape, ndi, width, sampling, diff_method):
        return {
            "arg_shape": arg_shape,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, arg_shape, gt_diffs, diff_method) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)  # random seed for reproducibility
        directions = tuple([(i, i) for i in range(len(arg_shape))])
        out = apply_hessian(
            arr,
            arg_shape=arg_shape,
            gt_diffs=gt_diffs,
            directions=directions,
            diff_method=diff_method,
        ).sum(axis=0)

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
