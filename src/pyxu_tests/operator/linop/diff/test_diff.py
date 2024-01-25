import collections.abc as cabc

import numpy as np
import pytest
import scipy.ndimage as scimage

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
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


def apply_derivative(arr, dim_shape, axis, gt_diffs, order, mode="constant"):
    # Apply derivative based on ground truth kernel

    axis = len(dim_shape) if axis == -1 else axis
    coefs = gt_diffs[order]["coefs"]
    origin = np.zeros(len(dim_shape) + 1, dtype="int8")
    origin[axis + 1] = gt_diffs[order]["origin"] - (len(coefs) // 2)
    kernel = np.array(coefs).reshape(*((1,) * len(dim_shape)), -1).swapaxes(axis + 1, -1)

    # Scipy and numpy padding modes have different names.
    mode = mode if mode != "reflect" else "mirror"
    mode = mode if mode != "symmetric" else "reflect"
    mode = mode if mode != "edge" else "nearest"

    return scimage.correlate(
        arr.reshape(1, *dim_shape),
        kernel,
        mode=mode,
        origin=origin,
        cval=0.0,
    )[0]


def apply_gradient(arr, dim_shape, gt_diffs, directions, diff_method, mode="constant"):
    if diff_method == "fd":
        pd = [apply_derivative(arr, dim_shape, axis, gt_diffs, 1, mode) for axis in directions]
    else:
        # diff_method == "gd"
        pd = []
        for axis in directions:
            out = apply_derivative(arr, dim_shape, axis, gt_diffs, 1, mode)
            # for smooth_axis in directions:
            for smooth_axis in range(len(dim_shape)):
                if smooth_axis != axis:
                    out = apply_derivative(out, dim_shape, smooth_axis, gt_diffs, 0, mode)
            pd.append(out)
    return np.stack(pd)


def apply_hessian(arr, dim_shape, gt_diffs, directions, diff_method, mode="constant"):
    pd = []
    for ax1, ax2 in directions:
        if ax1 == ax2:
            out = apply_derivative(arr, dim_shape, ax1, gt_diffs, 2, mode)
        else:
            out = apply_derivative(arr, dim_shape, ax1, gt_diffs, 1, mode)
            out = apply_derivative(out, dim_shape, ax2, gt_diffs, 1, mode)

        if diff_method == "gd":
            # for smooth_axis in directions:
            for smooth_axis in range(len(dim_shape)):
                if smooth_axis not in [ax1, ax2]:
                    out = apply_derivative(out, dim_shape, smooth_axis, gt_diffs, 0, mode)
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
            # (dim_shape, order, mode)
            ((10,), (1,), "constant"),
            ((10, 10), (2, 1), ("edge", "constant")),
            # ((10, 10), (0, 1), "edge"),
        ]
    )
    def _spec(self, request):
        # (dim_shape, order, mode) configs to test
        # * `request.param[0]` corresponds to raw inputs users provide to DiffOp().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec):  # canonical representation
        dim_shape, _, _ = _spec
        return dim_shape

    @pytest.fixture
    def codim_shape(self, _spec):  # canonical representation
        dim_shape, _, _ = _spec
        return dim_shape

    @pytest.fixture
    def order(self, _spec):  # canonical representation (NumPy)
        _, order, _ = _spec
        return order

    @pytest.fixture
    def mode(self, _spec):  # canonical representation
        _, _, mode = _spec
        return mode

    @pytest.fixture(
        params=[
            #  Finite Diff. ,   Gaussian Der.
            # (diff_typ, acc), (sigma, truncate)
            (("forward", 2), (2.0, 1.0)),
            # (("forward", 4), (1.0, 1.0)),
            (("backward", 2), (1.0, 1.0)),
            (("central", 2), (1.0, 1.0)),
            # (("central", 4), (1.0, 1.0)),
        ]
    )
    def init_params(self, diff_method, sampling, request):
        params_fd, params_gd = request.param
        if diff_method == "fd":
            return diff_params_fd(params_fd[0], params_fd[1], sampling)
        elif diff_method == "gd":
            return diff_params_gd(params_gd[0], params_gd[1], sampling)

    @pytest.fixture
    def diff_kwargs(self, order, dim_shape, mode, ndi, width, sampling):
        return {
            "order": order,
            "dim_shape": dim_shape,
            "mode": mode,
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
        }

    @pytest.fixture
    def data_apply(self, op, gt_diffs, order, dim_shape, mode) -> conftest.DataLike:
        arr = self._random_array(op.dim_shape, seed=20)

        order = (order,) if not isinstance(order, tuple) else order
        mode = (mode,) * len(dim_shape) if not isinstance(mode, tuple) else mode

        out = arr.copy()
        for ax, ord_ in enumerate(order):
            out = apply_derivative(out, dim_shape, ax, gt_diffs, ord_, mode[ax])

        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    @pytest.fixture
    def diff_op(self, diff_method):
        if diff_method == "fd":
            return pxo.PartialDerivative.finite_difference
        elif diff_method == "gd":
            return pxo.PartialDerivative.gaussian_derivative


class TestGradient(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (dim_shape, directions)
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
            # (
            #     (5, 5, 5),
            #     (0, 2),
            # ),
        ]
    )
    def _spec(self, request):
        # (dim_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec):
        return _spec[0]

    @pytest.fixture
    def codim_shape(self, _spec):  # canonical representation
        dim_shape, directions = _spec
        n_directions = len(directions) if directions is not None else (len(dim_shape))
        return (n_directions, *dim_shape)

    @pytest.fixture
    def directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def diff_op(self):
        return pxo.Gradient

    @pytest.fixture()
    def diff_kwargs(self, dim_shape, directions, ndi, width, sampling, diff_method):
        return {
            "dim_shape": dim_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, dim_shape, diff_method, gt_diffs, directions) -> conftest.DataLike:
        arr = self._random_array(dim_shape, seed=20)
        directions = np.arange(len(dim_shape)) if directions is None else directions
        out = apply_gradient(
            arr,
            dim_shape=dim_shape,
            gt_diffs=gt_diffs,
            directions=directions,
            diff_method=diff_method,
        )
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


class TestHessian(DiffOpMixin):
    @pytest.fixture(
        params=[
            #       (dim_shape, directions)
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
        # (dim_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[0][1]

    @pytest.fixture
    def canonical_directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, dim_shape, directions) -> pxt.NDArrayShape:
        if isinstance(directions, int):  # Case 0
            sh = (1, *dim_shape)
        elif isinstance(directions, cabc.Sequence):
            if isinstance(directions, str):  # case 3
                n_derivatives = len(dim_shape) * (len(dim_shape) + 1) // 2
                sh = (n_derivatives, *dim_shape)
            elif isinstance(directions[0], int):  # Case 1
                sh = (1, *dim_shape)
            elif isinstance(directions[0], cabc.Sequence):  # Case 2
                sh = (len(directions), *dim_shape)
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxo.Hessian

    @pytest.fixture
    def diff_kwargs(self, dim_shape, directions, ndi, width, sampling, diff_method):
        return {
            "dim_shape": dim_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, dim_shape, diff_method, gt_diffs, canonical_directions) -> conftest.DataLike:
        arr = self._random_array(dim_shape, seed=20)  # random seed for reproducibility

        out = apply_hessian(
            arr,
            dim_shape=dim_shape,
            gt_diffs=gt_diffs,
            directions=canonical_directions,
            diff_method=diff_method,
        )
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


class TestJacobian(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (dim_shape, directions)
            (
                (
                    2,
                    5,
                ),
                (1,),
            ),
            (
                (3, 5, 5),
                (1, 2),
            ),
            (
                (3, 5, 5),
                None,
            ),
            (
                (2, 5, 5, 5),
                (1, 3),
            ),
        ]
    )
    def _spec(self, request):
        # (dim_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec):
        return _spec[0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, dim_shape, directions) -> pxt.NDArrayShape:
        n_derivatives = len(directions) if directions is not None else len(dim_shape) - 1
        sh = (dim_shape[0], n_derivatives, *dim_shape[1:])
        return sh

    @pytest.fixture
    def diff_op(self):
        return pxo.Jacobian

    @pytest.fixture
    def diff_kwargs(self, dim_shape, directions, ndi, width, sampling, diff_method):
        return {
            "dim_shape": dim_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, dim_shape, diff_method, gt_diffs, directions) -> conftest.DataLike:
        arr = self._random_array(dim_shape, seed=20)  # random seed for reproducibility
        directions = np.arange(len(dim_shape[1:])) if directions is None else tuple(np.array(directions) - 1)
        out = []
        for ch in range(dim_shape[0]):
            out.append(
                apply_gradient(
                    arr[ch],
                    dim_shape=dim_shape[1:],
                    gt_diffs=gt_diffs,
                    directions=directions,
                    diff_method=diff_method,
                )
            )
        out = np.stack(out)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


class TestDivergence(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (dim_shape, directions)
            (
                (
                    1,
                    5,
                ),
                (1,),
            ),
            (
                (2, 5, 5),
                (1, 2),
            ),
            (
                (2, 5, 5),
                None,
            ),
            # (
            #     (3, 5, 5, 5),
            #     (1, 3),
            # ),
        ]
    )
    def _spec(self, request):
        # (dim_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec):
        return _spec[0]

    @pytest.fixture
    def directions(self, _spec):
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape[1:]

    @pytest.fixture
    def diff_op(self):
        return pxo.Divergence

    @pytest.fixture
    def diff_kwargs(self, dim_shape, directions, ndi, width, sampling, diff_method):
        return {
            "dim_shape": dim_shape,
            "directions": directions,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, dim_shape, gt_diffs, directions, diff_method) -> conftest.DataLike:
        directions = np.arange(0, len(dim_shape) - 1) if directions is None else tuple(np.array(directions) - 1)
        arr = self._random_array(dim_shape, seed=20)
        out = [
            apply_gradient(
                arr[i],
                dim_shape=dim_shape[1:],
                gt_diffs=gt_diffs,
                directions=(ax,),
                diff_method=diff_method,
            ).squeeze()
            for i, ax in enumerate(directions)
        ]
        out = np.stack(out).sum(axis=0)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


class TestLaplacian(DiffOpMixin):
    @pytest.fixture(params=[(5,), (5, 5, 5)])
    def dim_shape(self, request):
        return request.param

    @pytest.fixture
    def diff_op(self):
        return pxo.Laplacian

    @pytest.fixture
    def diff_kwargs(self, dim_shape, ndi, width, sampling, diff_method):
        return {
            "dim_shape": dim_shape,
            "mode": "constant",
            "gpu": ndi == pxd.NDArrayInfo.CUPY,
            "dtype": width.value,
            "sampling": sampling,
            "diff_method": diff_method,
        }

    @pytest.fixture
    def data_apply(self, op, dim_shape, gt_diffs, diff_method) -> conftest.DataLike:
        arr = self._random_array(dim_shape, seed=20)  # random seed for reproducibility
        directions = tuple([(i, i) for i in range(len(dim_shape))])
        out = apply_hessian(
            arr,
            dim_shape=dim_shape,
            gt_diffs=gt_diffs,
            directions=directions,
            diff_method=diff_method,
        ).sum(axis=0)

        return dict(
            in_=dict(arr=arr),
            out=out,
        )
