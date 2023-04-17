import numpy as np
import pytest
from test_diff import DiffOpMixin

import pycsou.operator.linop.diff as pycdiff
import pycsou_tests.operator.conftest as conftest


class TestDirectionalDerivative(DiffOpMixin):
    @pytest.fixture(
        params=[
            #          (arg_shape, directions)
            (
                (5,),
                (1.0,),
            ),
            (
                (5, 5),
                (0.1, 0.9),
            ),
            (
                (5, 5),
                (1.0, 0.0),
            ),
            (
                (5, 5, 5),
                (0.1, 2.0, 1.0),
            ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, directions) configs to test
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):
        return _spec[0]

    @pytest.fixture(params=["constant", "varying"])
    def directions(self, _spec, arg_shape, request):

        dirs = np.array(_spec[1])
        dirs = dirs / np.linalg.norm(dirs)

        if request.param == "constant":
            pass
        else:
            expand_dims = np.arange(len(arg_shape)) + 1
            broadcast_arr = np.ones((len(arg_shape),) + arg_shape)
            dirs = np.expand_dims(dirs, expand_dims.tolist()) * broadcast_arr

        return dirs

    @pytest.fixture(params=[1, 2])
    def which(self, request):
        return request.param

    @pytest.fixture
    def diff_kwargs(self, arg_shape, which, directions, ndi, width, sampling):
        xp = ndi.module()
        return {
            "arg_shape": arg_shape,
            "which": which,
            "directions": xp.array(directions, dtype=width.value),
            "mode": "constant",
            "sampling": sampling,
        }

    @pytest.fixture
    def diff_params(self, which):
        return {"diff_type": "central" if which == 1 else "forward", "accuracy": 1}

    @pytest.fixture
    def diff_op(self):
        return pycdiff.DirectionalDerivative

    @pytest.fixture
    def data_apply(self, op, arg_shape, sampling, directions, which) -> conftest.DataLike:
        arr = self._random_array(arg_shape, seed=20)
        out = np.zeros(arg_shape)
        grad_directions = np.arange(len(arg_shape))
        if which == 1:
            x_np = np.pad(arr, ((1, 1),) * len(arg_shape))
            slices = (slice(1, -1, None),) * len(arg_shape)
            grad = np.gradient(x_np, sampling, edge_order=2, axis=grad_directions)
            if len(grad_directions) == 1:
                grad = [
                    grad,
                ]
            for i, g in enumerate(grad):
                out += directions[i] * g[slices]
        elif which == 2:
            # compute forward finite diffs
            grad = []
            for ax in range(len(arg_shape)):
                arr_pad = np.pad(arr, ((0, 0),) * ax + ((1, 1),) + ((0, 0),) * (len(arg_shape) - ax - 1))
                slices = (
                    (slice(None, None),) * ax
                    + (slice(1, None, None),)
                    + (slice(None, None),) * (len(arg_shape) - ax - 1)
                )
                grad.append(np.diff(arr_pad, axis=ax)[slices] / sampling)

            # Canonical form for dimensions
            import itertools

            hessian_directions = tuple(
                list(_) for _ in itertools.product(np.arange(len(arg_shape)).astype(int), repeat=2)
            )
            hessian = np.empty(arg_shape + (len(arg_shape), len(arg_shape)))

            for k, l in hessian_directions:
                grad_pad = np.pad(grad[k], ((0, 0),) * l + ((1, 1),) + ((0, 0),) * (len(arg_shape) - l - 1))
                slices = (
                    (slice(None, None),) * l + (slice(1, None, None),) + (slice(None, None),) * (len(arg_shape) - l - 1)
                )
                hessian[..., k, l] = np.diff(grad_pad, axis=l)[slices] / sampling
                # Compute directional derivative
                out += directions[k] * hessian[..., k, l] * directions[l]

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
