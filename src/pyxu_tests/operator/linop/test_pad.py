import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop as pxl
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest

# Note: Pad.[asarray, pinv, dagger]() are expensive.
# Consider disabling them in test suite.


class TestPad(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # 1D, uni-mode -------------------------------------
            (
                ((5,), 2, "constant"),
                ((5,), ((2, 2),), ("constant",)),
            ),
            (
                ((5,), 2, "wrap"),
                ((5,), ((2, 2),), ("wrap",)),
            ),
            (
                ((5,), 2, "reflect"),
                ((5,), ((2, 2),), ("reflect",)),
            ),
            (
                ((5,), 2, "symmetric"),
                ((5,), ((2, 2),), ("symmetric",)),
            ),
            (
                ((5,), 2, "edge"),
                ((5,), ((2, 2),), ("edge",)),
            ),
            # ND, uni-mode -------------------------------------
            (
                ((5, 3, 4), 2, "constant"),
                ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("constant", "constant", "constant")),
            ),
            (
                ((5, 3, 4), 2, "wrap"),
                ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("wrap", "wrap", "wrap")),
            ),
            (
                ((5, 3, 4), 2, "reflect"),
                ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("reflect", "reflect", "reflect")),
            ),
            (
                ((5, 3, 4), 2, "symmetric"),
                ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("symmetric", "symmetric", "symmetric")),
            ),
            (
                ((5, 3, 4), 2, "edge"),
                ((5, 3, 4), ((2, 2), (2, 2), (2, 2)), ("edge", "edge", "edge")),
            ),
            # ND, multi-mode -----------------------------------
            (
                ((5, 3, 4), (2, 1, 3), ("constant", "edge", "wrap")),
                ((5, 3, 4), ((2, 2), (1, 1), (3, 3)), ("constant", "edge", "wrap")),
            ),
            (
                ((5, 3, 4), ((0, 2), (1, 3), (3, 2)), ("constant", "edge", "wrap")),
                ((5, 3, 4), ((0, 2), (1, 3), (3, 2)), ("constant", "edge", "wrap")),
            ),
        ]
    )
    def _spec(self, request):
        # (arg_shape, pad_width, mode) configs to test.
        # * `request.param[0]` corresponds to raw inputs users provide to Pad().
        # * `request.param[1]` corresponds to their ground-truth canonical parameterization.
        return request.param

    @pytest.fixture
    def arg_shape(self, _spec):  # canonical representation
        arg_shape, _, _ = _spec[1]
        return arg_shape

    @pytest.fixture
    def pad_width(self, _spec):  # canonical representation
        _, pad_width, _ = _spec[1]
        return pad_width

    @pytest.fixture
    def mode(self, _spec):  # canonical representation
        _, _, mode = _spec[1]
        return mode

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        arg_shape, pad_width, mode = _spec[0]  # user-provided form
        op = pxl.Pad(
            arg_shape=arg_shape,
            pad_width=pad_width,
            mode=mode,
        )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape, pad_width) -> pxt.OpShape:
        dim = np.prod(arg_shape)

        pad_shape = []
        for N, (lhs, rhs) in zip(arg_shape, pad_width):
            p = N + (lhs + rhs)
            pad_shape.append(p)
        codim = np.prod(pad_shape)

        return (codim, dim)

    @pytest.fixture
    def data_apply(self, arg_shape, pad_width, mode) -> conftest.DataLike:
        arr = self._random_array(arg_shape)
        if len(set(mode)) == 1:  # uni-mode
            out = np.pad(
                array=arr,
                pad_width=pad_width,
                mode=mode[0],
            )
        else:  # multi-mode
            N_dim = len(arg_shape)
            out = arr
            for i in range(N_dim):
                p = [(0, 0)] * N_dim
                p[i] = pad_width[i]
                out = np.pad(
                    array=out,
                    pad_width=p,
                    mode=mode[i],
                )
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )
