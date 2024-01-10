import collections.abc as cabc
import inspect

import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.util as pxu


class TestVectorize:
    @pytest.fixture(
        params=[
            # (func, dim_shape, codim_shape) triplets.
            # [dim_shape / codim_shape are in user-provided form.]
            (  # 1D-func
                lambda x: (x.sum(keepdims=True) + 1).astype(np.half),
                5,
                1,
            ),
            (  # 1D-func, multi-parameter
                lambda x, y: (x.sum(keepdims=True) + y).astype(np.half),
                5,
                1,
            ),
            (  # 1D-func, multi-parameter (with defaults)
                lambda x, y=1: (x.sum(keepdims=True) + y).astype(np.half),
                5,
                1,
            ),
            (  # already has ND behaviour
                lambda x: (x.sum(axis=(0, 2), keepdims=True)).astype(np.half),
                (5, 3, 4),
                (1, 3, 1),
            ),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def func(self, _spec) -> cabc.Callable:
        return _spec[0]

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        return pxu.as_canonical_shape(_spec[1])

    @pytest.fixture
    def codim_shape(self, _spec) -> pxt.NDArrayShape:
        return pxu.as_canonical_shape(_spec[2])

    @pytest.fixture
    def vfunc(self, _spec) -> cabc.Callable:
        # vectorized function
        func, dim_shape, codim_shape = _spec
        decorate = pxu.vectorize(
            i="x",
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        return decorate(func)

    @pytest.fixture
    def data_func(self, func, dim_shape):
        x = np.arange(np.prod(dim_shape)).reshape(dim_shape)

        sig = inspect.Signature.from_callable(func)
        if "y" in sig.parameters:
            y = 0.5
            data = dict(
                in_=dict(x=x, y=y),
                out=func(x, y),
            )
        else:
            data = dict(
                in_=dict(x=x),
                out=func(x),
            )
        return data

    def test_1d(self, vfunc, data_func, codim_shape):
        # No stacking dimensions -> rank[out] == codim_rank
        out_gt = data_func["out"]

        in_ = data_func["in_"]
        out = vfunc(**in_)

        assert out.ndim == len(codim_shape)
        assert np.allclose(out, out_gt)

    def test_nd(self, vfunc, data_func, codim_shape):
        sh_extra = (3, 2, 1)  # prepend input/output shape by this amount.

        out_gt = data_func["out"]
        out_gt = np.broadcast_to(out_gt, (*sh_extra, *codim_shape))

        in_ = data_func["in_"]
        in_["x"] = np.broadcast_to(in_["x"], (*sh_extra, *in_["x"].shape))
        out = vfunc(**in_)

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_precision(self, func, vfunc, data_func):
        # decorated function should have same output dtype as base function.
        in_ = data_func["in_"]

        out_f = func(**in_)
        out_vf = vfunc(**in_)

        assert out_f.dtype == out_vf.dtype

    def test_backend(self, vfunc, data_func, xp):
        # returned array from decorated function should have same type as input array.
        in_ = data_func["in_"]
        in_["x"] = xp.array(in_["x"])
        out = vfunc(**in_)

        assert type(out) == type(in_["x"])  # noqa: E721
