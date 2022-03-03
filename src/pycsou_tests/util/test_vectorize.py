import inspect

import numpy as np
import pytest

import pycsou.util as pycu


class TestVectorize:
    @pytest.fixture(
        params=[
            lambda x: np.array(  # 1D only
                x.sum(keepdims=True) + 1,
                dtype=np.half,
            ),
            lambda x, y: np.array(  # 1D only, multi-parameter
                x.sum(keepdims=True) + y,
                dtype=np.half,
            ),
            lambda x, y=1: np.array(  # 1D only, multi-parameter (with defaults)
                x.sum(keepdims=True) + y,
                dtype=np.half,
            ),
            lambda x: np.array(  # already has desired ND behaviour
                x.sum(axis=-1, keepdims=True) + 1,
                dtype=np.half,
            ),
        ]
    )
    def func(self, request):
        return request.param

    @pytest.fixture
    def vfunc(self, func):
        return pycu.vectorize("x")(func)

    @pytest.fixture
    def data_func(self, func):
        sig = inspect.Signature.from_callable(func)
        data = dict(
            in_=dict(x=np.arange(5)),
            out=np.array([11]),
        )
        if "y" in sig.parameters:
            data["in_"].update(y=1)
        return data

    def test_1d(self, vfunc, data_func):
        out_gt = data_func["out"]

        in_ = data_func["in_"]
        out = vfunc(**in_)

        assert out.ndim == 1
        assert np.allclose(out, out_gt)

    def test_nd(self, vfunc, data_func):
        sh_extra = (2, 1)  # prepend input/output shape by this amount.

        out_gt = data_func["out"]
        out_gt = np.broadcast_to(out_gt, (*sh_extra, out_gt.shape[-1]))

        in_ = data_func["in_"]
        in_["x"] = np.broadcast_to(in_["x"], (*sh_extra, *in_["x"].shape))
        out = vfunc(**in_)

        assert out.ndim == out_gt.ndim
        assert np.allclose(out, out_gt)

    def test_precision(self, func, vfunc, data_func):
        # decorated function should have same output dtype as base function.
        in_ = data_func["in_"]
        out_gt = data_func["out"]

        out_f = func(**in_)
        out_vf = vfunc(**in_)

        assert out_f.dtype == out_vf.dtype
