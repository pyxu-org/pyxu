import inspect

import numpy as np
import pytest

import pyxu.util as pxu


class TestInferSumShape:
    @pytest.mark.parametrize(
        ["sh1", "sh2", "sh3"],
        [
            ((5, 3), (5, 3), (5, 3)),  # same shape
            ((5, 3), (1, 3), (5, 3)),  # codomain broadcast
            ((1, 3), (5, 3), (5, 3)),  # codomain broadcast (commutativity)
        ],
    )
    def test_valid(self, sh1, sh2, sh3):
        assert pxu.infer_sum_shape(sh1, sh2) == sh3

    @pytest.mark.parametrize(
        ["sh1", "sh2"],
        [
            ((5, 1), (5, 3)),  # domain broadcast
            ((5, 2), (5, 3)),  # domain broadcast
            ((5, 3), (2, 3)),  # codomain broadcast
        ],
    )
    def test_invalid(self, sh1, sh2):
        with pytest.raises(ValueError):
            pxu.infer_sum_shape(sh1, sh2)


class TestInferCompositionShape:
    @pytest.mark.parametrize(
        ["sh1", "sh2", "sh3"],
        [
            ((5, 3), (3, 4), (5, 4)),
        ],
    )
    def test_valid(self, sh1, sh2, sh3):
        assert pxu.infer_composition_shape(sh1, sh2) == sh3

    @pytest.mark.parametrize(
        ["sh1", "sh2"],
        [
            ((5, 3), (1, 4)),
        ],
    )
    def test_invalid(self, sh1, sh2):
        with pytest.raises(ValueError):
            pxu.infer_composition_shape(sh1, sh2)


class TestVectorize:
    @pytest.fixture(
        params=[
            lambda x: (x.sum(keepdims=True) + 1).astype(np.half),  # 1D only
            lambda x, y: (x.sum(keepdims=True) + y).astype(np.half),  # 1D only, multi-parameter
            lambda x, y=1: (x.sum(keepdims=True) + y).astype(np.half),  # 1D only, multi-parameter (with defaults)
            lambda x: (x.sum(axis=-1, keepdims=True) + 1).astype(np.half),  # already has desired ND behaviour
        ]
    )
    def func(self, request):
        return request.param

    @pytest.fixture(
        params=[  # (method, codim)  [codim-size when specified comes from `data_func` fixture.]
            ("scan", None),
            ("scan", 10),  # codim irrelevant; bogus value to see if unused
            ("parallel", 5),
            ("scan_dask", 5),
        ]
    )
    def vfunc(self, func, request):
        method, codim = request.param
        decorate = pxu.vectorize(i="x", method=method, codim=codim)
        return decorate(func)

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

        out_f = func(**in_)
        out_vf = vfunc(**in_)

        assert out_f.dtype == out_vf.dtype

    def test_backend(self, vfunc, data_func, xp):
        # returned array from decorated function should have same type as input array.
        in_ = data_func["in_"]
        in_["x"] = xp.array(in_["x"])
        out = vfunc(**in_)

        assert type(out) == type(in_["x"])  # noqa: E721
