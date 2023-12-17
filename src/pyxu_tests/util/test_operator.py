import collections.abc as cabc
import inspect

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu


class TestAsCanonicalShape:
    @pytest.mark.parametrize(
        ["x", "gt"],
        [
            # int-valued types ------------------
            (1, (1,)),
            (np.int8(1), (1,)),
            # int-valued sequences --------------
            ((-1,), (-1,)),
            ((-1, np.int16(23)), (-1, 23)),
            ([2, 3], (2, 3)),  # other sequence types
        ],
    )
    def test_valid_input(self, x, gt):
        y = pxu.as_canonical_shape(x)
        assert y == gt

    @pytest.mark.parametrize(
        "x",
        [
            # float-valued types --------------------
            1.0,
            np.single(5),
            # float-valued sequences ----------------
            (2.1,),
            (2.1, np.single(-3)),
            [2.1, -1],
        ],
    )
    def test_invalid_input(self, x):
        with pytest.raises(AssertionError):
            pxu.as_canonical_shape(x)


class TestAsCanonicalAxes:
    @pytest.mark.parametrize(
        ["x", "gt"],
        [
            # TestAsCanonicalShape verifies in/out types,
            # so we concentrate on range of valid axes only.
            (-4, (0,)),
            (-3, (1,)),
            (-2, (2,)),
            (-1, (3,)),
            (0, (0,)),
            (1, (1,)),
            (2, (2,)),
            (3, (3,)),
        ],
    )
    def test_valid_input(self, x, gt):
        y = pxu.as_canonical_axes(x, rank=4)
        assert y == gt

    @pytest.mark.parametrize("x", [-5, 4])
    def test_invalid_input(self, x):
        with pytest.raises(AssertionError):
            pxu.as_canonical_axes(x, rank=4)


class TestVectorize:
    # Fixtures ----------------------------------------------------------------
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

    # Tests -------------------------------------------------------------------
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

    def test_chunk_preserved(self, vfunc, data_func, xp):
        # DASK-inputs only: chunk size in stack-dimensions preserved.
        if xp != pxd.NDArrayInfo.DASK.module():
            pytest.skip("Unsupported for non-DASK inputs.")

        sh_extra = (7, 7, 7)
        in_ = data_func["in_"]
        in_["x"] = xp.broadcast_to(
            in_["x"],
            shape=(*sh_extra, *in_["x"].shape),
        )

        sh_chunks = (2, 3, 4)  # stack chunks
        cr_chunks = ("auto",) * len(in_["x"].shape)  # core-chunks
        in_["x"] = in_["x"].rechunk(sh_chunks + cr_chunks)

        out = vfunc(**in_)
        sh_rank = len(sh_extra)
        assert out.chunks[:sh_rank] == in_["x"].chunks[:sh_rank]
