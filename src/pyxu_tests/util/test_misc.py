import dask.array as da
import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
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


class TestCompute:
    @pytest.fixture(
        params=[
            1,
            [1, 2, 3],
            np.arange(5),
            da.arange(5),
        ]
    )
    def single_input(self, request):
        return request.param

    def equal(self, x, y):
        if any(type(_) in pxd.supported_array_types() for _ in [x, y]):
            return np.allclose(x, y)
        else:
            return x == y

    @pytest.fixture(params=["compute", "persist"])
    def mode(self, request):
        return request.param

    def test_single_inputs(self, single_input, mode):
        cargs = pxu.compute(single_input, mode=mode)
        assert self.equal(cargs, single_input)

    def test_multi_inputs(self, mode):
        x = da.arange(5)
        y = x + 1
        i_args = (1, [1, 2, 3], np.arange(5), x, y)
        o_args = (1, [1, 2, 3], np.arange(5), np.arange(5), np.arange(1, 6))
        cargs = pxu.compute(*i_args, mode=mode)

        assert len(cargs) == len(o_args)
        for c, o in zip(cargs, o_args):
            assert self.equal(c, o)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            pxu.compute(1, mode="test")

    def test_kwargs_does_not_fail(self):
        x = 1
        pxu.compute(x, optimize_graph=False)


class TestToNumPy:
    @pytest.fixture
    def arr(self) -> pxt.NDArray:
        return np.arange(-5, 5)

    @pytest.fixture
    def _arr(self, arr, xp):
        return xp.array(arr, dtype=arr.dtype)

    def test_backend_change(self, _arr):
        N = pxd.NDArrayInfo
        np_arr = pxu.to_NUMPY(_arr)

        assert N.from_obj(np_arr) == N.NUMPY
        if N.from_obj(_arr) == N.NUMPY:
            assert _arr is np_arr


class TestCopyIfUnsafe:
    @pytest.fixture(
        params=[
            np.r_[1],
            np.ones((5,)),
            np.ones((5, 3, 4)),
        ]
    )
    def x(self, request):
        return request.param

    def test_no_copy(self, xp, x):
        x = xp.array(x)

        y = pxu.copy_if_unsafe(x)
        assert x is y

    @pytest.mark.parametrize("xp", set(pxd.supported_array_modules()) - {da})
    @pytest.mark.parametrize("mode", ["read_only", "view"])
    def test_copy(self, xp, x, mode):
        x = xp.array(x)
        if mode == "read_only":
            x.flags.writeable = False
        elif mode == "view":
            x = x.view()

        y = pxu.copy_if_unsafe(x)
        assert y.flags.owndata
        assert y.shape == x.shape
        assert xp.allclose(y, x)


class TestReadOnly:
    @pytest.fixture(
        params=[
            np.ones((1,)),  # contiguous
            np.ones((5,)),  # contiguous
            np.ones((5, 3, 4)),  # multi-dim, contiguous
            np.ones((5, 3, 4))[:, ::-1],  # multi-dim, view
        ]
    )
    def x(self, request):
        return request.param

    def test_transparent(self, x):
        x = da.array(x)
        y = pxu.read_only(x)
        assert y is x

    @pytest.mark.parametrize("xp", set(pxd.supported_array_modules()) - {da})
    def test_readonly(self, xp, x):
        x = xp.array(x)
        y = pxu.read_only(x)

        if hasattr(y.flags, "writeable"):
            assert not y.flags.writeable
        assert not y.flags.owndata
        assert y.shape == x.shape
        assert xp.allclose(y, x)
