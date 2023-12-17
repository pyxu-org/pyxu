import dask.array as da
import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.util as pxu


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
