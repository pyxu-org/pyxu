import contextlib

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu


class TestCopyIfUnsafe:
    @pytest.fixture
    def data(self, xp) -> pxt.NDArray:
        # Raw data which owns its memory.
        x = np.arange(50**3).reshape(50, 50, 50)
        y = xp.array(x, dtype=x.dtype)
        return y

    def test_no_copy(self, data):
        out = pxu.copy_if_unsafe(data)
        assert out is data

    @pytest.mark.parametrize("mode", ["read_only", "view"])
    def test_copy(self, data, mode):
        xp = pxu.get_array_module(data)
        if xp == pxd.NDArrayInfo.DASK.module():
            pytest.skip("Unsupported config.")

        data = data.copy()
        if mode == "read_only":
            data.flags.writeable = False
        elif mode == "view":
            data = data.view()

        out = pxu.copy_if_unsafe(data)
        assert out.flags.owndata
        assert out.shape == data.shape
        assert xp.allclose(out, data)


class TestReadOnly:
    @pytest.fixture(
        params=[
            (slice(None), slice(None), slice(None)),  # multi-dim, contiguous
            (slice(None), slice(1, None, 2), slice(None, None, -1)),  # multi-dim view
        ]
    )
    def data(self, request, xp) -> pxt.NDArray:
        # Raw data, potentially non-contiguous given provided selector
        x = np.arange(50**3).reshape(50, 50, 50)
        y = xp.array(x, dtype=x.dtype)

        selector = request.param
        z = y[selector]
        return z

    def test_DASK_transparent(self, data, xp):
        # DASK arrays go through un-modified
        if xp != pxd.NDArrayInfo.DASK.module():
            pytest.skip("Unsupported config.")

        out = pxu.read_only(data)
        assert out is data

    def test_NUMCUPY_readonly(self, data, xp):
        # NUMPY/CUPY arrays are read-only
        if xp == pxd.NDArrayInfo.DASK.module():
            pytest.skip("Unsupported config.")

        out = pxu.read_only(data)

        if hasattr(out.flags, "writeable"):
            assert not out.flags.writeable
        assert not out.flags.owndata
        assert out.shape == data.shape
        assert xp.allclose(out, data)


class TestImportModule:
    def test_successful_import(self):
        # Loading a module known to exist must work.
        xp = pxu.import_module("numpy")
        xp_gt = pxd.NDArrayInfo.NUMPY.module()

        assert xp == xp_gt

    @pytest.mark.parametrize("fail_on_error", [True, False])
    def test_unsuccessful_import(self, fail_on_error):
        # Raise error depending if `fail_on_error` flag.
        if fail_on_error:
            ctx = pytest.raises(ModuleNotFoundError)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            pxu.import_module(
                "my_inexistent_module",
                fail_on_error,
            )


class TestParseParams:
    @pytest.mark.parametrize(
        ["args", "kwargs", "gt"],
        [
            # Provide all parameters ----------------------
            ((1, 2), dict(), dict(x=1, y=2)),
            ((1,), dict(y=2), dict(x=1, y=2)),
            ((), dict(x=1, y=2), dict(x=1, y=2)),
            # Now omit specifying y -----------------------
            ((1,), dict(), dict(x=1, y=3)),
            ((), dict(x=1), dict(x=1, y=3)),
        ],
    )
    def test_no_default_args(self, args, kwargs, gt):
        f = lambda x, y=3: None

        params = pxu.parse_params(f, *args, **kwargs)
        assert params == gt
