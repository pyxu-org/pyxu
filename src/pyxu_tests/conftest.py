import collections.abc as cabc
import inspect
import types
import typing as typ

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu


@pytest.fixture(params=pxd.supported_array_modules())
def xp(request) -> types.ModuleType:
    return request.param


@pytest.fixture(params=pxrt.Width)
def width(request) -> pxrt.Width:
    return request.param


def flaky(func: cabc.Callable, args: dict, condition: bool, reason: str):
    # XFAIL if func(**args) fails when `condition` satisfied.
    # This function is required when pytest.mark.xfail() cannot be used.
    # (Reason: `condition` only available inside test body.)
    try:
        out = func(**args)
        return out
    except Exception:
        if condition:
            pytest.xfail(reason)
        else:
            raise


def isclose(
    a: typ.Union[pxt.Real, pxt.NDArray],
    b: typ.Union[pxt.Real, pxt.NDArray],
    as_dtype: pxt.DType,
) -> pxt.NDArray:
    """
    Equivalent of `xp.isclose`, but where atol is automatically chosen based on `as_dtype`.

    This function always returns a computed array, i.e. NumPy/CuPy output.
    """
    atol = {
        np.dtype(np.half): 3e-2,  # former pxrt.Width.HALF
        pxrt.Width.SINGLE.value: 2e-4,
        pxrt.CWidth.SINGLE.value: 2e-4,
        pxrt.Width.DOUBLE.value: 1e-8,
        pxrt.CWidth.DOUBLE.value: 1e-8,
    }
    # Numbers obtained by:
    # * \sum_{k >= (p+1)//2} 2^{-k}, where p=<number of mantissa bits>; then
    # * round up value to 3 significant decimal digits.
    # N_mantissa = [10, 23, 52, 112] for [half, single, double, quad] respectively.

    prec = atol.get(as_dtype, pxrt.Width.DOUBLE.value)  # default only should occur for integer types
    eq = np.isclose(pxu.compute(a), pxu.compute(b), atol=prec)
    return eq


def allclose(
    a: pxt.NDArray,
    b: pxt.NDArray,
    as_dtype: pxt.DType,
) -> bool:
    """
    Equivalent of `all(isclose)`, but where atol is automatically chosen based on `as_dtype`.
    """
    return bool(np.all(isclose(a, b, as_dtype)))


def less_equal(
    a: pxt.NDArray,
    b: pxt.NDArray,
    as_dtype: pxt.DType,
) -> pxt.NDArray:
    """
    Equivalent of `a <= b`, but where equality tests are done at a chosen numerical precision.

    This function always returns a computed array, i.e. NumPy/CuPy output.
    """
    x = pxu.compute(a <= b)
    y = isclose(a, b, as_dtype)
    return x | y


def sanitize(x, default):
    if x is not None:
        return x
    else:
        return default


def chunk_array(x: pxt.NDArray, complex_view: bool) -> pxt.NDArray:
    # Chunk DASK arrays to have (when possible) at least 2 chunks per axis.
    #
    # Parameters
    # ----------
    # complex_view: bool
    #     If True, `x` is assumed to be a (..., 2) array representing complex numbers.
    #     The final axis is not chunked in this case.
    ndi = pxd.NDArrayInfo.from_obj(x)
    if ndi == pxd.NDArrayInfo.DASK:
        chunks = {}
        for ax, sh in enumerate(x.shape):
            chunks[ax] = sh // 2 if sh > 1 else sh
        if complex_view:
            chunks[x.ndim - 1] = -1
        y = x.rechunk(chunks)
    else:
        y = x
    return y


class DisableTestMixin:
    """
    Disable certain tests based on user black-list.

    Example
    -------
    Consider the sample file below, where `test_1` and `test_3` should not be run.

    .. code-block:: python3

       # file: test_myclass.py

       import pytest
       import pyxu_tests.conftest as ct

       class TestMyClass(ct.DisableTestMixin):
           disable_test = {  # disable these tests
               "test_1",
               "test_3",
           }

           def test_1(self):
               self._skip_if_disabled()
               assert False  # should fail if not disabled

           def test_2(self):
               self._skip_if_disabled()
               assert True   # should pass

           def test_3(self):
               self._skip_if_disabled()
               assert False  # should fail if not disabled


    .. code-block:: bash

       $ pytest --quiet test_myclass.py

       s.s                           [100%]
       1 passed, 2 skipped in 0.59s             # -> test_[13]() were skipped/disabled.
    """

    disable_test: cabc.Set[str] = frozenset()

    def _skip_if_disabled(self):
        # Get name of function which invoked me.
        my_frame = inspect.currentframe()
        up_frame = inspect.getouterframes(my_frame)[1].frame
        up_finfo = inspect.getframeinfo(up_frame)
        up_fname = up_finfo.function
        if up_fname in self.disable_test:
            pytest.skip("disabled test")
