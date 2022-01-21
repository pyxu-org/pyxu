import collections.abc as cabc
import inspect
import types

import pytest

import pycsou.abc.operator as pyco
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def func_name() -> str:
    """
    Returns
    -------
    up_fname: str
        Name of the function which called `func_name()`.

    Example
    -------
    >>> def f() -> str:
    ...     return func_name()
    ...
    ... f()  # -> 'f'
    """
    my_frame = inspect.currentframe()
    up_frame = inspect.getouterframes(my_frame)[1].frame
    up_finfo = inspect.getframeinfo(up_frame)
    up_fname = up_finfo.function
    return up_fname


def check_signature() -> bool:
    pass


class MapT:
    disable_test: cabc.Collection[str] = frozenset()

    @pytest.fixture
    def op(self) -> pyco.Map:
        # override in subclass to instantiate the object to test.
        raise NotImplementedError

    @pytest.fixture(params=pycd.supported_array_modules())
    def xp(self, request) -> types.ModuleType:
        # override in subclass if numeric methods are to be tested on a subset of array backends.
        return request.param

    # -------------------------------------------------------------------------
    @pytest.fixture
    def data_shape(self) -> pyct.Shape:
        # override in subclass with the shape of op.
        # Don't return `op.shape`: hard-code what you are expecting.
        raise NotImplementedError

    def test_io_shape(self, op, data_shape):
        if func_name() not in disable_test:
            assert op.shape == data_shape

    def test_io_dim(self, op, data_shape):
        if func_name() not in disable_test:
            assert op.dim == data_shape[1]

    def test_io_codim(self, op, data_shape):
        if func_name() not in disable_test:
            assert op.codim == data_shape[0]

    # -------------------------------------------------------------------------
    # TODO: test apply/lipschitz

    # -------------------------------------------------------------------------
    # TODO: test squeeze()
    # TODO: test specialize()
