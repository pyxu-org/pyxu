import collections.abc as cabc
import inspect
import types

import pytest

import pycsou.util.deps as pycd


@pytest.fixture(params=pycd.supported_array_modules())
def xp(request) -> types.ModuleType:
    return request.param


class DisableTestMixin:
    disable_test: cabc.Set[str] = frozenset()

    # What this class does: ability to disab
    # Defines a special method `_skip_if_disabled()`, which

    def _skip_if_disabled(self):
        # Get name of function which invoked me.
        my_frame = inspect.currentframe()
        up_frame = inspect.getouterframes(my_frame)[1].frame
        up_finfo = inspect.getframeinfo(up_frame)
        up_fname = up_finfo.function
        if up_fname in self.disable_test:
            pytest.skip("disabled test")
