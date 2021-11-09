import types

import pytest

import pycsou.util.deps as pycd


@pytest.fixture(params=pycd.supported_array_modules())
def xp(request) -> types.ModuleType:
    return request.param
