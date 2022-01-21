import numpy as np
import pytest

import pycsou.util as pycu


class TestGetArrayModule:
    def test_array(self, xp):
        x = xp.arange(5)
        assert pycu.get_array_module(x) is xp

    @pytest.mark.parametrize(
        ["obj", "fallback", "fail"],
        [
            [None, None, True],
            [None, np, False],
            [1, None, True],
            [1, np, False],
        ],
    )
    def test_fallback(self, obj, fallback, fail):
        # object is not an array type, so fail or return provided fallback
        if not fail:
            assert pycu.get_array_module(obj, fallback) is fallback
        else:
            with pytest.raises(ValueError):
                assert pycu.get_array_module(obj, fallback)
