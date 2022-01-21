import dask.array as da
import numpy as np
import pytest

import pycsou.util as pycu
import pycsou.util.deps as pycd


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
        if any(type(_) in pycd.supported_array_types() for _ in [x, y]):
            return np.allclose(x, y)
        else:
            return x == y

    @pytest.fixture(params=["compute", "persist"])
    def mode(self, request):
        return request.param

    def test_single_inputs(self, single_input, mode):
        cargs = pycu.compute(single_input, mode=mode)
        assert self.equal(cargs, single_input)

    def test_multi_inputs(self, mode):
        x = da.arange(5)
        y = x + 1
        i_args = (1, [1, 2, 3], np.arange(5), x, y)
        o_args = (1, [1, 2, 3], np.arange(5), np.arange(5), np.arange(1, 6))
        cargs = pycu.compute(*i_args, mode=mode)

        assert len(cargs) == len(o_args)
        for c, o in zip(cargs, o_args):
            assert self.equal(c, o)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            pycu.compute(1, mode="test")

    def test_kwargs_does_not_fail(self):
        x = 1
        pycu.compute(x, optimize_graph=False)
