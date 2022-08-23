import itertools

import numpy as np
import pytest

import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


# We disable PrecisionWarnings since DiagonalOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.PrecisionWarning")
class TestDiagonalOp(conftest.PosDefOpT):
    @pytest.fixture
    def dim(self):
        return 20

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture(params=range(5))
    def vec(self, dim, request):
        v = self._random_array((dim,), seed=request.param)
        return v

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, vec, request):
        ndi = request.param[0]
        if (xp := ndi.module()) is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        width = request.param[1]

        vec = xp.array(vec, dtype=width.value)
        op = pyco.DiagonalOp(vec=vec)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, vec):
        arr = 15 + np.arange(vec.size)
        out = vec * arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    def test_math_eig(self, _op_eig, vec):
        if np.any(pycu.compute(vec < 0)):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_eig(_op_eig)

    def test_math_posdef(self, op, xp, width, vec):
        if np.any(pycu.compute(vec < 0)):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op, xp, width)
