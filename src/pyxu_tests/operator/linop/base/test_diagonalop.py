import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


# We disable PrecisionWarnings since DiagonalOp() is not precision-agnostic, but the outputs
# computed must still be valid.
@pytest.mark.filterwarnings("ignore::pyxu.info.warning.PrecisionWarning")
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
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, vec, request):
        ndi = request.param[0]
        self._skip_if_unsupported(ndi)
        xp = ndi.module()
        width = request.param[1]

        vec = xp.array(vec, dtype=width.value)
        op = pxo.DiagonalOp(vec=vec)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, vec):
        arr = 15 + np.arange(vec.size)
        out = vec * arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    def test_math_posdef(self, op, xp, width, vec):
        if np.any(pxu.compute(vec < 0)):
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op, xp, width)
