import itertools

import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestDiagonalOp(conftest.PosDefOpT):
    @pytest.fixture(
        params=[
            (5,),
            (5, 3),
            (5, 3, 4, 2),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture(params=[True, False])
    def broadcasted(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True, False])
    def posdef(self, request) -> bool:
        return request.param

    @pytest.fixture
    def vec(self, dim_shape, broadcasted, posdef) -> pxt.NDArray:
        # NUMPY version of `vec` supplied to DiagonalOp()
        v = self._random_array(dim_shape)

        if posdef:
            v -= v.min() + 1e-3  # guaranteed to be positive

        if broadcasted:
            ax = v.ndim // 2
            v = v.sum(axis=ax, keepdims=True)

        return v

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, vec, broadcasted, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        xp = ndi.module()

        v = xp.array(vec, dtype=width.value)
        if broadcasted:
            op = pxo.DiagonalOp(
                vec=v,
                dim_shape=dim_shape,
                enable_warnings=False,
            )
        else:
            op = pxo.DiagonalOp(
                vec=v,
                # omit dim_shape to see if inference correct
                enable_warnings=False,
            )

        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape, vec) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x * vec
        return dict(
            in_=dict(arr=x),
            out=y,
        )

    # Tests -------------------------------------------------------------------
    def test_math_posdef(self, op, xp, width, posdef):
        if posdef:
            super().test_math_posdef(op, xp, width)
        else:
            pytest.skip("disabled since operator is not positive-definite.")
