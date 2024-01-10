import itertools

import pytest

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class Permutation(pxa.UnitOp):
    # f: \bR^{M1,...,MD} -> \bR^{M1,...,MD}
    #      x             -> x[::-1]
    #                       (reverse order of each dimension)
    def __init__(self, dim_shape: pxt.NDArrayShape):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,
        )

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        selector = (slice(None, None, -1),) * self.dim_rank
        return pxu.read_only(arr[..., *selector])

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        selector = (slice(None, None, -1),) * self.codim_rank
        return pxu.read_only(arr[..., *selector])


class TestPermutation(conftest.UnitOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = Permutation(dim_shape)
        return op, ndi, width

    @pytest.fixture(
        params=[
            (1,),
            (5,),
            (5, 3, 4),
        ]
    )
    def dim_shape(self, request) -> pxt.NDArrayShape:
        return request.param

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture(params=[0, 17, 93])
    def data_apply(self, dim_shape, request) -> conftest.DataLike:
        seed = request.param

        x = self._random_array(dim_shape, seed=seed)
        selector = (slice(None, None, -1),) * len(dim_shape)
        y = x[selector]

        return dict(
            in_=dict(arr=x),
            out=y,
        )
