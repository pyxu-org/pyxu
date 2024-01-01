import itertools

import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestSubSample(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     indices (user-specified; tuple[whatever] form),
            #     indices (canonical; tuple[whatever indexes NP arrays correctly] form),
            #     codim_shape (canonical),
            # 1D cases ----------------------------------------------
            (10, (10,), (3,), ([3],), (1,)),
            (10, (10,), (-3,), ([7],), (1,)),
            (10, (10,), (slice(None, None, 4),), ([0, 4, 8],), (3,)),
            (10, (10,), (slice(9, 2, -2),), ([9, 7, 5, 3],), (4,)),
            (10, (10,), ([-2, 9],), ([8, 9],), (2,)),
            (5, (5,), ([True, False, False, True, True],), ([0, 3, 4],), (3,)),
            # 2D cases (omit 2nd axis specifier) --------------------
            ((10, 10), (10, 10), (3,), ([3], slice(None)), (1, 10)),
            ((10, 10), (10, 10), (-3,), ([7], slice(None)), (1, 10)),
            ((10, 10), (10, 10), (slice(None, None, 4),), ([0, 4, 8], slice(None)), (3, 10)),
            ((10, 10), (10, 10), (slice(9, 2, -2),), ([9, 7, 5, 3], slice(None)), (4, 10)),
            ((10, 10), (10, 10), ([-2, 9],), ([8, 9], slice(None)), (2, 10)),
            ((5, 5), (5, 5), ([True, False, False, True, True],), ([0, 3, 4], slice(None)), (3, 5)),
            # 2D cases (select 2nd axis only) -----------------------
            ((10, 10), (10, 10), (slice(None), 3), (slice(None), [3]), (10, 1)),
            (
                (10, 10),
                (10, 10),
                (
                    slice(None),
                    -3,
                ),
                (slice(None), [7]),
                (10, 1),
            ),
            (
                (10, 10),
                (10, 10),
                (
                    slice(None),
                    slice(None, None, 4),
                ),
                (slice(None), [0, 4, 8]),
                (10, 3),
            ),
            (
                (10, 10),
                (10, 10),
                (
                    slice(None),
                    slice(9, 2, -2),
                ),
                (slice(None), [9, 7, 5, 3]),
                (10, 4),
            ),
            (
                (10, 10),
                (10, 10),
                (
                    slice(None),
                    [-2, 9],
                ),
                (slice(None), [8, 9]),
                (10, 2),
            ),
            (
                (5, 5),
                (5, 5),
                (
                    slice(None),
                    [True, False, False, True, True],
                ),
                (slice(None), [0, 3, 4]),
                (5, 3),
            ),
            # 2D cases (select both axes) ---------------------------
            ((10, 10), (10, 10), (3, 4), (slice(3, 4), slice(4, 5)), (1, 1)),
            ((10, 10), (10, 10), (-3, -1), (slice(7, 8), slice(9, 10)), (1, 1)),
            ((10, 10), (10, 10), (slice(None, None, 4), 0), ([0, 4, 8], slice(0, 1)), (3, 1)),
            ((10, 10), (10, 10), (slice(9, 2, -2), -3), ([9, 7, 5, 3], slice(7, 8)), (4, 1)),
            ((10, 10), (10, 10), (slice(9, 2, -2), slice(3, 7)), ([9, 7, 5, 3], slice(3, 7)), (4, 4)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        dim_shape, indices = _spec[0], _spec[2]  # user-specified dim_shape/indices
        op = pxo.SubSample(
            dim_shape,
            *indices,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, _spec) -> pxt.NDArrayShape:
        return _spec[4]

    @pytest.fixture
    def indices(self, _spec) -> tuple:
        # canonical axial indices
        # Must be len(dim_shape)
        return _spec[3]

    @pytest.fixture
    def data_apply(self, dim_shape, indices) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x[indices]
        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestTrim(conftest.LinOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     trim_width (user-specified),
            #     selector (canonical tuple[slice] form),
            #     codim_shape (canonical),
            # 1D cases ----------------------------------------------
            (10, (10,), 2, (slice(2, -2),), (6,)),
            (10, (10,), (2,), (slice(2, -2),), (6,)),
            (10, (10,), ((2, 4),), (slice(2, -4),), (4,)),
            # 2D cases ----------------------------------------------
            ((10, 10), (10, 10), 2, (slice(2, -2), slice(2, -2)), (6, 6)),
            ((10, 10), (10, 10), (2, 3), (slice(2, -2), slice(3, -3)), (6, 4)),
            ((10, 10), (10, 10), ((2, 3), (1, 2)), (slice(2, -3), slice(1, -2)), (5, 7)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        dim_shape, trim_width = _spec[0], _spec[2]  # user-specified dim_shape/trim_width
        op = pxo.Trim(
            dim_shape=dim_shape,
            trim_width=trim_width,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical codim_shape
        return _spec[4]

    @pytest.fixture
    def selector(self, _spec) -> tuple[slice]:
        # canonical axial indices
        # Must be len(dim_shape)
        return _spec[3]

    @pytest.fixture
    def data_apply(self, dim_shape, selector) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x[selector]
        return dict(
            in_=dict(arr=x),
            out=y,
        )
