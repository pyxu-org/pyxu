import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class TestTransposeAxes(conftest.UnitOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     axes (user-specified),
            #     axes (canonical).
            # 1D cases --------------------
            (5, (5,), None, (0,)),
            (5, (5,), 0, (0,)),
            # 2D cases --------------------
            ((5, 3), (5, 3), None, (1, 0)),
            ((5, 3), (5, 3), (1, 0), (1, 0)),
            ((5, 3), (5, 3), (0, 1), (0, 1)),
            # 3D cases --------------------
            ((5, 3, 4), (5, 3, 4), None, (2, 1, 0)),
            ((5, 3, 4), (5, 3, 4), (0, 1, 2), (0, 1, 2)),
            ((5, 3, 4), (5, 3, 4), (1, 0, 2), (1, 0, 2)),
            ((5, 3, 4), (5, 3, 4), (2, 0, 1), (2, 0, 1)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, dim_shape, axes) -> pxt.NDArrayShape:
        sh = (dim_shape[ax] for ax in axes)
        return tuple(sh)

    @pytest.fixture
    def axes(self, _spec) -> pxt.NDArrayAxis:
        # canonical axes
        return _spec[3]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        dim_shape, axes = _spec[0], _spec[2]  # user-specified version
        ndi, width = request.param

        op = pxo.TransposeAxes(
            dim_shape=dim_shape,
            axes=axes,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape, axes) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x.transpose(axes)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestSqueezeAxes(conftest.UnitOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     axes (user-specified),
            #     axes (canonical).
            # 2D cases --------------------
            ((5, 3), (5, 3), None, ()),
            ((5, 1), (5, 1), None, (1,)),
            ((5, 1), (5, 1), 1, (1,)),
            ((1, 5), (1, 5), None, (0,)),
            ((1, 5), (1, 5), 0, (0,)),
            # 3D cases --------------------
            ((5, 3, 4), (5, 3, 4), None, ()),
            ((5, 1, 4), (5, 1, 4), None, (1,)),
            ((1, 3, 4), (1, 3, 4), 0, (0,)),
            ((1, 3, 1), (1, 3, 1), (0, 2), (0, 2)),
            ((1, 3, 1), (1, 3, 1), 2, (2,)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, dim_shape, axes) -> pxt.NDArrayShape:
        sh = []
        for ax in range(len(dim_shape)):
            if ax not in axes:
                codim = dim_shape[ax]
                sh.append(codim)
        return tuple(sh)

    @pytest.fixture
    def axes(self, _spec) -> pxt.NDArrayAxis:
        # canonical axes
        return _spec[3]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        dim_shape, axes = _spec[0], _spec[2]  # user-specified version
        ndi, width = request.param

        op = pxo.SqueezeAxes(
            dim_shape=dim_shape,
            axes=axes,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape, axes) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        if len(axes) > 0:
            y = x.squeeze(axes)
        else:
            y = x

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestReshapeAxes(conftest.UnitOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (user-specified),
            #     dim_shape (canonical),
            #     codim_shape (user-specified),
            #     codim_shape (canonical).
            # 1D cases --------------------
            (6, (6,), 6, (6,)),
            ((6,), (6,), (2, -1), (2, 3)),
            (6, (6,), (1, 2, -1), (1, 2, 3)),
            # 3D cases --------------------
            ((5, 3, 4), (5, 3, 4), 60, (60,)),
            ((5, 3, 4), (5, 3, 4), -1, (60,)),
            ((5, 3, 4), (5, 3, 4), (15, 4), (15, 4)),
            ((5, 3, 4), (5, 3, 4), (-1, 1, 4), (15, 1, 4)),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical dim_shape
        return _spec[1]

    @pytest.fixture
    def codim_shape(self, _spec) -> pxt.NDArrayShape:
        # canonical codim_shape
        return _spec[3]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        dim_shape, codim_shape = _spec[0], _spec[2]  # user-specified version
        ndi, width = request.param

        op = pxo.ReshapeAxes(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape, codim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x.reshape(codim_shape)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestBroadcastAxes(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, dim_shape, codim_shape, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        op = pxo.BroadcastAxes(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
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

    @pytest.fixture(params=["no_op", "grow_rank", "grow_size"])
    def codim_shape(self, dim_shape, request) -> pxt.NDArrayShape:
        op_type = request.param

        if op_type == "no_op":
            sh = ()
        elif op_type == "grow_rank":
            sh = (1, 1)
        elif op_type == "grow_size":
            sh = (7, 1, 3)
        else:
            raise NotImplementedError

        c_sh = sh + dim_shape
        return c_sh

    @pytest.fixture
    def data_apply(self, dim_shape, codim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = np.broadcast_to(x, codim_shape)

        return dict(
            in_=dict(arr=x),
            out=y,
        )


class TestRechunkAxes(conftest.UnitOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     dim_shape (canonical form),
            #     chunks (user-specified),
            #     chunks (canonical ground-truth),
            # 1D cases --------------------
            ((9,), {0: 2}, ((2, 2, 2, 2, 1),)),
            ((9,), {0: 3}, ((3, 3, 3),)),
            ((9,), {0: -1}, ((9,),)),
            # ((9,), {0: None}, ???),  # can't know chunk size beforehand
            # ((9,), {0: "auto"}, ???),  # can't know chunk size beforehand
            # 2D cases --------------------
            ((9, 9), {0: 3, -1: 2}, ((3, 3, 3), (2, 2, 2, 2, 1))),
            ((9, 9), {0: -1, -1: 2}, ((9,), (2, 2, 2, 2, 1))),
            # 3D cases --------------------
            ((9, 9, 9), {0: 4, -1: 3, -2: 2}, ((4, 4, 1), (2, 2, 2, 2, 1), (3, 3, 3))),
            ((9, 9, 9), {0: (2, 7), -1: (3, 5, 1), -2: -1}, ((2, 7), (9,), (3, 5, 1))),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # Canonical dim_shape
        return _spec[0]

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture
    def chunks(self, _spec) -> tuple[tuple[int]]:
        # Canonical core-dim chunks after rechunking.
        return _spec[2]

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        dim_shape, chunks = _spec[0], _spec[1]  # user-specified version
        ndi, width = request.param

        op = pxo.RechunkAxes(
            dim_shape=dim_shape,
            chunks=chunks,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(self, dim_shape) -> conftest.DataLike:
        x = self._random_array(dim_shape)
        y = x.copy()

        return dict(
            in_=dict(arr=x),
            out=y,
        )

    # Helper Methods ----------------------------------------------------------
    @classmethod
    def _check_rechunk(
        cls,
        func,
        data: conftest.DataLike,
        core_chunks: tuple[tuple[int]],
    ):
        # Similar to `MapT._check_chunk()`, but verifies core-chunk changes too.
        in_ = data["in_"].copy()
        arr = in_["arr"]
        xp = pxu.get_array_module(arr)
        arr = xp.broadcast_to(
            arr,
            shape=(5, 1, 3, *arr.shape),
            chunks=(2, 2, 2, *arr.chunks),
        )
        in_.update(arr=arr)
        with pxrt.EnforcePrecision(False):
            out = func(**in_)

        assert out.chunks[:3] == arr.chunks[:3]
        assert out.chunks[3:] == core_chunks

    # Tests -------------------------------------------------------------------
    def test_chunk_apply(self, op, ndi, chunks, _data_apply):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_rechunk(op.apply, _data_apply, chunks)

    def test_chunk_call(self, op, ndi, chunks, _data_apply):
        self._skip_if_disabled()
        self._skip_unless_DASK(ndi)
        self._check_rechunk(op.__call__, _data_apply, chunks)

    # Q: Why not apply LinOp's test_interface_jacobian() and rely instead on DiffMap's?
    # A: Because .jacobian() method is forwarded by IdentityOp().asop(UnitOp), hence RechunkAxes().jacobian() is the
    #    original IdentityOp() object, and not the RechunkAxes() object.  Modifying .asop() to avoid this behaviour is
    #    complex, and doesn't matter in practice.
    def test_interface_jacobian(self, op, _data_apply):
        self._skip_if_disabled()
        conftest.DiffMapT.test_interface_jacobian(self, op, _data_apply)
