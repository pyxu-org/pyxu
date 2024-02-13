import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.runtime as pxrt
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest


class TestRayXRT(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.Width,
        )
    )
    def spec(
        self,
        dim_shape,
        origin,
        pitch,
        nt_spec,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        n_spec = ct.chunk_array(
            xp.array(nt_spec[0], dtype=width.value),
            complex_view=True,
            # `n_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )
        t_spec = ct.chunk_array(
            xp.array(nt_spec[1], dtype=width.value),
            complex_view=True,
            # `t_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )

        op = pxo.RayXRT(
            dim_shape=dim_shape,
            n_spec=n_spec,
            t_spec=t_spec,
            origin=origin,
            pitch=pitch,
            enable_warnings=False,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, space_dim) -> pxt.NDArrayShape:
        if space_dim == 2:
            return (5, 6)
        else:
            return (5, 3, 4)

    @pytest.fixture
    def codim_shape(self, nt_spec) -> pxt.NDArrayShape:
        n_spec, _ = nt_spec
        return (len(n_spec),)

    @pytest.fixture
    def data_apply(
        self,
        dim_shape,
        pitch,
    ) -> conftest.DataLike:
        V = self._random_array(dim_shape)  # (N1,...,ND)

        D = len(dim_shape)
        P = []
        for axis in range(D):
            p = np.sum(V * pitch[axis], axis=axis)
            P.append(p.reshape(-1))
        P = np.concatenate(P, axis=0)

        return dict(
            in_=dict(arr=V),
            out=P,
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture(params=[2, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def origin(self, space_dim) -> tuple[float]:
        # Volume origin
        orig = self._random_array((space_dim,))
        return tuple(orig)

    @pytest.fixture
    def pitch(self, space_dim) -> tuple[float]:
        # Voxel pitch
        pitch = abs(self._random_array((space_dim,))) + 1e-3
        return tuple(pitch)

    @pytest.fixture
    def nt_spec(self, dim_shape, origin, pitch) -> tuple[np.ndarray]:
        # To analytically test XRT correctness, we cast rays only along cardinal X/Y/Z directions, with one ray per
        # voxel side.

        D = len(dim_shape)
        n_spec = []
        t_spec = []
        for axis in range(D):
            # compute axes which are not projected
            dim = list(range(D))
            dim.pop(axis)

            # number of rays per dimension
            N_ray = np.array(dim_shape)[dim]

            n = np.zeros((*N_ray, D))
            n[..., axis] = 1
            n_spec.append(n.reshape(-1, D))

            t = np.zeros((*N_ray, D))
            _t = np.meshgrid(
                *[(np.arange(dim_shape[d]) + 0.5) * pitch[d] + origin[d] for d in dim],
                indexing="ij",
            )
            _t = np.stack(_t, axis=-1)
            t[..., dim] = _t
            t_spec.append(t.reshape(-1, D))

        n_spec = np.concatenate(n_spec, axis=0)
        t_spec = np.concatenate(t_spec, axis=0)
        return n_spec, t_spec
