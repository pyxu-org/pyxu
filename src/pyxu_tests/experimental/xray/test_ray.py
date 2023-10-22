import itertools

import numpy as np
import pytest

import pyxu.experimental.xray as pxr
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest


class TestRayXRT(conftest.LinOpT):
    @pytest.fixture(
        params=[
            2,
            3,
        ]
    )
    def dimensionality(self, request) -> int:
        return request.param

    @pytest.fixture
    def arg_shape(self, dimensionality) -> pxt.NDArrayShape:
        if dimensionality == 2:
            return (5, 6)
        else:
            return (5, 3, 4)

    @pytest.fixture(params=[0])  # seed
    def origin(self, dimensionality, request) -> tuple[float]:
        seed = request.param
        rng = np.random.default_rng(seed)
        orig = rng.uniform(-1, 1, size=dimensionality)
        return tuple(orig)

    @pytest.fixture(params=[1])  # seed
    def pitch(self, dimensionality, request) -> tuple[float]:
        seed = request.param
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.5, 3, size=dimensionality)
        return tuple(p)

    @pytest.fixture
    def nt_spec(self, arg_shape, origin, pitch) -> np.ndarray:
        # To analytically test XRT correctness, we cast rays only along cardinal X/Y/Z directions, with one ray per
        # voxel side.

        D = len(arg_shape)
        n_spec = []
        t_spec = []
        for axis in range(D):
            # compute axes which are not projected
            dim = list(range(D))
            dim.pop(axis)

            # number of rays per dimension
            N_ray = np.array(arg_shape)[dim]

            n = np.zeros((*N_ray, D))
            n[..., axis] = 1
            n_spec.append(n.reshape(-1, D))

            t = np.zeros((*N_ray, D))
            _t = np.meshgrid(
                *[(np.arange(arg_shape[d]) + 0.5) * pitch[d] + origin[d] for d in dim],
                indexing="ij",
            )
            _t = np.stack(_t, axis=-1)
            t[..., dim] = _t
            t_spec.append(t.reshape(-1, D))

        n_spec = np.concatenate(n_spec, axis=0)
        t_spec = np.concatenate(t_spec, axis=0)
        return n_spec, t_spec

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
            ],
            pxrt.Width,
        )
    )
    def spec(
        self,
        arg_shape,
        origin,
        pitch,
        nt_spec,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        xp = ndi.module()

        with pxrt.Precision(width):
            op = pxr.XRayTransform.init(
                arg_shape=arg_shape,
                origin=origin,
                pitch=pitch,
                method="ray-trace",
                n_spec=xp.array(nt_spec[0]),
                t_spec=xp.array(nt_spec[1]),
            )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape, nt_spec) -> pxt.OpShape:
        dim = np.prod(arg_shape)
        codim = len(nt_spec[0])
        return (codim, dim)

    @pytest.fixture(params=[53])  # seed
    def data_apply(self, arg_shape, pitch, request) -> conftest.DataLike:
        seed = request.param
        rng = np.random.default_rng(seed)
        V = rng.standard_normal(size=arg_shape)

        D = len(arg_shape)
        P = []
        for axis in range(D):
            p = np.sum(V * pitch[axis], axis=axis)
            P.append(p.reshape(-1))
        P = np.concatenate(P, axis=0)

        return dict(
            in_=dict(arr=V.reshape(-1)),
            out=P,
        )
