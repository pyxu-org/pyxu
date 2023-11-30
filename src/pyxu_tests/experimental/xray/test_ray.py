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

    @pytest.fixture(params=[True, False])
    def weighted(self, request) -> bool:
        return request.param

    @pytest.fixture
    def ntw_spec(self, arg_shape, origin, pitch, weighted) -> np.ndarray:
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
        if weighted:
            # To avoid numerical inaccuracies in computing the ground-truth [due to use of np.exp()],
            # we limit the range of valid `w`.

            rng = np.random.default_rng(seed=0)
            N_cell = np.prod(arg_shape)
            w_spec = np.linspace(0.5, 1, N_cell, endpoint=True).reshape(arg_shape)
            w_spec *= rng.choice([-1, 1], size=w_spec.shape)
        else:
            w_spec = None
        return n_spec, t_spec, w_spec

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
        ntw_spec,
        weighted,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)
        xp = ndi.module()

        with pxrt.Precision(width):
            op = pxr.XRayTransform.init(
                arg_shape=arg_shape,
                origin=origin,
                pitch=pitch,
                method="ray-trace",
                n_spec=xp.array(ntw_spec[0]),
                t_spec=xp.array(ntw_spec[1]),
                w_spec=xp.array(ntw_spec[2]) if weighted else ntw_spec[2],
                enable_warnings=False,
            )
        return op, ndi, width

    @pytest.fixture
    def data_shape(self, arg_shape, ntw_spec) -> pxt.OpShape:
        dim = np.prod(arg_shape)
        codim = len(ntw_spec[0])
        return (codim, dim)

    @pytest.fixture(params=[53])  # seed
    def data_apply(
        self,
        arg_shape,
        pitch,
        weighted,
        ntw_spec,
        request,
    ) -> conftest.DataLike:
        seed = request.param
        rng = np.random.default_rng(seed)
        V = rng.standard_normal(size=arg_shape)
        w = ntw_spec[2]  # weights

        D = len(arg_shape)
        P = []
        for axis in range(D):
            if weighted:
                # Compute accumulated attenuation
                pad_width = [(0, 0)] * D
                pad_width[axis] = (1, 0)
                selector = [slice(None)] * D
                selector[axis] = slice(0, -1)
                _w = np.pad(w, pad_width)[tuple(selector)]

                A = np.exp(-pitch[axis] * np.cumsum(_w, axis=axis))
                B = np.where(np.isclose(w, 0), pitch[axis], (1 - np.exp(-w * pitch[axis])) / w)
                p = np.sum(V * A * B, axis=axis)
            else:
                p = np.sum(V * pitch[axis], axis=axis)
            P.append(p.reshape(-1))
        P = np.concatenate(P, axis=0)

        return dict(
            in_=dict(arr=V.reshape(-1)),
            out=P,
        )
