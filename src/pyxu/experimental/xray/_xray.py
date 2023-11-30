import collections.abc as cabc
import typing as typ

import pyxu.abc as pxa
import pyxu.experimental.xray._fourier as _fourier
import pyxu.experimental.xray._rt as _rt
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt

__all__ = [
    "XRayTransform",
]


def _as_canonical(x, klass) -> tuple:
    # Transform a lone object into tuple form.
    if isinstance(x, cabc.Iterable):
        x = tuple(x)
    else:
        x = (x,)
    if klass is not None:
        x = tuple(map(klass, x))
    return x


class XRayTransform(pxa.LinOp):
    r"""
    X-Ray Transform (for :math:`D = \{2, 3\}`).

    The X-Ray Transform (XRT) of a function :math:`f: \mathbb{R}^{D} \to \mathbb{R}` is defined as

    .. math::

       \mathcal{P}[f](\mathbf{n}, \mathbf{t})
       =
       \int_{\mathbb{R}} f(\mathbf{t} + \mathbf{n} \alpha) d\alpha,

    where :math:`\mathbf{n}\in \mathbb{S}^{D-1}` and :math:`\mathbf{t} \in \mathbf{n}^{\perp}`.  :math:`\mathcal{P}[f]`
    hence denotes the set of *line integrals* of :math:`f`.

    Two class of algorithms exist to evaluate the XRT:

    * **Fourier methods** leverage the `Fourier Slice Theorem (FST)
      <https://en.wikipedia.org/wiki/Projection-slice_theorem>`_ to efficiently evaluate the XRT *when multiple values
      of* :math:`\mathbf{t}` *are desired for each* :math:`\mathbf{n}`.
    * **Ray methods** compute estimates of the XRT via quadrature rules by assuming :math:`f` is piecewise constant on
      short intervals.

    :py:class:`~pyxu.experimental.xray.XRayTransform` computes samples of the XRT assuming :math:`f` is a pixelized image/volume where:

    * the lower-left element of :math:`f` is located at :math:`\mathbf{o} \in \mathbb{R}^{D}`,
    * pixel dimensions are :math:`\mathbf{\Delta} \in \mathbb{R}_{+}^{D}`, i.e.

    .. math::

       f(\mathbf{r}) = \sum_{\{\mathbf{q}\} \subset \mathbb{N}^{D}}
                       \alpha_{\mathbf{q}}
                       1_{[\mathbf{0}, \mathbf{\Delta}]}(\mathbf{r} - \mathbf{q} \odot \mathbf{\Delta} - \mathbf{o}),
       \quad
       \alpha_{\mathbf{q}} \in \mathbb{R}.

    .. image:: /_static/api/xray/xray_parametrization.svg
       :alt: 2D XRay Geometry
       :width: 50%
       :align: center
    """

    @staticmethod
    @pxrt.enforce_precision(i="n_spec", o=False, allow_None=False)
    def init(
        arg_shape: pxt.NDArrayShape,
        origin=0,
        pitch=1,
        method="ray-trace",
        n_spec: pxt.NDArray = None,
        t_spec: typ.Union[pxt.NDArray, list[dict]] = None,
        **kwargs,
    ):
        r"""
        Instantiate an X-Ray Transform.

        Parameters
        ----------
        arg_shape: NDArrayShape
            Pixel count in each dimension.
        origin: Real, NDArray
            Bottom-left coordinate :math:`\mathbf{o} \in \mathbb{R}^{D}`.
        pitch: Real, NDArray
            Pixel size :math:`\mathbf{\Delta} \in \mathbb{R}_{+}^{D}`.
        method: "ray-trace", "fourier"
            How to evaluate the XRT.

            * `"ray-trace"` computes the XRT via a ray-marching method implemented using the `Dr.Jit
              <https://drjit.readthedocs.io/en/latest/reference.html>`_ compiler.
            * `"fourier"` computes the XRT using *NUFFT methods*. (See :py:class:`~pyxu.operator.NUFFT`.)
        n_spec: NDArray

            * If `method = "ray-trace"`: (N_l, D) ray directions :math:`\mathbf{n} \in \mathbb{S}^{D-1}`.

            * If `method = "fourier"`: (N_n, D, D) reference frame specifiers
              :math:`\{\mathbf{n},\mathbf{u},\mathbf{v}\} \in \mathbb{S}^{D-1}`.

            The 1st column of each (D, D) orthogonal sub-matrix :math:`\mathbf{U}` denotes the XRay line directions
            :math:`\mathbf{n} \in \mathbb{S}^{D-1}`.  Subsequent columns define the orientation of the parallel samples.
        t_spec: NDArray, list(dict)

            * If `method = "ray-trace"`: (N_l, D) offset specifiers :math:`\mathbf{t} \in \mathbb{R}^{D}`.

            * If `method = "fourier"`:

              * **non-uniform case**: (N_t, D-1) coordinates :math:`\mathbf{s} \in \mathbb{R}^{D-1}` such that
                :math:`\mathbf{t} = \mathbf{U}_{\mathbf{u}, \mathbf{v}} \mathbf{s}`.

              * **uniform case**: (D-1,) dictionaries of (a, b, M) triplets defining implicitly :math:`\mathbf{s} \in
                \mathbb{R}^{D-1}` such that :math:`\mathbf{U}_{\mathbf{u}, \mathbf{v}} \mathbf{s}`.
        kwargs: dict
            Extra keyword parameters forwarded to
            :py:class:`~pyxu.experimental.xray._rt.RayXRT` or
            :py:class:`~pyxu.experimental.xray._fourier.FourierXRT`.

            See
            :py:class:`~pyxu.experimental.xray._rt.RayXRT`,
            :py:class:`~pyxu.experimental.xray._fourier.FourierXRT`
            for more details.

        Notes
        -----
        * :py:class:`~pyxu.experimental.xray.XRayTransform` is not backend-agnostic: :py:class:`~pyxu.abc.Operator`
          methods of the instantiated object assume inputs have the same array backend as `n_spec`.
        * `method="ray-trace"` requires LLVM installed on the system. See the `Dr.Jit documentation
          <https://drjit.readthedocs.io/en/latest/index.html>`_ for details.
        * DASK inputs are currently unsupported.
        * :py:class:`~pyxu.experimental.xray.XRayTransform` is **not** precision-agnostic:
          fourier methods perform inner computations at the precision of `n_spec`, whereas ray-methods are always done
          in single-precision.
        """
        arg_shape = _as_canonical(arg_shape, int)
        D = len(arg_shape)
        assert D in {2, 3}

        origin = _as_canonical(origin, float)
        if len(origin) == 1:
            origin = origin * D
        assert len(origin) == D

        pitch = _as_canonical(pitch, float)
        if len(pitch) == 1:
            pitch = pitch * D
        assert len(pitch) == D
        assert all(p > 0 for p in pitch)

        method = method.strip().lower()
        assert method in {"ray-trace", "fourier"}

        if method == "ray-trace":
            # Validate `n_spec`
            ndi_n = pxd.NDArrayInfo.from_obj(n_spec)
            assert n_spec.ndim == 2
            N_l = n_spec.shape[0]
            assert n_spec.shape == (N_l, D)

            # Validate `t_spec`
            ndi_t = pxd.NDArrayInfo.from_obj(t_spec)
            assert ndi_n == ndi_t, "(n_spec, t_spec) must lie in same array namespace."
            assert t_spec.ndim == 2
            assert t_spec.shape == (N_l, D)

            # Scale `n_spec` to unit norm.
            # (Not really required; just to have everything in standardized form.)
            xp = ndi_n.module()
            n_spec = n_spec / xp.linalg.norm(n_spec, axis=-1, keepdims=True)

            ### Project `t_spec` to lie in n^{\perp}.
            ### (We don't do this by default since it messes up diagnostic_plot() analysis.)
            # rng = xp.random.default_rng()
            # U = rng.standard_normal((N_l, D, D))  # create reference frames n^{\perp}
            # U[..., 0] = n_spec
            # U, _ = xp.linalg.qr(U)
            # s = U[..., 1:].transpose(0, 2, 1) @ t_spec.reshape(N_l, D, 1)
            # t_spec = (U[..., 1:] @ s).reshape(N_l, D)

            klass = _rt.RayXRT
        else:  # method == "fourier"
            # Validate `n_spec`
            ndi_n = pxd.NDArrayInfo.from_obj(n_spec)
            assert n_spec.ndim == 3
            N_n = n_spec.shape[0]
            assert n_spec.shape == (N_n, D, D)
            xp = ndi_n.module()
            assert xp.allclose(
                n_spec @ n_spec.transpose(0, 2, 1),
                xp.eye(D),
            ), "Sub-matrices are not orthogonal."

            # Validate `t_spec`
            try:
                # NDArray case
                ndi_t = pxd.NDArrayInfo.from_obj(t_spec)
                assert t_spec.ndim == 2
                N_t = t_spec.shape[0]
                assert t_spec.shape == (N_t, D - 1)

                assert ndi_n == ndi_t, "(n_spec, t_spec) must lie in same array namespace."
            except ValueError:
                # {dict, list[dict]} case
                if isinstance(t_spec, dict):
                    t_spec = (t_spec,) * (D - 1)
                assert len(t_spec) == D - 1
                for _t in t_spec:
                    a, b, M = _t["start"], _t["stop"], _t["num"]
                    assert (a < b) and isinstance(M, int) and (M > 0)

            klass = _fourier.FourierXRT

        op = klass(
            arg_shape=arg_shape,
            origin=origin,
            pitch=pitch,
            n_spec=n_spec,
            t_spec=t_spec,
            **kwargs,
        )
        return op
