import collections.abc as cabc
import operator
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "RayXRT",
]


class RayXRT(pxa.LinOp):
    r"""
    X-Ray Transform (for :math:`D = \{2, 3\}`).

    The X-Ray Transform (XRT) of a function :math:`f: \mathbb{R}^{D} \to \mathbb{R}` is defined as

    .. math::

       \mathcal{P}[f](\mathbf{n}, \mathbf{t})
       =
       \int_{\mathbb{R}} f(\mathbf{t} + \mathbf{n} \alpha) d\alpha,

    where :math:`\mathbf{n}\in \mathbb{S}^{D-1}` and :math:`\mathbf{t} \in \mathbf{n}^{\perp}`.
    :math:`\mathcal{P}[f]` hence denotes the set of *line integrals* of :math:`f`.

    This implementation computes samples of the XRT using a ray-marching method based on the `Dr.Jit
    <https://drjit.readthedocs.io/en/latest/reference.html>`_ compiler. It assumes :math:`f` is a pixelized image/volume
    where:

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

    Notes
    -----
    * :py:class:`~pyxu.operator.RayXRT` requires LLVM installed on the system. See the `Dr.Jit documentation
      <https://drjit.readthedocs.io/en/latest/index.html>`_ for details.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        n_spec: pxt.NDArray,
        t_spec: pxt.NDArray,
        origin: tuple[float] = 0,
        pitch: tuple[float] = 1,
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (N1,...,ND) pixel count in each dimension.
        n_spec: NDArray
            (N_ray, D) ray directions :math:`\mathbf{n} \in \mathbb{S}^{D-1}`.
        t_spec: NDArray
            (N_ray, D) offset specifiers :math:`\mathbf{t} \in \mathbb{R}^{D}`.
        origin: float, tuple[float]
            Bottom-left coordinate :math:`\mathbf{o} \in \mathbb{R}^{D}`.
        pitch: float, tuple[float]
            Pixel size :math:`\mathbf{\Delta} \in \mathbb{R}_{+}^{D}`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.

        Notes
        -----
        * :py:class:`~pyxu.operator.RayXRT` instances are **not arraymodule-agnostic**: they will only work with
          NDArrays belonging to the same array module as (`n_spec`, `t_spec`).
        * :py:class:`~pyxu.operator.RayXRT` is **not** precision-agnostic: it will only work on NDArrays in
          single-precision. A warning is emitted if inputs must be cast.
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=len(n_spec),
        )
        assert self.dim_rank in (2, 3)

        # Put all variables in canonical form & validate ----------------------
        #   n_spec: (N_ray, D) array[float32] (NUMPY/CUPY/DASK)
        #   t_spec: (N_ray, D) array[float32] (NUMPY/CUPY/DASK)
        #   origin: (D,) float
        #   pitch: (D,) float
        origin = self._as_seq(origin, self.dim_rank, float)
        pitch = self._as_seq(pitch, self.dim_rank, float)
        assert all(p > 0 for p in pitch)
        with pxrt.Precision(pxrt.Width.SINGLE):
            n_spec = pxrt.coerce(n_spec)  # (N_ray, D)
            t_spec = pxrt.coerce(t_spec)  # (N_ray, D)
        assert n_spec.shape == (self.codim_size, self.dim_rank)
        assert t_spec.shape == (self.codim_size, self.dim_rank)
        assert operator.eq(
            ndi := pxd.NDArrayInfo.from_obj(n_spec),
            pxd.NDArrayInfo.from_obj(t_spec),
        ), "[n_spec,t_spec] Must belong to the same array backend."

        # Initialize Operator Variables ---------------------------------------
        self._n_spec = n_spec
        self._t_spec = t_spec
        self._origin = origin
        self._pitch = pitch
        self._enable_warnings = bool(enable_warnings)

        # Cheap analytical Lipschitz upper bound given by
        #   \sigma_{\max}(P) <= \norm{P}{F},
        # with
        #   \norm{P}{F}^{2}
        #   <= (max cell weight)^{2} * #non-zero elements
        #   <= (max cell weight)^{2} * N_ray * (maximum number of cells traversable by a ray)
        #    = (max cell weight)^{2} * N_ray * \norm{dim_shape}{2}
        #
        #   (max cell weight) = \norm{pitch}{2}
        max_cell_weight = np.linalg.norm(pitch)
        self.lipschitz = max_cell_weight * np.sqrt(self.codim_size * np.linalg.norm(self.dim_shape))

        # Auto-vectorize [apply,adjoint]() ------------------------------------
        v_apply = pxu.vectorize(
            i="arr",
            dim_shape=self.dim_shape,
            codim_shape=self.codim_shape,
        )
        self.apply = v_apply(self.apply)
        v_adjoint = pxu.vectorize(
            i="arr",
            dim_shape=self.codim_shape,
            codim_shape=self.dim_shape,
        )
        self.adjoint = v_adjoint(self.adjoint)

        # Run-time vs. Init-time Setup ----------------------------------------
        if ndi == pxd.NDArrayInfo.DASK:
            # Build data structures at runtime; just validate (n_spec, t_spec)
            assert self._n_spec.chunks == self._t_spec.chunks, "[n_spec,t_spec] Must have same chunk sizes."
            assert self._n_spec.chunks[1] == (
                self.dim_rank,
            ), "[n_spec,t_spec] Chunking along last dimension unsupported."
        else:  # init-time instantiation: create drjit variables
            self._dr = self._init_dr_metadata()

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N1,...,ND) spatial weights.

        Returns
        -------
        out: NDArray
            (..., N_ray) XRT samples.
        """
        arr, dtype = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)
        xp = ndi.module()

        if ndi == pxd.NDArrayInfo.DASK:
            # High-level idea:
            # 1. foreach (nt_spec, I-chunk) pair: compute projections.
            # 2. collapse projections across all I-chunks.
            #
            # Concretely, we rely on DASK.blockwise() to achieve this.
            #
            # For each sub-problem to compute the right outputs, it must know "where" the I-chunk is located in space,
            # i.e. what `origin` is for that chunk.  As such we need to encode the chunk origins as an array and give
            # them to blockwise(). This is encoded in `origin` below.
            #
            # Reminder of array shape/block structures that blockwise() will use:
            # [legend] array: shape, blocks/dim, dimension index {see blockwise().}]
            # * I: (N1,...,ND), (Bi1,...,BiD), (1,...,D)
            # * n_spec: (N_ray, D), (Bp, 1), (0, D+1)
            # * t_spec: (N_ray, D), (Bp, 1), (0, D+1)
            # * origin: (Bi1,...,BiD, D), (Bi1,...,BiD, 1), (1,...,D, D+2)
            # * parts: [ this is the output of blockwise() ]
            #       (N_ray, Bi1,...,BiD),
            #       (Bp,    Bi1,...,BiD), -> we 'sumed' over the single-block axes (D+1, D+2)
            #       ( 0,      1,...,  D)
            # * out [ = parts.sum(axis=(-D,...,-1)) ]
            #       (N_ray,), (Bp,)

            # Compute `origin` info.
            offset = [np.r_[0, chks].cumsum()[:-1] for chks in arr.chunks]
            offset = np.stack(  # (Bi1,...,BiD, D)
                np.meshgrid(*offset, indexing="ij"),
                axis=-1,
            )
            origin = xp.asarray(  # (Bi1,...,BiD, D)
                np.r_[self._origin] + np.r_[self._pitch] * offset,
                chunks=(1,) * self.dim_rank + (self.dim_rank,),
            )

            # Compute (I,n,t,orig,o)_ind & output chunks
            I_ind = tuple(range(1, self.dim_rank + 1))
            n_ind = (0, self.dim_rank + 1)
            t_ind = (0, self.dim_rank + 1)
            orig_ind = (*range(1, self.dim_rank + 1), self.dim_rank + 2)
            o_ind = tuple(range(self.dim_rank + 1))
            o_chunks = {d: 1 for d in range(1, self.dim_rank + 1)}

            parts = xp.blockwise(
                # shape:  (N_ray | Bi1,...,BiD)
                # bcount: (Bp    | Bi1,...,BiD)
                *(self._blockwise_apply, o_ind),
                *(arr, I_ind),
                *(self._n_spec, n_ind),
                *(self._t_spec, t_ind),
                *(origin, orig_ind),
                dtype=dtype,
                adjust_chunks=o_chunks,
                align_arrays=False,
                concatenate=True,
                meta=arr._meta,
            )
            out = parts.sum(axis=tuple(range(-self.dim_rank, 0)))  # (N_ray,)
        else:  # NUMPY/CUPY
            from . import _drjit as drh

            # Load the right drjit function
            drb = drh._load_dr_variant(ndi)
            fwd, _ = drh._build_xrt(drb, D=self.dim_rank, weighted=False)

            # Transform (I,) to drjit data structure
            _I = arr.ravel()  # (N1*...*ND,) contiguous
            I_dr = drb.Float(drh._xp2dr(_I))

            # Compute projections
            P = fwd(**self._dr, I=I_dr)
            out = xp.asarray(P, dtype=dtype)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N_ray) XRT samples.

        Returns
        -------
        out: NDArray
            (..., N1,...,ND) spatial weights.
        """
        arr, dtype = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)
        xp = ndi.module()

        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError
        else:  # NUMPY/CUPY
            from . import _drjit as drh

            # Load the right drjit function
            drb = drh._load_dr_variant(ndi)
            _, bwd = drh._build_xrt(drb, D=self.dim_rank, weighted=False)

            # Transform (P,) to drjit data structure
            P = arr.ravel()  # (N_ray,) contiguous
            P_dr = drb.Float(drh._xp2dr(P))

            # Compute back-projections
            _I = bwd(**self._dr, P=P_dr)
            _I = xp.asarray(_I, dtype=dtype)
            out = _I.reshape(self.dim_shape)  # (N1,...,ND)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `nt_spec`-backend ... ------------------------
        A = super().asarray(
            xp=pxu.get_array_module(self._n_spec),
            dtype=pxrt.Width.SINGLE.value,
        )

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        B = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return B

    def diagnostic_plot(
        self,
        ray_idx: pxt.NDArray = None,
        show_grid: bool = False,
    ):
        r"""
        Plot ray trajectories.

        Parameters
        ----------
        ray_idx: NDArray
            (Q,) indices of rays to plot. (Default: show all rays.)
        show_grid: bool
            If true, overlay the pixel grid.

        Returns
        -------
        fig: :py:class:`~matplotlib.figure.Figure`
            Diagnostic plot.

        Notes
        -----
        * Rays which do not intersect the volume are **not** shown.
        * This method only works if (n_spec,t_spec) were provided as NUMPY/CUPY arrays.

        Examples
        --------

        .. plot::

           import numpy as np
           import pyxu.operator as pxo

           op = pxo.RayXRT(
               dim_shape=(5, 6),
               origin=0,
               pitch=1,
               n_spec=np.array([[1   , 0   ],  # 3 rays ...
                                [0.5 , 0.5 ],
                                [0.75, 0.25]]),
               t_spec=np.array([[2.5, 3],  # ... all defined w.r.t volume center
                                [2.5, 3],
                                [2.5, 3]]),
           )
           fig = op.diagnostic_plot()
           fig.show()

        Notes
        -----
        Requires `Matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        from . import _drjit as drh

        dr = pxu.import_module("drjit")
        plt = pxu.import_module("matplotlib.pyplot")
        collections = pxu.import_module("matplotlib.collections")
        patches = pxu.import_module("matplotlib.patches")

        # Setup Figure ========================================================
        if self.dim_rank == 2:
            fig, ax = plt.subplots()
            data = [(ax, [0, 1], ["x", "y"])]
        else:  # D == 3 case
            fig, ax = plt.subplots(ncols=3)
            data = [
                (ax[0], [0, 1], ["x", "y"]),
                (ax[1], [0, 2], ["x", "z"]),
                (ax[2], [1, 2], ["y", "z"]),
            ]

        # Determine which rays intersect with BoundingBox =====================
        ndi = pxd.NDArrayInfo.from_obj(self._n_spec)
        drb = drh._load_dr_variant(ndi)
        BBoxf = drh._BoundingBoxf_Factory(drb, self.dim_rank)
        active, a1, a2 = BBoxf(
            pMin=self._dr["o"],
            pMax=self._dr["o"] + self._dr["pitch"] * self._dr["N"],
        ).ray_intersect(self._dr["r"])
        dr.eval(active, a1, a2)

        # Then extract subset of interest (which intersect bbox)
        if ray_idx is None:
            ray_idx = slice(None)
        active, a1, a2 = map(lambda _: _.numpy()[ray_idx], [active, a1, a2])  # (Q,)
        a12 = np.stack([a1, a2], axis=-1)[active]  # (N_active, 2)
        ray_n = pxu.to_NUMPY(self._n_spec[ray_idx][active])  # (N_active, D)
        ray_t = pxu.to_NUMPY(self._t_spec[ray_idx][active])  # (N_active, D)

        for _ax, dim_idx, dim_label in data:
            # Subsample right dimensions ======================================
            origin = np.array(self._origin)[dim_idx]
            dim_shape = np.array(self.dim_shape)[dim_idx]
            pitch = np.array(self._pitch)[dim_idx]
            _ray_n = ray_n[:, dim_idx]
            _ray_t = ray_t[:, dim_idx]

            # Helper variables ================================================
            bbox_dim = dim_shape * pitch

            # Draw BBox =======================================================
            rect = patches.Rectangle(
                xy=origin,
                width=bbox_dim[0],
                height=bbox_dim[1],
                facecolor="none",
                edgecolor="k",
                label="volume BBox",
            )
            _ax.add_patch(rect)

            # Draw Pitch =======================================================
            p_rect = patches.Rectangle(
                xy=origin + bbox_dim - pitch,
                width=pitch[0],
                height=pitch[1],
                facecolor="r",
                edgecolor="none",
                label="pitch size",
            )
            _ax.add_patch(p_rect)

            # Draw Origin =====================================================
            _ax.scatter(
                origin[0],
                origin[1],
                color="k",
                label="origin",
                marker="x",
            )

            # Draw Rays & Anchor Points =======================================
            # Each (2,2) sub-array in `coords` represents line start/end coordinates.
            coords = _ray_t.reshape(-1, 1, 2) + a12.reshape(-1, 2, 1) * _ray_n.reshape(-1, 1, 2)  # (N_active, 2, 2)
            lines = collections.LineCollection(
                coords,
                label=r"$t + \alpha n$",
                color="k",
                alpha=0.5,
                linewidth=1,
            )
            _ax.add_collection(lines)
            _ax.scatter(
                _ray_t[:, 0],
                _ray_t[:, 1],
                label=r"t",
                color="g",
                marker=".",
            )

            # Draw Overlay Grid ===============================================
            if show_grid:
                x_ticks = origin[0] + pitch[0] * np.arange(dim_shape[0])
                y_ticks = origin[1] + pitch[1] * np.arange(dim_shape[1])
                _ax.set_xticks(x_ticks)
                _ax.set_yticks(y_ticks)
                _ax.grid(
                    linestyle="--",
                    color="gray",
                )

            # Misc Details ====================================================
            pad_width = 0.1 * bbox_dim  # 10% axial pad
            _ax.set_xlabel(dim_label[0])
            _ax.set_ylabel(dim_label[1])
            _ax.set_xlim(origin[0] - pad_width[0], origin[0] + bbox_dim[0] + pad_width[0])
            _ax.set_ylim(origin[1] - pad_width[1], origin[1] + bbox_dim[1] + pad_width[1])
            _ax.legend(loc="lower right", bbox_to_anchor=(1, 1))
            _ax.set_aspect(1)

        fig.tight_layout()
        return fig

    # Internal Helpers --------------------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> tuple[pxt.NDArray, pxt.DType]:
        W = pxrt.Width  # shorthand
        if W(arr.dtype) != W.SINGLE:
            if self._enable_warnings:
                msg = f"Only {W.SINGLE}-precision inputs are supported: casting."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=W.SINGLE.value)
        else:
            out = arr
        return out, arr.dtype

    @staticmethod
    def _as_seq(x, N, _type=None) -> tuple:
        if isinstance(x, cabc.Iterable):
            _x = tuple(x)
        else:
            _x = (x,)
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

        if _type is None:
            return _x
        else:
            return tuple(map(_type, _x))

    def _init_dr_metadata(self) -> dict:
        # Compute all RayXRT parameters.
        #
        # * o: (D,) Arrayf        [volume reference point]
        # * pitch: (D,) Arrayf    [pixel pitch]
        # * N: (D,) Arrayu        [pixel count]
        # * r: (N_ray,) Rayf      [zero-copy view of (n_spec, t_spec)]
        from . import _drjit as drh

        ndi = pxd.NDArrayInfo.from_obj(self._n_spec)
        drb = drh._load_dr_variant(ndi)

        Arrayf = drh._Arrayf_Factory(drb, self.dim_rank)
        Arrayu = drh._Arrayu_Factory(drb, self.dim_rank)
        Rayf = drh._Rayf_Factory(drb, self.dim_rank)

        meta = dict(
            o=Arrayf(*self._origin),
            pitch=Arrayf(*self._pitch),
            N=Arrayu(*self.dim_shape),
            r=Rayf(
                o=Arrayf(*[drh._xp2dr(_) for _ in self._t_spec.T]),
                d=Arrayf(*[drh._xp2dr(_) for _ in self._n_spec.T]),
            ),
        )
        return meta

    def _blockwise_apply(self, I, n_spec, t_spec, origin) -> pxt.NDArray:
        # Project rays through sub-volume.
        # [All arrays are NUMPY/CUPY.]
        #
        # Parameters
        # ----------
        # I: NDArray[float32]
        #     (S1,...,SD) sub-volume entries.
        # n_spec: NDArray[float32]
        #     (L, D) ray directions :math:`\mathbf{n} \in \mathbb{S}^{D-1}`.
        # t_spec: NDArray[float32]
        #     (L, D) offset specifiers :math:`\mathbf{t} \in \mathbb{R}^{D}`.
        # origin: NDArray[float]
        #     (<D 1s>, D) bottom-left coordinate of sub-volume.
        #
        # Returns
        # -------
        # P: NDArray[float32]
        #     (L, <D 1s>) projection weights.
        #
        #     [Note the trailing size-1 dims; these are required since blockwise() expects to
        #      stack these outputs given how it was called.]
        select = (0,) * self.dim_rank
        origin = origin[*select]  # (D,)

        op = RayXRT(
            dim_shape=I.shape,
            n_spec=n_spec,
            t_spec=t_spec,
            origin=origin,
            pitch=self._pitch,
            enable_warnings=self._enable_warnings,
        )
        P = op.apply(I)  # (L,)

        expand = (np.newaxis,) * self.dim_rank
        return P[..., *expand]  # (L, <D 1s>)
