import types
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu


class RayXRT(pxa.LinOp):
    r"""
    2D/3D X-Ray Transform :math:`\mathcal{P}[f](\mathbf{n}, \mathbf{t})`.

    This implementation computes the XRT using a ray-marching method based on the `Dr.Jit
    <https://drjit.readthedocs.io/en/latest/reference.html>`_ compiler.

    See :py:class:`~pyxu.experimental.xray.XRayTransform` for notational conventions used herein.
    """

    def __init__(
        self,
        arg_shape,
        origin,
        pitch,
        n_spec,
        t_spec,
        w_spec=None,
        enable_warnings: bool = True,
    ):
        r"""
        Parameters
        ----------
        arg_shape: NDArrayShape
            Pixel count in each dimension.
        origin: NDArray
            Bottom-left coordinate :math:`\mathbf{o} \in \mathbb{R}^{D}`.
        pitch: NDArray
            Pixel size :math:`\mathbf{\Delta} \in \mathbb{R}_{+}^{D}`.
        n_spec: NDArray
            (N_l, D) ray directions :math:`\mathbf{n} \in \mathbb{S}^{D-1}`.
        t_spec: NDArray
            (N_l, D) offset specifiers :math:`\mathbf{t} \in \mathbb{R}^{D}`.
        w_spec: NDArray, None
            (*arg_shape,) spatial decay weights.
            If unspecified an un-weighted XRT is performed. (Default)

            A *weighted* XRT is defined as

            .. math::

               \mathcal{P}_{w}[f](\mathbf{n}, \mathbf{t})
               =
               \int_{\mathbb{R}} f(\mathbf{t} + \mathbf{n} \alpha)
               \exp\left[ -\int_{-\infty}^{\alpha} w(\mathbf{t} + \mathbf{n} \beta) d\beta \right]
               d\alpha,

            where :math:`w: \mathbb{R}^{D} \to \mathbb{R}` takes the form
            (same pixelized form as :math:`\mathbf{f}`):

            .. math::

               w(\mathbf{r}) = \sum_{\{\mathbf{q}\} \subset \mathbb{N}^{D}}
                               \alpha_{\mathbf{q}}
                               1_{[\mathbf{0}, \mathbf{\Delta}]}(\mathbf{r} - \mathbf{q} \odot \mathbf{\Delta} - \mathbf{o}),
               \quad
               \alpha_{\mathbf{q}} \in \mathbb{R}.
        enable_warnings: bool


        .. warning::

           This is a low-level interface which does not perform any much validation.  Users are expected to
           instantiate :py:class:`~pyxu.experimental.xray._rt.RayXRT` by calling
           :py:meth:`~pyxu.experimental.xray.XRayTransform.init` instead.
        """
        N_px = np.prod(arg_shape)
        N_l = len(n_spec)
        super().__init__(shape=(N_l, N_px))

        W = pxrt.Width  # shorthand
        n_spec = n_spec.astype(W.SINGLE.value, copy=False)
        t_spec = t_spec.astype(W.SINGLE.value, copy=False)
        if weighted := w_spec is not None:
            w_spec = w_spec.astype(W.SINGLE.value, copy=False)

        # Internal variables. (Have shapes consistent with user inputs.)
        self._arg_shape = arg_shape  # (D,)
        self._origin = origin  # (D,)
        self._pitch = pitch  # (D,)
        self._ray_n = n_spec  # (N_l, D)
        self._ray_t = t_spec  # (N_l, D)
        self._ndi = pxd.NDArrayInfo.from_obj(n_spec)
        self._enable_warnings = bool(enable_warnings)

        # Validate RayXRT-only parameters: w_spec
        if weighted:
            assert w_spec.shape == arg_shape
            assert pxd.NDArrayInfo.from_obj(w_spec) == self._ndi

        # Cheap analytical Lipschitz upper bound given by
        #   \sigma_{\max}(P) <= \norm{P}{F},
        # with
        #   \norm{P}{F}^{2}
        #   <= (max cell weight)^{2} * #non-zero elements
        #    = (max cell weight)^{2} * N_ray * (maximum number of cells traversable by a ray)
        #    = (max cell weight)^{2} * N_ray * \norm{arg_shape}{2}
        #
        #    (max cell weight) =
        #      unweighted              : \norm{pitch}{2}
        #        weighted & (w_min > 0): \norm{pitch}{2}
        #        weighted & (w_min < 0): cannot infer
        if weighted and (w_spec.min() < 0):
            max_cell_weight = np.inf
        else:
            max_cell_weight = np.linalg.norm(pitch)
        self.lipschitz = max_cell_weight * np.sqrt(N_l * np.linalg.norm(arg_shape))

        # Dr.Jit variables. {Have shapes consistent for xrt_[apply,adjoint]().}
        #   xrt_[apply,adjoint]() only support D=3 case.
        #     -> D=2 case is embedded into 3D.
        drb = _load_dr_variant(self._ndi)
        Ray3f = _Ray3f_Factory(self._ndi)
        self._fwd, self._bwd = _build_xrt(self._ndi, weighted)
        if len(arg_shape) == 2:
            self._dr = dict(
                o=drb.Array3f(*self._origin, 0),
                pitch=drb.Array3f(*self._pitch, 1),
                N=drb.Array3u(*self._arg_shape, 1),
                r=Ray3f(
                    o=drb.Array3f(*[_xp2dr(_) for _ in self._ray_t.T], 0.5),  # Z-extension mid-point
                    d=drb.Array3f(*[_xp2dr(_) for _ in self._ray_n.T], 0),
                ),
            )
        else:
            self._dr = dict(
                o=drb.Array3f(*self._origin),
                pitch=drb.Array3f(*self._pitch),
                N=drb.Array3u(*self._arg_shape),
                r=Ray3f(
                    o=drb.Array3f(*[_xp2dr(_) for _ in self._ray_t.T]),
                    d=drb.Array3f(*[_xp2dr(_) for _ in self._ray_n.T]),
                ),
            )
        if weighted:
            self._dr["w"] = drb.Float(_xp2dr(w_spec.reshape(-1)))

    @pxrt.enforce_precision(i="arr")
    @pxu.vectorize(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (...,  arg_shape.prod()) spatial weights.

        Returns
        -------
        P: NDArray
            (..., N_l) XRT samples.
        """
        arr, dtype = self._warn_cast(arr)

        drb = _load_dr_variant(self._ndi)
        P = self._fwd(
            **self._dr,
            I=drb.Float(_xp2dr(arr)),
        )

        xp = self._ndi.module()
        P = xp.asarray(P, dtype=dtype)
        return P

    @pxrt.enforce_precision(i="arr")
    @pxu.vectorize(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N_l) XRT samples.

        Returns
        -------
        I: NDArray
            (...,  arg_shape.prod()) spatial weights.
        """
        arr, dtype = self._warn_cast(arr)

        drb = _load_dr_variant(self._ndi)
        I = self._bwd(  # noqa: E741
            **self._dr,
            P=drb.Float(_xp2dr(arr)),
        )

        xp = self._ndi.module()
        I = xp.asarray(I, dtype=dtype)  # noqa: E741
        return I

    def _warn_cast(self, arr: pxt.NDArray) -> tuple[pxt.NDArray, pxt.DType]:
        W = pxrt.Width  # shorthand
        if W(arr.dtype) != W.SINGLE:
            if self._enable_warnings:
                msg = f"Only {W.SINGLE}-precision inputs are supported: casting."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=W.SINGLE.value)
        else:
            out = arr
        return out, arr.dtype

    def asarray(self, **kwargs) -> pxt.NDArray:
        # DASK not yet supported, hence we support `xp=DASK` case sub-optimally for now.
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)

        # compute array representation using instance's backend.
        kwargs.update(xp=self._ndi.module())
        A = super().asarray(**kwargs)

        # then cast to user specs.
        A = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return A

    def gram(self) -> pxt.OpT:
        # We replace apply() with a variant which does not force evaluation between FW/BW calls.

        @pxrt.enforce_precision(i="arr")
        @pxu.vectorize(i="arr")
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            arr, dtype = _._warn_cast(arr)

            drb = _load_dr_variant(_._ndi)
            P = _._fwd(
                **_._dr,
                I=drb.Float(_xp2dr(arr)),
            )
            I = _._bwd(  # noqa: E741
                **_._dr,
                P=P,
            )

            xp = _._ndi.module()
            I = xp.asarray(I, dtype=dtype)  # noqa: E741
            return I

        op = super().gram()
        op.apply = types.MethodType(op_apply, self)
        return op

    def cogram(self) -> pxt.OpT:
        # We replace apply() with a variant which does not enforce evaluation between BW/FW calls.

        @pxrt.enforce_precision(i="arr")
        @pxu.vectorize(i="arr")
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            arr, dtype = _._warn_cast(arr)

            drb = _load_dr_variant(_._ndi)
            I = _._bwd(  # noqa: E741
                **_._dr,
                P=drb.Float(_xp2dr(arr)),
            )
            P = _._fwd(
                **_._dr,
                I=I,
            )

            xp = _._ndi.module()
            P = xp.asarray(P, dtype=dtype)
            return P

        op = super().cogram()
        op.apply = types.MethodType(op_apply, self)
        return op

    def diagnostic_plot(self, ray_idx: pxt.NDArray = None):
        r"""
        Plot ray trajectories.

        Parameters
        ----------
        ray_idx: NDArray
            (Q,) indices of rays to plot. (Default: show all rays.)

        Returns
        -------
        fig: :py:class:`~matplotlib.figure.Figure`
            Diagnostic plot.

        Notes
        -----
        * Rays which do not intersect the volume are **not** shown.

        Examples
        --------

        .. plot::

           import numpy as np
           import pyxu.experimental.xray as pxr

           op = pxr.XRayTransform.init(
               arg_shape=(5, 6),
               origin=0,
               pitch=1,
               method="ray-trace",
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
        dr = pxu.import_module("drjit")
        plt = pxu.import_module("matplotlib.pyplot")
        collections = pxu.import_module("matplotlib.collections")
        patches = pxu.import_module("matplotlib.patches")

        # Setup Figure ========================================================
        D = len(self._arg_shape)
        if D == 2:
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
        BBox3f = _BoundingBox3f_Factory(self._ndi)
        active, a1, a2 = BBox3f(
            pMin=self._dr["o"],
            pMax=self._dr["o"] + self._dr["pitch"] * self._dr["N"],
        ).ray_intersect(self._dr["r"])
        dr.eval(active, a1, a2)

        # Then extract subset of interest (which intersect bbox)
        if ray_idx is None:
            ray_idx = slice(None)
        active, a1, a2 = map(lambda _: _.numpy()[ray_idx], [active, a1, a2])  # (Q,)
        a12 = np.stack([a1, a2], axis=-1)[active]  # (N_active, 2)
        ray_n = self._ray_n[ray_idx][active]  # (N_active, D)
        ray_t = self._ray_t[ray_idx][active]  # (N_active, D)

        for _ax, dim_idx, dim_label in data:
            # Subsample right dimensions ======================================
            origin = np.array(self._origin)[dim_idx]
            arg_shape = np.array(self._arg_shape)[dim_idx]
            pitch = np.array(self._pitch)[dim_idx]
            _ray_n = ray_n[:, dim_idx]
            _ray_t = ray_t[:, dim_idx]

            # Helper variables ================================================
            bbox_dim = arg_shape * pitch

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


# Dr.Jit Helper Functions =====================================================
def _xp2dr(x: pxt.NDArray):
    # Transform NP/CP inputs to format allowing zero-copy casts to {drb.Float, drb.Array3f}
    ndi = pxd.NDArrayInfo.from_obj(x)
    if ndi == pxd.NDArrayInfo.NUMPY:
        return x
    elif ndi == pxd.NDArrayInfo.CUPY:
        return x.__dlpack__()
    else:
        raise NotImplementedError


def _load_dr_variant(ndi: pxd.NDArrayInfo):
    # Load right computation backend
    if ndi == pxd.NDArrayInfo.NUMPY:
        drb = pxu.import_module("drjit.llvm")
    elif ndi == pxd.NDArrayInfo.CUPY:
        drb = pxu.import_module("drjit.cuda")
    else:
        raise NotImplementedError
    return drb


def _Ray3f_Factory(ndi: pxd.NDArrayInfo):
    # Create a Ray3f class associated with a compute backend.
    dr = pxu.import_module("drjit")
    drb = _load_dr_variant(ndi)

    class Ray3f:
        # Dr.JIT-backed 3D ray.
        #
        # Rays take the parametric form
        #     r(t) = o + d * t,
        # where {o, d} \in \bR^{3} and t \in \bR.
        DRJIT_STRUCT = dict(
            o=drb.Array3f,
            d=drb.Array3f,
        )

        def __init__(self, o=drb.Array3f(), d=drb.Array3f()):
            # Parameters
            # ----------
            # o: Array3f
            #    (3,) ray origin.
            # d: Array3f
            #    (3,) ray direction.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.o = drb.Array3f()
            self.o.assign(o)

            self.d = drb.Array3f()
            self.d.assign(d)

        def __call__(self, t: drb.Float) -> drb.Array3f:
            # Compute r(t).
            #
            # Parameters
            # ----------
            # t: Float
            #
            # Returns
            # -------
            # p: Array3f
            #    p = r(t)
            return dr.fma(self.d, t, self.o)

        def assign(self, r: "Ray3f"):
            # See __init__'s docstring for more info.
            self.o.assign(r.o)
            self.d.assign(r.d)

        def __repr__(self) -> str:
            return f"Ray3f(o={dr.shape(self.o)}, d={dr.shape(self.d)})"

    return Ray3f


def _BoundingBox3f_Factory(ndi: pxd.NDArrayInfo):
    # Create a BoundingBox3f class associated with a compute backend.
    dr = pxu.import_module("drjit")
    drb = _load_dr_variant(ndi)
    Ray3f = _Ray3f_Factory(ndi)

    class BoundingBox3f:
        # Dr.JIT-backed 3D bounding box.
        #
        # A bounding box is described by coordinates {pMin, pMax} of two of its diagonal corners.
        #
        # Important
        # ---------
        # We do NOT check if (pMin < pMax) when constructing the BBox: users are left responsible of their actions.
        DRJIT_STRUCT = dict(
            pMin=drb.Array3f,
            pMax=drb.Array3f,
        )

        def __init__(self, pMin=drb.Array3f(), pMax=drb.Array3f()):
            # Parameters
            # ----------
            # pMin: Array3f
            #     (3,) corner coordinate.
            # pMax: Array3f
            #     (3,) corner coordinate.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.pMin = drb.Array3f()
            self.pMin.assign(pMin)

            self.pMax = drb.Array3f()
            self.pMax.assign(pMax)

        def contains(self, p: drb.Array3f) -> drb.Bool:
            # Check if point `p` lies in/on the BBox.
            return dr.all((self.pMin <= p) & (p <= self.pMax))

        def ray_intersect(self, r: Ray3f) -> tuple[drb.Bool, drb.Float, drb.Float]:
            # Compute ray/bbox intersection parameters. [Adapted from Mitsuba3's ray_intersect().]
            #
            # Parameters
            # ----------
            # r: Ray3f
            #
            # Returns
            # -------
            # active: Bool
            #     True if intersection occurs.
            # mint, maxt: Float
            #     Ray parameters `t` such that r(t) touches a BBox border.
            #     The value only makes sense if `active` is enabled.

            # Ensure ray has a nonzero slope on each axis, or that its origin on a 0-valued axis is within the box bounds.
            active = dr.all(dr.neq(r.d, 0) | (self.pMin < r.o) | (r.o < self.pMax))

            # Compute intersection intervals for each axis
            d_rcp = dr.rcp(r.d)
            t1 = (self.pMin - r.o) * d_rcp
            t2 = (self.pMax - r.o) * d_rcp

            # Ensure proper ordering
            t1p = dr.minimum(t1, t2)
            t2p = dr.maximum(t1, t2)

            # Intersect intervals
            mint = dr.max(t1p)
            maxt = dr.min(t2p)
            active &= mint <= maxt

            return active, mint, maxt

        def assign(self, bbox: "BoundingBox3f"):
            # See __init__'s docstring for more info.
            self.pMin.assign(bbox.pMin)
            self.pMax.assign(bbox.pMax)

        def __repr__(self) -> str:
            return f"BoundingBox3f(pMin={dr.shape(self.pMin)}, pMax={dr.shape(self.pMax)})"

    return BoundingBox3f


def _build_xrt(ndi: pxd.NDArrayInfo, weighted: bool):
    # Create DrJIT FW/BW transforms.
    #
    # Parameters
    #   weighted: create attenuated FW/BW transforms.
    dr = pxu.import_module("drjit")
    drb = _load_dr_variant(ndi)

    Ray3f = _Ray3f_Factory(ndi)
    BoundingBox3f = _BoundingBox3f_Factory(ndi)

    def ray_step(r: Ray3f) -> Ray3f:
        # Advance ray until next unit-step lattice intersection.
        #
        # Parameters
        #   r(o, d): ray to move. (`d` assumed normalized.)
        # Returns
        #   r_next(o_next, d): next ray position on unit-step lattice intersection.
        eps = 1e-4  # threshold for proximity tests with 0

        # Go to local coordinate system.
        o_ref = dr.floor(r.o)
        r_local = Ray3f(o=r.o - o_ref, d=r.d)

        # Find bounding box for ray-intersection tests.
        on_boundary = r_local.o <= eps
        bbox_border = dr.select(on_boundary, dr.sign(r.d), 1)
        bbox = BoundingBox3f(
            dr.minimum(0, bbox_border),
            dr.maximum(0, bbox_border),
        )

        # Compute step size to closest bounding box wall.
        #   (a1, a2) may contain negative values or Infs.
        #   In any case, we must always choose min(a1, a2) > 0.
        _, a1, a2 = bbox.ray_intersect(r_local)
        a_min = dr.minimum(a1, a2)
        a_max = dr.maximum(a1, a2)
        a = dr.select(a_min >= eps, a_min, a_max)

        # Move ray to new position in global coordinates.
        # r_next located on lattice intersection (up to FP error).
        r_next = Ray3f(o=o_ref + r_local(a), d=r.d)
        return r_next

    def xrt_apply(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        I: drb.Float,
        r: Ray3f,
    ) -> drb.Float:
        # X-Ray Forward-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,0,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (Nx,Ny,Nz) lattice size
        #   I:     (Nx*Ny*Nz,) cell weights \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   P:     (L,) forward-projected samples \in \bR

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, L)  # Forward-Projection samples

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBox3f(drb.Array3f(0), drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        loop = drb.Loop("XRT FW", lambda: (r, r_next, active, P))
        while loop(active):
            # Read (I,) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            weight = dr.gather(type(I), I, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)

            # Update line integral estimates
            P += weight * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return P

    def wxrt_apply(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        I: drb.Float,
        w: drb.Float,
        r: Ray3f,
    ) -> drb.Float:
        # Weighted X-Ray Forward-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,0,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (Nx,Ny,Nz) lattice size
        #   I:     (Nx*Ny*Nz,) cell weights \in \bR [C-order]
        #   w:     (Nx*Ny*Nz,) cell decay rates \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   P:     (L,) forward-projected samples \in \bR

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, L)  # Forward-Projection samples
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBox3f(drb.Array3f(0), drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        loop = drb.Loop("WXRT FW", lambda: (r, r_next, active, P, d_acc))
        while loop(active):
            # Read (I, w) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            weight = dr.gather(type(I), I, flat_index(idx_I), mask)
            decay = dr.gather(type(w), w, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)
            A = dr.exp(-d_acc)
            B = dr.select(
                dr.eq(decay, 0),
                length,
                (1 - dr.exp(-decay * length)) * dr.rcp(decay),
            )

            # Update line integral estimates
            P += weight * A * B
            d_acc += decay * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return P

    def xrt_adjoint(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        P: drb.Float,
        r: Ray3f,
    ) -> drb.Float:
        # X-Ray Back-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,0,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (Nx,Ny,Nz) lattice size
        #   P:     (L,) X-Ray samples \in \bR
        #   r:     (L,) ray descriptors
        # Returns
        #   I:     (Nx*Ny*Nz,) back-projected samples \in \bR [C-order]

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBox3f(drb.Array3f(0), drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        active &= dr.neq(P, 0)
        loop = drb.Loop("XRT BW", lambda: (r, r_next, active))
        while loop(active):
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)

            # Update back-projections
            dr.scatter_reduce(dr.ReduceOp.Add, I, P * length, flat_index(idx_I), mask)

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return I

    def wxrt_adjoint(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        P: drb.Float,
        w: drb.Float,
        r: Ray3f,
    ) -> drb.Float:
        # Weighted X-Ray Back-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,0,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (Nx,Ny,Nz) lattice size
        #   P:     (L,) X-Ray samples \in \bR
        #   w:     (Nx*Ny*Nz,) cell decay rates \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   I:     (Nx*Ny*Nz,) back-projected samples \in \bR [C-order]

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        L = dr.shape(P)[0]
        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBox3f(drb.Array3f(0), drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        active &= dr.neq(P, 0)
        loop = drb.Loop("WXRT BW", lambda: (r, r_next, active, d_acc))
        while loop(active):
            # Read (w,) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            decay = dr.gather(type(w), w, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)
            A = dr.exp(-d_acc)
            B = dr.select(
                dr.eq(decay, 0),
                length,
                (1 - dr.exp(-decay * length)) * dr.rcp(decay),
            )

            # Update back-projections
            dr.scatter_reduce(dr.ReduceOp.Add, I, P * A * B, flat_index(idx_I), mask)
            d_acc += decay * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return I

    if weighted:
        return wxrt_apply, wxrt_adjoint
    else:
        return xrt_apply, xrt_adjoint
