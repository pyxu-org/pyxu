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
    # See XRayTransform() for instantiation details.
    def __init__(
        self,
        arg_shape,
        origin,
        pitch,
        n_spec,
        t_spec,
        enable_warnings: bool = True,
    ):
        N_px = np.prod(arg_shape)
        N_l = len(n_spec)
        super().__init__(shape=(N_l, N_px))

        W = pxrt.Width  # shorthand
        n_spec = n_spec.astype(W.SINGLE.value, copy=False)
        t_spec = t_spec.astype(W.SINGLE.value, copy=False)

        # Internal variables. (Have shapes consistent with user inputs.)
        self._arg_shape = arg_shape  # (D,)
        self._origin = origin  # (D,)
        self._pitch = pitch  # (D,)
        self._ray_n = n_spec  # (N_l, D)
        self._ray_t = t_spec  # (N_l, D)
        self._ndi = pxd.NDArrayInfo.from_obj(n_spec)
        self._enable_warnings = bool(enable_warnings)

        # Dr.Jit variables. {Have shapes consistent for xrt_[apply,adjoint]().}
        #   xrt_[apply,adjoint]() only support D=3 case.
        #     -> D=2 case is embedded into 3D.
        drb = _load_dr_variant(self._ndi)
        Ray3f, self._fwd, self._bwd = _get_dr_obj(self._ndi)
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

    @pxrt.enforce_precision(i="arr")
    @pxu.vectorize(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
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
    if ndi == pxd.NDArrayInfo.NUMPY:
        drb = pxu.import_module("drjit.llvm")
    elif ndi == pxd.NDArrayInfo.CUPY:
        drb = pxu.import_module("drjit.cuda")
    else:
        raise NotImplementedError
    return drb


def _get_dr_obj(ndi: pxd.NDArrayInfo):
    # Create DrJIT objects needed for fwd/bwd transforms.
    import drjit as dr

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
        # (a1, a2) may contain negative values or Infs.
        # In any case, we must always choose min(a1, a2) > 0.
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
        P = dr.zeros(drb.Float, shape=L)  # Forward-Projection samples
        idx_P = dr.arange(drb.UInt32, L)

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBox3f(drb.Array3f(0), drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        loop = drb.Loop("XRT FW", lambda: (r, r_next, active))
        while loop(active):
            length = dr.norm((r_next.o - r.o) * pitch)

            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            weight = dr.gather(
                type(I),
                I,
                flat_index(idx_I),
                active & dr.all(idx_I >= 0),
                # Careful to disable out-of-bound queries.
                # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
            )

            # Update line integral estimates
            dr.scatter_reduce(dr.ReduceOp.Add, P, weight * length, idx_P, active)

            # Walk to next lattice intersection.
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
            length = dr.norm((r_next.o - r.o) * pitch)

            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            dr.scatter_reduce(
                dr.ReduceOp.Add,
                I,
                P * length,
                flat_index(idx_I),
                active & dr.all(idx_I >= 0),
                # Careful to disable out-of-bound queries.
                # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
            )

            # Walk to next lattice intersection.
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return I

    return Ray3f, xrt_apply, xrt_adjoint
