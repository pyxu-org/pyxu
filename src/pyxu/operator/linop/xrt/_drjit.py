# Helper classes/functions related to RayXRT.
#
# These are low-level routines NOT meant to be imported by default via `import pyxu.operator`.
# Import this module when/where needed only.

import types

import drjit as dr

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt

DrJitBackend = types.ModuleType  # (drjit.llvm, drjit.cuda)


def _xp2dr(x: pxt.NDArray):
    # Transform NP/CP inputs to format allowing zero-copy casts to {drb.Float, drb.Array[23]f}
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
        import drjit.llvm as drb
    elif ndi == pxd.NDArrayInfo.CUPY:
        import drjit.cuda as drb
    else:
        raise NotImplementedError
    return drb


def _Arrayf_Factory(drb: DrJitBackend, D: int):
    # Load right Array[23]f class
    if D == 2:
        Arrayf = drb.Array2f
    elif D == 3:
        Arrayf = drb.Array3f
    else:
        raise NotImplementedError
    return Arrayf


def _Arrayu_Factory(drb: DrJitBackend, D: int):
    # Load right Array[23]u class
    if D == 2:
        Arrayu = drb.Array2u
    elif D == 3:
        Arrayu = drb.Array3u
    else:
        raise NotImplementedError
    return Arrayu


def _Rayf_Factory(drb: DrJitBackend, D: int):
    # Create a Ray[23]f class associated with a compute backend.
    Arrayf = _Arrayf_Factory(drb, D)

    class Rayf:
        # Dr.JIT-backed ray.
        #
        # Rays take the parametric form
        #     r(t) = o + d * t,
        # where {o, d} \in \bR^{D} and t \in \bR.
        DRJIT_STRUCT = dict(
            o=Arrayf,
            d=Arrayf,
        )

        def __init__(self, o=Arrayf(), d=Arrayf()):
            # Parameters
            # ----------
            # o: Array3f
            #    (D,) ray origin.
            # d: Array3f
            #    (D,) ray direction.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.o = Arrayf()
            self.o.assign(o)

            self.d = Arrayf()
            self.d.assign(d)

        def __call__(self, t: drb.Float) -> Arrayf:
            # Compute r(t).
            #
            # Parameters
            # ----------
            # t: Float
            #
            # Returns
            # -------
            # p: Arrayf
            #    p = r(t)
            return dr.fma(self.d, t, self.o)

        def assign(self, r: "Rayf"):
            # See __init__'s docstring for more info.
            self.o.assign(r.o)
            self.d.assign(r.d)

        def __repr__(self) -> str:
            return f"Rayf(o={dr.shape(self.o)}, d={dr.shape(self.d)})"

    return Rayf


def _BoundingBoxf_Factory(drb: DrJitBackend, D: int):
    # Create a BoundingBox[23]f class associated with a compute backend.
    Arrayf = _Arrayf_Factory(drb, D)
    Rayf = _Rayf_Factory(drb, D)

    class BoundingBoxf:
        # Dr.JIT-backed bounding box.
        #
        # A bounding box is described by coordinates {pMin, pMax} of two of its diagonal corners.
        #
        # Important
        # ---------
        # We do NOT check if (pMin < pMax) when constructing the BBox: users are left responsible of their actions.
        DRJIT_STRUCT = dict(
            pMin=Arrayf,
            pMax=Arrayf,
        )

        def __init__(self, pMin=Arrayf(), pMax=Arrayf()):
            # Parameters
            # ----------
            # pMin: Arrayf
            #     (D,) corner coordinate.
            # pMax: Arrayf
            #     (D,) corner coordinate.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.pMin = Arrayf()
            self.pMin.assign(pMin)

            self.pMax = Arrayf()
            self.pMax.assign(pMax)

        def contains(self, p: Arrayf) -> drb.Bool:
            # Check if point `p` lies in/on the BBox.
            return dr.all((self.pMin <= p) & (p <= self.pMax))

        def ray_intersect(self, r: Rayf) -> tuple[drb.Bool, drb.Float, drb.Float]:
            # Compute ray/bbox intersection parameters. [Adapted from Mitsuba3's ray_intersect().]
            #
            # Parameters
            # ----------
            # r: Rayf
            #
            # Returns
            # -------
            # active: Bool
            #     True if intersection occurs.
            # mint, maxt: Float
            #     Ray parameters `t` such that r(t) touches a BBox border.
            #     The value only makes sense if `active` is enabled.

            # Ensure ray has a nonzero slope on each axis, or that its origin on a 0-valued axis is within the box
            # bounds.
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

        def assign(self, bbox: "BoundingBoxf"):
            # See __init__'s docstring for more info.
            self.pMin.assign(bbox.pMin)
            self.pMax.assign(bbox.pMax)

        def __repr__(self) -> str:
            return f"BoundingBoxf(pMin={dr.shape(self.pMin)}, pMax={dr.shape(self.pMax)})"

    return BoundingBoxf


def _build_xrt(drb: DrJitBackend, D: int, weighted: bool):
    # Create DrJIT FW/BW transforms.
    #
    # Parameters
    #   weighted: create attenuated FW/BW transforms.
    Arrayf = _Arrayf_Factory(drb, D)
    Arrayu = _Arrayu_Factory(drb, D)
    Rayf = _Rayf_Factory(drb, D)
    BoundingBoxf = _BoundingBoxf_Factory(drb, D)

    def ray_step(r: Rayf) -> Rayf:
        # Advance ray until next unit-step lattice intersection.
        #
        # Parameters
        #   r(o, d): ray to move. (`d` assumed normalized.)
        # Returns
        #   r_next(o_next, d): next ray position on unit-step lattice intersection.
        eps = 1e-4  # threshold for proximity tests with 0

        # Go to local coordinate system.
        o_ref = dr.floor(r.o)
        r_local = Rayf(o=r.o - o_ref, d=r.d)

        # Find bounding box for ray-intersection tests.
        on_boundary = r_local.o <= eps
        bbox_border = dr.select(on_boundary, dr.sign(r.d), 1)
        bbox = BoundingBoxf(
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
        r_next = Rayf(o=o_ref + r_local(a), d=r.d)
        return r_next

    def xrt_apply(
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        I: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # X-Ray Forward-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   I:     (N1*...*ND,) cell weights \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   P:     (L,) forward-projected samples \in \bR

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = Arrayu(N[1], 1) if (D == 2) else Arrayu(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, Arrayu(i))  # Arrayf (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, L)  # Forward-Projection samples

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
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
        r: Rayf,
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
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, L)  # Forward-Projection samples
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(drb.Array3f(0), drb.Array3f(N))
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
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        P: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # X-Ray Back-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   P:     (L,) X-Ray samples \in \bR
        #   r:     (L,) ray descriptors
        # Returns
        #   I:     (N1*...*ND,) back-projected samples \in \bR [C-order]

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = Arrayu(N[1], 1) if (D == 2) else Arrayu(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, Arrayu(i))  # Array3f (int-valued) -> UInt32

        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
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
        r: Rayf,
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
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, drb.Array3u(i))  # Array3f (int-valued) -> UInt32

        L = dr.shape(P)[0]
        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(drb.Array3f(0), drb.Array3f(N))
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
