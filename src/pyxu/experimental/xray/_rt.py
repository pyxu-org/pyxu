import warnings

import drjit as dr
import mitsuba as mi
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

        # drjit/mitsuba variables. {Have shapes consistent for xrt_[apply,adjoint]().}
        #   xrt_[apply,adjoint]() only support D=3 case.
        #     -> D=2 case (RT-part) is embedded into 3D.
        drb = _load_dr_variant(self._ndi)
        self._fwd, self._bwd = _get_dr_funcs(self._ndi)
        if len(arg_shape) == 2:
            self._dr = dict(
                o=drb.Array3f(*self._origin, 0),
                pitch=drb.Array3f(*self._pitch, 1),
                N=drb.Array3u(*self._arg_shape, 1),
                r=mi.Ray3f(
                    o=drb.Array3f(*[_xp2dr(_) for _ in self._ray_t.T], 0.5),  # Z-extension mid-point
                    d=drb.Array3f(*[_xp2dr(_) for _ in self._ray_n.T], 0),
                ),
            )
        else:
            self._dr = dict(
                o=drb.Array3f(*self._origin),
                pitch=drb.Array3f(*self._pitch),
                N=drb.Array3u(*self._arg_shape),
                r=mi.Ray3f(
                    o=drb.Array3f(*[_xp2dr(_) for _ in self._ray_t.T]),
                    d=drb.Array3f(*[_xp2dr(_) for _ in self._ray_n.T]),
                ),
            )

    @pxrt.enforce_precision(i="arr")
    @pxu.vectorize(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)

        drb = _load_dr_variant(self._ndi)
        P = self._fwd(
            **self._dr,
            I=drb.Float(_xp2dr(arr)),
        )

        xp = self._ndi.module()
        P = xp.asarray(P)
        return P

    @pxrt.enforce_precision(i="arr")
    @pxu.vectorize(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)

        drb = _load_dr_variant(self._ndi)
        I = self._bwd(  # noqa: E741
            **self._dr,
            P=drb.Float(_xp2dr(arr)),
        )

        xp = self._ndi.module()
        I = xp.asarray(I)  # noqa: E741
        return I

    def _warn_cast(self, arr: pxt.NDArray) -> pxt.NDArray:
        W = pxrt.Width  # shorthand
        if W(arr.dtype) != W.SINGLE:
            msg = f"Only {W.SINGLE}-precision inputs are supported: casting."
            warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=W.SINGLE.value)
        else:
            out = arr
        return out


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
        mi.set_variant("llvm_ad_rgb")
        drb = pxu.import_module("drjit.llvm.ad")
    elif ndi == pxd.NDArrayInfo.CUPY:
        mi.set_variant("cuda_ad_rgb")
        drb = pxu.import_module("drjit.cuda.ad")
    else:
        raise NotImplementedError
    return drb


def _get_dr_funcs(ndi: pxd.NDArrayInfo):
    # Create DrJIT functions performing fwd/bwd transforms.
    drb = _load_dr_variant(ndi)

    def ray_step(r: mi.Ray3f) -> mi.Ray3f:
        # Advance ray until next unit-step lattice intersection.
        #
        # Parameters
        #   r(o, d): ray to move. (`d` assumed normalized.)
        # Returns
        #   r_next(o_next, d): next ray position on unit-step lattice intersection.
        eps = 1e-4  # threshold for proximity tests with 0

        # Go to local coordinate system.
        o_ref = dr.floor(r.o)
        r_local = mi.Ray3f(o=r.o - o_ref, d=r.d)

        # Find bounding box for ray-intersection tests.
        on_boundary = r_local.o <= eps
        any_boundary = dr.any(on_boundary)
        bbox_border = dr.select(
            any_boundary,
            dr.select(on_boundary, dr.sign(r.d), 1),
            1,
        )
        bbox = mi.BoundingBox3f(
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
        r_next = mi.Ray3f(o=o_ref + r_local(a), d=r.d)
        return r_next

    def xrt_apply(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        I: drb.Float,
        r: mi.Ray3f,
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
        r = mi.Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )

        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, mi.Point3u(i))  # Point3f (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, shape=L)  # Forward-Projection samples
        idx_P = dr.arange(drb.UInt32, L)

        # Move (intersecting) rays to volume surface
        bbox_vol = mi.BoundingBox3f(0, drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o = dr.select(active, r(a_min), r.o)

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
            r = r_next
            r_next = ray_step(r)
            active &= bbox_vol.contains(r_next.o)
        return P

    def xrt_adjoint(
        o: drb.Array3f,
        pitch: drb.Array3f,
        N: drb.Array3u,
        P: drb.Float,
        r: mi.Ray3f,
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
        #   I: (Nx*Ny*Nz,) back-projected samples \in \bR [C-order]

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = mi.Ray3f(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )

        stride = drb.Array3u(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, mi.Point3u(i))  # Point3f (int-valued) -> UInt32

        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)

        # Move (intersecting) rays to volume surface
        bbox_vol = mi.BoundingBox3f(0, drb.Array3f(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o = dr.select(active, r(a_min), r.o)

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
            r = r_next
            r_next = ray_step(r)
            active &= bbox_vol.contains(r_next.o)
        return I

    return xrt_apply, xrt_adjoint
