import dataclasses

import numpy as np

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu
from pyxu.operator.linop.fft.fft import FFT
from pyxu.operator.linop.pad import Pad
from pyxu.operator.linop.stencil._stencil import _Stencil
from pyxu.operator.linop.stencil.stencil import Stencil

__all__ = [
    "FFTCorrelate",
    "FFTConvolve",
    "FFTCorrelateFast",
    "FFTConvolveFast",
]


KernelInfo = dataclasses.make_dataclass(
    "KernelInfo",
    fields=["_kernel", "_center"],
)


class FFTCorrelate(Stencil):
    r"""
    Multi-dimensional FFT-based correlation.

    :py:class:`~pyxu.operator.FFTCorrelate` has the same interface as :py:class:`~pyxu.operator.Stencil`.

    .. rubric:: Implementation Notes

    * :py:class:`~pyxu.operator.FFTCorrelate` can scale to much larger kernels than :py:class:`~pyxu.operator.Stencil`.
    * Kernels must be small enough to fit in memory, i.e. unbounded kernels are not allowed.
    * Kernels should be supplied an NUMPY/CUPY arrays. DASK arrays will be evaluated if provided.
    * :py:class:`~pyxu.operator.FFTCorrelate` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `kernel`, or DASK arrays where the chunk backend matches the
      init-supplied kernel backend.
    * A warning is emitted if inputs must be cast to the kernel dtype.
    * The input array is transformed by calling :py:class:`~pyxu.operator.FFT`.
    * When operating on DASK inputs, the kernel DFT is computed per chunk at the best size to handle the inputs.
      This was deemed preferable than pre-computing a huge kernel DFT once, then sending it to each worker process to
      compute its chunk.

    See Also
    --------
    :py:class:`~pyxu.operator.Stencil`,
    :py:class:`~pyxu.operator.FFTConvolve`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: Stencil.KernelSpec,
        center: _Stencil.IndexSpec,
        mode: Pad.ModeSpec = "constant",
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) input dimensions.
        kernel: ~pyxu.operator.Stencil.KernelSpec
            Kernel coefficients.  Two forms are accepted:

            * NDArray of rank-:math:`D`: denotes a non-seperable kernel.
            * tuple[NDArray_1, ..., NDArray_D]: a sequence of 1D kernels such that dimension[k] is filtered by kernel
              `kernel[k]`, that is:

              .. math::

                 k = k_1 \otimes\cdots\otimes k_D,

              or in Python: ``k = functools.reduce(numpy.multiply.outer, kernel)``.

        center: ~pyxu.operator._Stencil.IndexSpec
            (i1,...,iD) index of the kernel's center.

            `center` defines how a kernel is overlaid on inputs to produce outputs.

        mode: str, :py:class:`list` ( str )
            Boundary conditions.  Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (zero-padding)
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: dimension[k] uses `mode[k]` as boundary condition.

            (See :py:func:`numpy.pad` for details.)
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        kwargs: dict
            Extra kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        """
        super().__init__(
            dim_shape=dim_shape,
            kernel=kernel,
            center=center,
            mode=mode,
            enable_warnings=enable_warnings,
        )
        self._fft_kwargs = kwargs  # Extra kwargs passed to FFT()

    def configure_dispatcher(self, **kwargs):
        raise NotImplementedError("Irrelevant for FFT-backed filtering.")

    # Helper routines (internal) ----------------------------------------------
    @staticmethod
    def _compute_pad_width(_kernel, _center, _mode) -> Pad.WidthSpec:
        N = _kernel[0].ndim
        pad_width = [None] * N
        for i in range(N):
            if _mode[i] == "constant":
                pad_width[i] = (0, 0)
            else:
                if len(_kernel) == 1:  # non-seperable filter
                    n = _kernel[0].shape[i]
                else:  # seperable filter(s)
                    n = _kernel[i].size

                # 1. Pad/Trim operators are shared amongst [apply,adjoint]():
                #    lhs/rhs are thus padded equally.
                # 2. Pad width must match kernel dimensions to retain border effects.
                pad_width[i] = (n - 1, n - 1)
        return tuple(pad_width)

    @staticmethod
    def _init_fw(_kernel, _center) -> list:
        # Initialize kernels used in apply().
        # The returned objects must have the following fields:
        # * _kernel: ndarray[float] (D,)
        # * _center: ndarray[int] (D,)

        # Store kernels in convolution form.
        _st_fw = [None] * len(_kernel)
        _kernel, _center = Stencil._bw_equivalent(_kernel, _center)
        for i, (k_fw, c_fw) in enumerate(zip(_kernel, _center)):
            _st_fw[i] = KernelInfo(k_fw, c_fw)
        return _st_fw

    @staticmethod
    def _init_bw(_kernel, _center) -> list:
        # Initialize kernels used in adjoint().
        # The returned objects must have the following fields:
        # * _kernel: ndarray[float] (D,)
        # * _center: ndarray[int] (D,)

        # Store kernels in convolution form.
        _st_bw = [None] * len(_kernel)
        for i, (k_bw, c_bw) in enumerate(zip(_kernel, _center)):
            _st_bw[i] = KernelInfo(k_bw, c_bw)
        return _st_bw

    def _stencil_chain(self, x: pxt.NDArray, stencils: list) -> pxt.NDArray:
        # Apply sequence of stencils to `x`.
        #
        # x: (..., M1,...,MD)
        # z: (..., M1,...,MD)

        # Contrary to Stencil._stencil_chain(), the `stencils` parameter is picklable directly.
        def _chain(x, stencils, fft_kwargs):
            xp = pxu.get_array_module(x)
            xpf = FFT.fft_backend(xp)

            # Compute constants -----------------------------------------------
            if uni_kernel := (len(stencils) == 1):
                M = np.r_[stencils[0]._kernel.shape]
                dim_rank = stencils[0]._kernel.ndim
            else:
                M = np.array([st._kernel.size for st in stencils])
                dim_rank = len(stencils)
            Np = np.r_[x.shape[-dim_rank:]]
            L = FFT.next_fast_len(Np + M - 1, xp=xp)
            axes = tuple(range(-dim_rank, 0))

            # Apply stencils in DFT domain ------------------------------------
            fft = FFT(L, axes, **fft_kwargs)
            Z = fft.capply(x)
            if uni_kernel:
                if len(M) == 1:  # 1D kernel
                    K = xpf.fft(stencils[0]._kernel, n=L[0])
                else:  # ND kernel
                    K = fft.capply(stencils[0]._kernel)
                Z *= K
            else:
                for ax, st in enumerate(stencils):
                    K = xpf.fft(st._kernel, n=L[ax], axis=ax)
                    Z *= K
            z = fft.cpinv(Z, damp=0).real

            # Extract ROI -----------------------------------------------------
            if uni_kernel:
                center = stencils[0]._center
            else:
                center = [st._center[i] for (i, st) in enumerate(stencils)]
            extract = [slice(c, c + n) for (c, n) in zip(center, Np)]
            return z[..., *extract]

        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi == pxd.NDArrayInfo.DASK:
            # Compute (depth,boundary) values for [overlap,trim_internal]()
            N_stack = x.ndim - self.dim_rank
            depth = {ax: 0 for ax in range(x.ndim)}
            for ax in range(self.dim_rank):
                if len(stencils) == 1:  # non-seperable filter
                    n = stencils[0]._kernel.shape[ax]
                    c = stencils[0]._center[ax]
                else:  # seperable filter(s)
                    n = stencils[ax]._kernel.size
                    c = stencils[ax]._center[ax]
                max_dist = max(c, n - c)
                # depth[N_stack + ax] = n - 1
                depth[N_stack + ax] = max_dist
            boundary = 0

            xp = ndi.module()
            x_overlap = xp.overlap.overlap(  # Share padding between chunks
                x,
                depth=depth,
                boundary=boundary,
            )
            z_overlap = x_overlap.map_blocks(  # Map _chain() to each chunk
                func=_chain,
                dtype=x.dtype,
                chunks=x_overlap.chunks,
                meta=x._meta,
                # extra _chain() kwargs -------------------
                stencils=stencils,
                fft_kwargs=self._fft_kwargs,
            )
            z = xp.overlap.trim_internal(  # Trim inter-chunk excess
                z_overlap,
                axes=depth,
                boundary=boundary,
            )
        else:
            z = _chain(x, stencils, self._fft_kwargs)
        return z


class FFTConvolve(FFTCorrelate):
    r"""
    Multi-dimensional FFT-based convolution.

    :py:class:`~pyxu.operator.FFTConvolve` has the same interface as :py:class:`~pyxu.operator.Convolve`.

    See :py:class:`~pyxu.operator.FFTCorrelate` for implementation notes.

    See Also
    --------
    :py:class:`~pyxu.operator.Stencil`,
    :py:class:`~pyxu.operator.FFTCorrelate`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: Stencil.KernelSpec,
        center: _Stencil.IndexSpec,
        mode: Pad.ModeSpec = "constant",
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        See :py:meth:`~pyxu.operator.FFTCorrelate.__init__` for a description of the arguments.
        """
        super().__init__(
            dim_shape=dim_shape,
            kernel=kernel,
            center=center,
            mode=mode,
            enable_warnings=enable_warnings,
            **kwargs,
        )

        # flip FW/BW kernels (& centers)
        self._st_fw, self._st_bw = self._st_bw, self._st_fw


class FFTCorrelateFast(FFTCorrelate):
    r"""
    Multi-dimensional FFT-based correlation.

    :py:class:`~pyxu.operator.FFTCorrelate` has the same interface as :py:class:`~pyxu.operator.Stencil`.

    .. rubric:: Implementation Notes

    * :py:class:`~pyxu.operator.FFTCorrelate` can scale to much larger kernels than :py:class:`~pyxu.operator.Stencil`.
    * Kernels must be small enough to fit in memory, i.e. unbounded kernels are not allowed.
    * Kernels should be supplied an NUMPY/CUPY arrays. DASK arrays will be evaluated if provided.
    * :py:class:`~pyxu.operator.FFTCorrelate` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `kernel`, or DASK arrays where the chunk backend matches the
      init-supplied kernel backend.
    * A warning is emitted if inputs must be cast to the kernel dtype.
    * The input array is transformed by calling :py:class:`~pyxu.operator.FFT`.
    * When operating on DASK inputs, the kernel DFT is computed per chunk at the best size to handle the inputs.
      This was deemed preferable than pre-computing a huge kernel DFT once, then sending it to each worker process to
      compute its chunk.

    See Also
    --------
    :py:class:`~pyxu.operator.Stencil`,
    :py:class:`~pyxu.operator.FFTConvolve`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: Stencil.KernelSpec,
        center: _Stencil.IndexSpec,
        chunksize: pxt.NDArrayShape = None,
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) input dimensions.
        kernel: ~pyxu.operator.Stencil.KernelSpec
            Kernel coefficients.  Two forms are accepted:

            * NDArray of rank-:math:`D`: denotes a non-seperable kernel.
            * tuple[NDArray_1, ..., NDArray_D]: a sequence of 1D kernels such that dimension[k] is filtered by kernel
              `kernel[k]`, that is:

              .. math::

                 k = k_1 \otimes\cdots\otimes k_D,

              or in Python: ``k = functools.reduce(numpy.multiply.outer, kernel)``.

        center: ~pyxu.operator._Stencil.IndexSpec
            (i1,...,iD) index of the kernel's center.

            `center` defines how a kernel is overlaid on inputs to produce outputs.

        mode: str, :py:class:`list` ( str )
            Boundary conditions.  Multiple forms are accepted:

            * str: unique mode shared amongst dimensions.
              Must be one of:

              * 'constant' (zero-padding)
              * 'wrap'
              * 'reflect'
              * 'symmetric'
              * 'edge'
            * tuple[str, ...]: dimension[k] uses `mode[k]` as boundary condition.

            (See :py:func:`numpy.pad` for details.)
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        kwargs: dict
            Extra kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        """
        super().__init__(
            dim_shape=dim_shape,
            kernel=kernel,
            center=center,
            mode="constant",
            enable_warnings=enable_warnings,
        )
        self._fft_kwargs = kwargs  # Extra kwargs passed to FFT()

        self._chunksize = self.dim_shape if chunksize is None else chunksize
        assert (
            self._chunksize.ndim == self.dim_shape.ndim
        ), "chunksize and dim_shape must have the same number of dimensions"
        # msg = "chunksize must be larger than the kernel size"
        # if len(kernel) == 1:
        #     assert all([cs <= ds for (cs, ds) in zip(self._chunksize == self.kernel[0].shape)])
        # else:
        #     assert all([cs <= ds for (cs, ds) in zip(self._chunksize.ndim == self.dim_shape.ndim)])

        self._axes = tuple(range(-self.dim_rank, 0))
        self._compute_constants()

    def _compute_constants(self):
        # Compute constants -----------------------------------------------
        Np = np.r_[self._chunksize]
        self._st_fw = self._compute_fkernels(self._st_fw, Np, **self._fft_kwargs)
        self._st_bw = self._compute_fkernels(self._st_bw, Np, **self._fft_kwargs)

    @staticmethod
    def _compute_fkernels(stencils, Np, **fft_kwargs):
        ndi = pxd.NDArrayInfo.from_obj(stencils[0]._kernel)
        xp = ndi.module()
        xpf = FFT.fft_backend(xp)
        axes = tuple(range(-len(Np), 0))

        # Apply stencils in DFT domain ------------------------------------
        if uni_kernel := (len(stencils) == 1):
            M = np.r_[stencils[0]._kernel.shape]
        else:
            M = np.array([st._kernel.size for st in stencils])

        L = FFT.next_fast_len(Np + M - 1, xp=xp)
        fft = FFT(L, axes, **fft_kwargs)
        if uni_kernel:
            if len(M) == 1:  # 1D kernel
                K = xpf.fft(stencils[0]._kernel, n=L[0])
            else:  # ND kernel
                K = fft.capply(stencils[0]._kernel)
            return [
                {"K": K, "L": L, "center": stencils[0]._center},
            ]
        else:
            K = [None] * Np.ndim
            return [
                {
                    "K": xpf.fft(st._kernel, n=L[ax], axis=ax),
                    "L": L[ax],
                    "center": st._center,
                }
                for ax, st in enumerate(stencils)
            ]

    def _stencil_chain(self, x: pxt.NDArray, stencils: list) -> pxt.NDArray:
        # Apply sequence of stencils to `x`.
        #
        # x: (..., M1,...,MD)
        # z: (..., M1,...,MD)
        ndi = pxd.NDArrayInfo.from_obj(x)
        xp = ndi.module()
        if ndi == pxd.NDArrayInfo.DASK:
            assert x.chunksize == self._chunksize, "specified chunksize and Dask Array chunksize don't match"
            # Compute (depth,boundary) values for [overlap,trim_internal]()
            N_stack = x.ndim - self.dim_rank
            depth = {ax: 0 for ax in range(x.ndim)}
            for ax in range(self.dim_rank):
                if len(stencils) == 1:  # non-seperable filter
                    n = stencils[0]["K"].shape[ax]
                    c = stencils[0]["center"][ax]
                else:  # seperable filter(s)
                    n = stencils[ax]["K"].size
                    c = stencils[ax]["center"][ax]
                max_dist = max(c, n - c)
                # depth[N_stack + ax] = n - 1
                depth[N_stack + ax] = max_dist
            boundary = 0

            x_overlap = xp.overlap.overlap(  # Share padding between chunks
                x,
                depth=depth,
                boundary=boundary,
            )
            z_overlap = x_overlap.map_blocks(  # Map _stencil_chain() to each chunk
                func=self._stencil_chain,
                dtype=x.dtype,
                chunks=x_overlap.chunks,
                meta=x._meta,
                stencils=stencils,
            )
            z = xp.overlap.trim_internal(  # Trim inter-chunk excess
                z_overlap,
                axes=depth,
                boundary=boundary,
            )
            return z
        else:
            # Apply stencils in DFT domain ------------------------------------
            axes = tuple(range(-self.dim_rank, 0))
            if uni_kernel := (len(stencils) == 1):
                L = stencils[0]["L"]
            else:
                L = [st["L"] for st in stencils]
            fft = FFT(L, axes, **self._fft_kwargs)
            Z = fft.capply(x)

            if uni_kernel:
                K = stencils[0]["K"]
                Z *= K
            else:
                for st in stencils:
                    K = st["K"]
                    Z *= K
            z = fft.cpinv(Z, damp=0).real

            # Extract ROI -----------------------------------------------------
            if uni_kernel:
                center = stencils[0]["center"]
            else:
                center = [st["center"][i] for (i, st) in enumerate(stencils)]
            extract = [slice(c, c + n) for (c, n) in zip(center, self._chunksize)]
            return z[..., *extract]


class FFTConvolveFast(FFTCorrelateFast):
    r"""
    Multi-dimensional FFT-based convolution.

    :py:class:`~pyxu.operator.FFTConvolve` has the same interface as :py:class:`~pyxu.operator.Convolve`.

    See :py:class:`~pyxu.operator.FFTCorrelate` for implementation notes.

    See Also
    --------
    :py:class:`~pyxu.operator.Stencil`,
    :py:class:`~pyxu.operator.FFTCorrelate`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        kernel: Stencil.KernelSpec,
        center: _Stencil.IndexSpec,
        chunksize: pxt.NDArrayShape = None,
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        See :py:meth:`~pyxu.operator.FFTCorrelate.__init__` for a description of the arguments.
        """
        super().__init__(
            dim_shape=dim_shape,
            kernel=kernel,
            center=center,
            chunksize=chunksize,
            enable_warnings=enable_warnings,
            **kwargs,
        )

        # flip FW/BW kernels (& centers)
        self._st_fw, self._st_bw = self._st_bw, self._st_fw
