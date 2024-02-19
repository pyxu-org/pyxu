import collections
import collections.abc as cabc
import functools
import operator
import warnings

import numpy as np
import scipy.optimize as sopt

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.math as pxm
import pyxu.math.cluster as pxm_cl
import pyxu.runtime as pxrt
import pyxu.util as pxu

isign_default = 1
eps_default = 1e-4
upsampfac_default = 2
T_default = 2 * np.pi
Tc_default = 0
enable_warnings_default = True

__all__ = [
    "NUFFT1",
    "NUFFT2",
    "NUFFT3",
]


class NUFFT1(pxa.LinOp):
    r"""
    Type-1 Non-Uniform FFT :math:`\mathbb{A}: \mathbb{C}^{M} \to \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

    NUFFT1 approximates, up to a requested relative accuracy :math:`\varepsilon \geq 0`, the following exponential sum:

      .. math::

         v_{\mathbf{n}} = (\mathbf{A} \mathbf{w})_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \cdot 2\pi \langle \mathbf{n},
         \mathbf{x}_{m} / \mathbf{T} \rangle},

      where

      * :math:`s \in \pm 1` defines the sign of the transform;
      * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
        :math:`L_{d} = 2 * N_{d} + 1`;
      * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]` are non-uniform support points;
      * :math:`\mathbf{w} \in \mathbb{C}^{M}` are weights associated with :math:`\{\mathbf{x}\}_{m=1}^{M}`;
      * :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}` and :math:`\mathbf{T_{c}} \in \mathbb{R}^{D}`.

      Concretely NUFFT1 computes approximations to the (scaled) Fourier Series coefficients of the :math:`T`-periodic
      function:

      .. math::

         \tilde{f}(\mathbf{x}) = \sum_{\mathbf{q} \in \mathbb{Z}^{D}} \sum_{m=1}^{M} w_{m} \delta(\mathbf{x} -
         \mathbf{x}_{m} - \mathbf{q} \odot \mathbf{T}),

         v_{\mathbf{n}} = \left( \prod_{d} T_{d} \right) \tilde{f}_{\mathbf{n}}^{FS}.

    .. rubric:: Implementation Notes

    * :py:func:`~pyxu.operator.NUFFT1` instances are **not arraymodule-agnostic**: they will only work with NDArrays
      belonging to the same array module as `x`.
    * :py:class:`~pyxu.operator.NUFFT1` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    """

    def __init__(
        self,
        x: pxt.NDArray,
        N: tuple[int],
        *,
        isign: int = isign_default,
        eps: float = eps_default,
        spp: tuple[int] = None,
        upsampfac: tuple[float] = upsampfac_default,
        T: tuple[float] = T_default,
        Tc: tuple[float] = Tc_default,
        enable_warnings: bool = enable_warnings_default,
        fft_kwargs: dict = None,
        spread_kwargs: dict = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\mathbf{x}_{m} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
            \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]`.
        N: int, tuple[int]
            Number of coefficients [-N,...,N] to compute per dimension.
        isign: 1, -1
            Sign :math:`s` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon \geq 0`. (See also `spp`.)
        spp: int, tuple[int]
            Samples-per-pulse, i.e. the width of the spreading kernel in each dimension.  Must be odd-valued. Supplying
            `spp` is an alternative to using `eps`, however only one can be non-`None` at a time.
        upsampfac: float, tuple[float]
            NUFFT upsampling factors :math:`\sigma_{d} > 1`.
        T: float, tuple[float]
            (D,) scalar factors :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}`, i.e. the periodicity of
            :math:`f(\mathbf{x})`.
        Tc: float, tuple[float]
            (D,) center of one period :math:`T_{c} \in \mathbb{R}^{D}`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        fft_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        spread_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu.operator.UniformSpread`.
        """
        # Put all variables in canonical form & validate ----------------------
        #   x: (M, D) array (NUMPY/CUPY/DASK)
        #   N: (D,) int
        #   isign: {-1, +1}
        #   eps: float
        #   spp: (D,) int
        #   upsampfac: (D,) float
        #   T: (D,) float
        #   Tc: (D,) float
        #   fft_kwargs: dict
        #   spread_kwargs: dict
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape
        N = self._as_seq(N, D, int)
        isign = isign // abs(isign)
        upsampfac = self._as_seq(upsampfac, D, float)
        T = self._as_seq(T, D, float)
        Tc = self._as_seq(Tc, D, float)
        spp = self._as_seq(spp, D, _type=None if spp is None else int)
        if fft_kwargs is None:
            fft_kwargs = dict()
        if spread_kwargs is None:
            spread_kwargs = dict()

        assert (N > 0).all()
        if eps_provided := eps is not None:
            assert 0 < eps < 1
        if spp_provided := all(s is not None for s in spp):  # user provided `spp`
            assert (spp > 0).all() & (spp % 2 == 1).all()
        assert operator.or_(
            eps_provided and (not spp_provided),
            (not eps_provided) and spp_provided,
        ), "[eps,spp] Only one of (eps, spp) can be provided at a time."
        assert (upsampfac > 1).all()
        assert (T > 0).all()

        # Initialize Operator -------------------------------------------------
        self.cfg = self._init_metadata(N, isign, eps, upsampfac, T, Tc, spp)
        super().__init__(
            dim_shape=(M, 2),
            codim_shape=(*self.cfg.L, 2),
        )
        self._x = pxrt.coerce(x)
        self._enable_warnings = bool(enable_warnings)
        self.lipschitz = np.sqrt(self.cfg.L.prod() * M)
        self._init_ops(fft_kwargs, spread_kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        # 1. Spread over periodic boundaries
        arr = xp.moveaxis(arr, -1, 0)  # (2,..., M)
        g = self._spread.apply(arr)  # (2,..., fft1+2P1,...,fftD+2PD)
        g = xp.moveaxis(g, 0, -1)  # (..., fft1+2P1,...,fftD+2PD,2)

        # 2. Remove periodic excess from lattice, but apply periodic effect beforehand
        g = self._pad.adjoint(g)  # (..., fft1,...,fftD,2)

        # 3. FFS of up-sampled gridded data
        scale = xp.array([1, -self.cfg.isign], dtype=arr.dtype)
        g *= scale
        g_FS = self._ffs.apply(g)  # (..., fft1,...,fftD,2)
        g_FS *= scale

        # 4. Remove up-sampled sections
        g_FS = self._trim.apply(g_FS)  # (..., L1,...,LD,2)

        # 5. Correct for spreading effect
        psi_FS = self._kernelFS(xp, g_FS.dtype, True)
        out = pxm.hadamard_outer(g_FS, *psi_FS)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.
        """
        x = pxu.view_as_real(pxu.require_viewable(arr))  # (..., M,2)
        y = self.apply(x)  # (..., L1,...,LD,2)
        out = pxu.view_as_complex(pxu.require_viewable(y))  # (..., L1,...,LD)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        # 1. Correct  for spreading effect
        psi_FS = self._kernelFS(xp, arr.dtype, True)
        g_FS = pxm.hadamard_outer(arr, *psi_FS)  # (..., L1,...,LD,2)

        # 2. Go to up-sampled grid
        g_FS = self._trim.adjoint(g_FS)  # (..., fft1,...,fftD,2)

        # 3. FFS of up-sampled gridded data
        scale = xp.array([1, -self.cfg.isign], dtype=arr.dtype)
        g_FS *= scale
        g = self._ffs.adjoint(g_FS)  # (..., fft1,...,fftD,2)
        g *= scale

        # 4. Extend FFS mesh with periodic border effects
        g = self._pad.apply(g)  # (..., fft1+2P1,...,fftD+2PD,2)

        # 5. Interpolate over periodic boundaries
        g = xp.moveaxis(g, -1, 0)  # (2,..., fft1+2P1,...,fftD+2PD)
        out = self._spread.adjoint(g)  # (2,..., M)
        out = xp.moveaxis(out, 0, -1)  # (..., M,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.
        """
        x = pxu.view_as_real(pxu.require_viewable(arr))  # (..., L1,...,LD,2)
        y = self.adjoint(x)  # (..., M,2)
        out = pxu.view_as_complex(pxu.require_viewable(y))  # (..., M)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `x`-backend ... ------------------------------
        xp = pxu.get_array_module(self._x)

        A = xp.stack(  # (L1,...,LD, D)
            xp.meshgrid(
                *[xp.arange(-n, n + 1) for n in self.cfg.N],
                indexing="ij",
            ),
            axis=-1,
        )
        B = xp.exp(  # (L1,...,LD, M)
            (2j * self.cfg.isign * np.pi)
            * xp.tensordot(
                A,
                self._x / self.cfg.T,
                axes=[[-1], [-1]],
            )
        )

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        C = xp.array(
            pxu.as_real_op(B, dim_rank=1),
            dtype=pxrt.Width(dtype).value,
        )
        return C

    # Internal Helpers --------------------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        if arr.dtype == self._x.dtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=self._x.dtype)
        return out

    @staticmethod
    def _as_seq(x, N, _type=None) -> np.ndarray:
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
            return np.r_[tuple(map(_type, _x))]

    @staticmethod
    def _init_metadata(N, isign, eps, upsampfac, T, Tc, spp) -> collections.namedtuple:
        # Compute all NUFFT1 parameters & store in namedtuple with (sub-)fields:
        # [All sequences are NumPy arrays]
        #
        # * D: int                    [Transform Dimensionality]
        # * N: (D,) int               [One-sided FS count /w upsampling]
        # * Ns: (D,) int              [One-sided FS count w/ upsampling]
        # * L: (D,) int               [Two-sided FS size  /w upsampling]
        # * Ls: (D,) int              [Two-sided FS size  w/ upsampling]
        # * T: (D,) float             [Function period]
        # * Tc: (D,) float            [Mid-point of period of interest]
        # * isign: int                [Sign of the exponent]
        # * upsampfac: (D,) float     [Upsampling factor \sigma]
        # * eps: float | None         [Approximate rel-error]
        # * fft_shape: (D,) int       [FFT dimensions]
        # * kernel_spp: (D,) int      [Kernel sample count]
        # * kernel_alpha: (D,) float  [Kernel arg-scale factor]
        # * kernel_beta: (D,) float   [Kernel bandwidth (before arg-scaling)]
        # * z_start: (D,) float       [Lattice start coordinate /w padding]
        # * z_stop: (D,) float        [Lattice stop  coordinate /w padding]
        # * z_num: (D,) int           [Lattice node-count       /w padding]
        # * z_step: (D,) float        [Lattice pitch; useful to have explicitly]
        # * z_pad: (D,) int           [Padding size to add to lattice head/tail for periodic boundary conditions.]
        from pyxu.operator import FFT
        from pyxu.operator.linop.fft._ffs import _FFS

        # FFT parameters
        D = len(N)
        L = 2 * N + 1
        Ns = np.ceil(upsampfac * N).astype(int)  # N^{\sigma}
        Ls = 2 * Ns + 1  # N_{FS}^{\sigma}
        fft_shape = np.r_[FFT.next_fast_len(Ls)]

        # Kernel parameters
        if eps is not None:  # infer `kernel_spp` approximately
            kernel_spp = int(np.ceil(np.log10(1 / eps))) + 1
            kernel_spp += 1 if (kernel_spp % 2 == 0) else 0
            kernel_spp = np.r_[(kernel_spp,) * len(N)]
        else:  # take what the user specified
            kernel_spp = spp
        kernel_alpha = (2 / T) * (fft_shape / kernel_spp)
        kernel_beta = (np.pi * kernel_spp) * (Ns / fft_shape)

        # Lattice parameters
        ffs = _FFS(T=T, Tc=Tc, Nfs=Ls, Ns=fft_shape)
        nodes = ffs.sample_points(xp=np, dtype=pxrt.Width.DOUBLE.value)
        z_start = np.array([n[0] for n in nodes])
        z_stop = np.array([n[-1] for n in nodes])
        z_num = fft_shape
        z_step = T / fft_shape
        z_pad = kernel_spp // 2

        CONFIG = collections.namedtuple(
            "CONFIG",
            field_names=[
                "D",
                "N",
                "Ns",
                "L",
                "Ls",
                "T",
                "Tc",
                "isign",
                "upsampfac",
                "eps",
                "fft_shape",
                "kernel_spp",
                "kernel_alpha",
                "kernel_beta",
                "z_start",
                "z_stop",
                "z_num",
                "z_step",
                "z_pad",
            ],
        )
        return CONFIG(
            D=D,
            N=N,
            Ns=Ns,
            L=L,
            Ls=Ls,
            T=T,
            Tc=Tc,
            isign=isign,
            upsampfac=upsampfac,
            eps=eps,  # may be None
            fft_shape=fft_shape,
            kernel_spp=kernel_spp,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            z_start=z_start,
            z_stop=z_stop,
            z_num=z_num,
            z_step=z_step,
            z_pad=z_pad,
        )

    def _init_ops(self, fft_kwargs, spread_kwargs):
        from pyxu.operator import KaiserBessel, Pad, Trim, UniformSpread
        from pyxu.operator.linop.fft._ffs import _FFS

        self._spread = UniformSpread(  # spreads support points onto uniform lattice
            x=self._x,
            z=dict(
                start=self.cfg.z_start - self.cfg.z_step * self.cfg.z_pad,
                stop=self.cfg.z_stop + self.cfg.z_step * self.cfg.z_pad,
                num=self.cfg.z_num + 2 * self.cfg.z_pad,
            ),
            kernel=[
                KaiserBessel(1, b).argscale(a)
                for (a, b) in zip(
                    self.cfg.kernel_alpha,
                    self.cfg.kernel_beta,
                )
            ],
            enable_warnings=self._enable_warnings,
            **spread_kwargs,
        )
        self._pad = Pad(  # applies periodic border effects after spreading
            dim_shape=(*self.cfg.z_num, 2),
            pad_width=(*self.cfg.z_pad, 0),
            mode="wrap",
        )
        self._ffs = _FFS(  # FFS transform on up-sampled gridded data
            T=self.cfg.T,
            Tc=self.cfg.Tc,
            Nfs=self.cfg.Ls,
            Ns=self.cfg.fft_shape,
            **fft_kwargs,
        )
        self._trim = Trim(  # removes up-sampled FFS sections
            dim_shape=(*self.cfg.fft_shape, 2),
            trim_width=[
                (ns - n, ns - n + tot - ls)
                for (n, ns, ls, tot) in zip(
                    self.cfg.N,
                    self.cfg.Ns,
                    self.cfg.Ls,
                    self.cfg.fft_shape,
                )
            ]
            + [(0, 0)],
        )

    def _kernelFS(self, xp: pxt.ArrayModule, dtype: pxt.DType, invert: bool) -> list[pxt.NDArray]:
        # Returns
        # -------
        # psi_FS: list[NDArray]
        #     (D+1,) kernel FS coefficients (1D), or their reciprocal.
        #     The trailing dimension is just there to operate on real-valued views directly.
        psi_FS = [None] * self.cfg.D + [xp.ones(2, dtype=dtype)]
        f = xp.reciprocal if invert else lambda _: _
        for d in range(self.cfg.D):
            psi = self._spread._kernel[d]
            T = self.cfg.T[d]
            N = self.cfg.N[d]

            pFS = psi.applyF(xp.arange(-N, N + 1) / T) / T
            psi_FS[d] = f(pFS).astype(dtype)
        return psi_FS


def NUFFT2(
    x: pxt.NDArray,
    N: tuple[int],
    *,
    isign: int = isign_default,
    eps: float = eps_default,
    spp: tuple[int] = None,
    upsampfac: tuple[float] = upsampfac_default,
    T: tuple[float] = T_default,
    Tc: tuple[float] = Tc_default,
    enable_warnings: bool = enable_warnings_default,
    fft_kwargs: dict = None,
    spread_kwargs: dict = None,
    **kwargs,
) -> pxt.OpT:
    r"""
    Type-2 Non-Uniform FFT :math:`\mathbb{A}: \mathbb{C}^{L_{1} \times\cdots\times L_{D}} \to \mathbb{C}^{M}`.

    NUFFT2 approximates, up to a requested relative accuracy :math:`\varepsilon \geq 0`, the following exponential sum:

      .. math::

         \mathbf{w}_{m} = (\mathbf{A} \mathbf{v})_{m} = \sum_{\mathbf{n}} v_{\mathbf{n}} e^{j \cdot s \cdot 2\pi \langle \mathbf{n}, \mathbf{x}_{m} / \mathbf{T} \rangle},

      where

      * :math:`s \in \pm 1` defines the sign of the transform;
      * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
        :math:`L_{d} = 2 * N_{d} + 1`;
      * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]` are non-uniform support points;
      * :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` are weights;
      * :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}` and :math:`\mathbf{T_{c}} \in \mathbb{R}^{D}`.

      Concretely NUFFT2 can be interpreted as computing approximately non-uniform samples of a :math:`T`-periodic
      function from its Fourier Series coefficients.  It is the adjoint of a type-1 NUFFT.

    .. rubric:: Implementation Notes

    * :py:func:`~pyxu.operator.NUFFT2` instances are **not arraymodule-agnostic**: they will only work with NDArrays
      belonging to the same array module as `x`.
    * :py:func:`~pyxu.operator.NUFFT2` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.


    Parameters
    ----------
    x: NDArray
        (M, D) support points :math:`\mathbf{x}_{m} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]`.
    N: int, tuple[int]
        Number of coefficients [-N,...,N] to compute per dimension.
    isign: 1, -1
        Sign :math:`s` of the transform.
    eps: float
        Requested relative accuracy :math:`\varepsilon \geq 0`. (See also `spp`.)
    spp: int, tuple[int]
        Samples-per-pulse, i.e. the width of the spreading kernel in each dimension.  Must be odd-valued. Supplying
        `spp` is an alternative to using `eps`, however only one can be non-`None` at a time.
    upsampfac: float, tuple[float]
        NUFFT upsampling factors :math:`\sigma_{d} > 1`.
    T: float, tuple[float]
        (D,) scalar factors :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}`, i.e. the periodicity of :math:`f(\mathbf{x})`.
    Tc: float, tuple[float]
        (D,) center of one period :math:`T_{c} \in \mathbb{R}^{D}`.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    fft_kwargs: dict
        kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
    spread_kwargs: dict
        kwargs forwarded to :py:class:`~pyxu.operator.UniformSpread`.
    """
    op1 = NUFFT1(
        x=x,
        N=N,
        isign=-isign,
        eps=eps,
        spp=spp,
        upsampfac=upsampfac,
        T=T,
        Tc=Tc,
        enable_warnings=enable_warnings,
        fft_kwargs=fft_kwargs,
        spread_kwargs=spread_kwargs,
        **kwargs,
    )
    op2 = op1.T
    op2._name = "NUFFT2"

    # Expose c[apply,adjoint]()
    op2.capply = op1.cadjoint
    op2.cadjoint = op1.capply

    return op2
