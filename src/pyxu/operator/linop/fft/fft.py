import functools
import inspect

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "FFT",
]


class FFT(pxa.NormalOp):
    r"""
    Multi-dimensional Discrete Fourier Transform (DFT) :math:`A: \mathbb{C}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.

    The FFT is defined as follows:

    .. math::

       (A \, \mathbf{x})[\mathbf{k}]
       =
       \sum_{\mathbf{n}} \mathbf{x}[\mathbf{n}]
       \exp\left[-j 2 \pi \langle \frac{\mathbf{n}}{\mathbf{N}}, \mathbf{k} \rangle \right],

    .. math::

       (A^{*} \, \hat{\mathbf{x}})[\mathbf{n}]
       =
       \sum_{\mathbf{k}} \hat{\mathbf{x}}[\mathbf{k}]
       \exp\left[j 2 \pi \langle \frac{\mathbf{n}}{\mathbf{N}}, \mathbf{k} \rangle \right],

    .. math::

       (\mathbf{x}, \, \hat{\mathbf{x}}) \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}},
       \quad
       (\mathbf{n}, \, \mathbf{k}) \in \{0, \ldots, M_{1}-1\} \times\cdots\times \{0, \ldots, M_{D}-1\}.

    The DFT is taken over any number of axes by means of the Fast Fourier Transform algorithm (FFT).


    .. rubric:: Implementation Notes

    * The CPU implementation uses `SciPy's FFT implementation <https://docs.scipy.org/doc/scipy/reference/fft.html>`_.
    * The GPU implementation uses cuFFT via `CuPy <https://docs.cupy.dev/en/latest/reference/scipy_fft.html>`_.
    * The DASK implementation evaluates the FFT in chunks using the `CZT algorithm
      <https://en.wikipedia.org/wiki/Chirp_Z-transform>`_.

      Caveat: the cost of assembling the DASK graph grows with the total number of chunks; just calling ``FFT.apply()``
      may take a few seconds or more if inputs are highly chunked. Performance is ~7-10x slower than equivalent
      non-chunked NUMPY version (assuming it fits in memory).


    Examples
    --------

    * 1D DFT of a cosine pulse.

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N = 10
         op = FFT(N)

         x = np.cos(2 * np.pi / N * np.arange(N), dtype=complex)  # (N,)
         x_r = pxu.view_as_real(x)                                # (N, 2)

         y_r = op.apply(x_r)                                      # (N, 2)
         y = pxu.view_as_complex(y_r)                             # (N,)
         # [0, N/2, 0, 0, 0, 0, 0, 0, 0, N/2]

         z_r = op.adjoint(op.apply(x_r))                          # (N, 2)
         z = pxu.view_as_complex(z_r)                             # (N,)
         # np.allclose(z, N * x) -> True

    * 1D DFT of a complex exponential pulse.

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N = 10
         op = FFT(N)

         x = np.exp(1j * 2 * np.pi / N * np.arange(N))            # (N,)
         x_r = pxu.view_as_real(x)                                # (N, 2)

         y_r = op.apply(x_r)                                      # (N, 2)
         y = pxu.view_as_complex(y_r)                             # (N,)
         # [0, N, 0, 0, 0, 0, 0, 0, 0, 0]

         z_r = op.adjoint(op.apply(x_r))                          # (N, 2)
         z = pxu.view_as_complex(z_r)                             # (N,)
         # np.allclose(z, N * x) -> True

    * 2D DFT of an image

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N_h, N_w = 10, 8
         op = FFT((N_h, N_w))

         x = np.pad(                                              # (N_h, N_w)
             np.ones((N_h//2, N_w//2), dtype=complex),
             pad_width=((0, N_h//2), (0, N_w//2)),
         )
         x_r = pxu.view_as_real(x)                                # (N_h, N_w, 2)

         y_r = op.apply(x_r)                                      # (N_h, N_w, 2)
         y = pxu.view_as_complex(y_r)                             # (N_h, N_w)

         z_r = op.adjoint(op.apply(x_r))                          # (N_h, N_w, 2)
         z = pxu.view_as_complex(z_r)                             # (N_h, N_w)
         # np.allclose(z, (N_h * N_w) * x) -> True
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) dimensions of the input :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        axes: NDArrayAxis
            Axes over which to compute the FFT. If not given, all axes are used.
        kwargs: dict
            Extra kwargs passed to :py:func:`scipy.fft.fftn` or :py:func:`cupyx.scipy.fft.fftn`.

            Supported parameters for :py:func:`scipy.fft.fftn` are:

                * workers: int = None

            Supported parameters for :py:func:`cupyx.scipy.fft.fftn` are:

                * NOT SUPPORTED FOR NOW

            Default values are chosen if unspecified.
        """
        dim_shape = pxu.as_canonical_shape(dim_shape)
        super().__init__(
            dim_shape=(*dim_shape, 2),
            codim_shape=(*dim_shape, 2),
        )

        if axes is None:
            axes = tuple(range(self.codim_rank - 1))
        axes = pxu.as_canonical_axes(axes, rank=self.codim_rank - 1)
        self._axes = tuple(sorted(set(axes)))  # drop duplicates

        self._kwargs = {
            pxd.NDArrayInfo.NUMPY: dict(
                workers=kwargs.get("workers", None),
            ),
            pxd.NDArrayInfo.CUPY: dict(),
            pxd.NDArrayInfo.DASK: dict(),
        }
        self.lipschitz = self.estimate_lipschitz()

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        M = [self.codim_shape[ax] for ax in self._axes]
        L = np.sqrt(np.prod(M))
        return L

    def gram(self) -> pxt.OpT:
        from pyxu.operator import HomothetyOp

        G = HomothetyOp(
            dim_shape=self.dim_shape,
            cst=self.lipschitz**2,
        )
        return G

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        out /= (self.lipschitz**2) + damp
        return out

    def cpinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) inputs :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) pseudo-inverse :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        """
        out = self.cadjoint(arr)
        out /= (self.lipschitz**2) + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / ((self.lipschitz**2) + damp)
        return op

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxa.UnitOp.svdvals(self, **kwargs) * self.lipschitz
        return D

    def asarray(self, **kwargs) -> pxt.NDArray:
        # We compute 1D transforms per axis, then Kronecker product them.

        # Since non-NP backends may be faulty, we do everything in NUMPY ...
        A_1D = [None] * (D := self.dim_rank - 1)
        for ax in range(D):
            N = self.dim_shape[ax]
            if ax in self._axes:
                n = np.arange(N)
                A_1D[ax] = np.exp((-2j * np.pi / N) * np.outer(n, n))
            else:
                A_1D[ax] = np.eye(N)

        A_ND = functools.reduce(np.multiply.outer, A_1D)
        B_ND = np.transpose(
            A_ND,
            axes=np.r_[
                np.arange(0, 2 * D, 2),
                np.arange(1, 2 * D, 2),
            ],
        )

        # ... then use the backend/precision user asked for.
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        C = xp.array(
            pxu.as_real_op(B_ND, dim_rank=D),
            dtype=pxrt.Width(dtype).value,
        )
        return C

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD,2) inputs :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M1,...,MD,2) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}` viewed
            as a real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., M1,...,MD)
        y = self.capply(x)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., M1,...,MD,2)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) inputs :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        """
        out = self._transform(arr, mode="fw")
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD,2) inputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}` viewed
            as a real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M1,...,MD,2) outputs :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., M1,...,MD)
        y = self.cadjoint(x)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., M1,...,MD,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) inputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) outputs :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        """
        out = self._transform(arr, mode="bw")
        return out

    # Helpers (public) --------------------------------------------------------
    @classmethod
    def fft_backend(cls, xp: pxt.ArrayModule = None):
        """
        Retrieve the namespace containing [i]fftn().

        Parameters
        ----------
        xp: ArrayModule
            Array module used to compute the FFT. (Default: NumPy.)

        Returns
        -------
        xpf: ModuleType
        """
        N = pxd.NDArrayInfo  # short-hand
        if xp is None:
            xp = N.default().module()

        if xp == N.NUMPY.module():
            xpf = pxu.import_module("scipy.fft")
        elif pxd.CUPY_ENABLED and (xp == N.CUPY.module()):
            xpf = pxu.import_module("cupyx.scipy.fft")
        else:
            raise NotImplementedError

        return xpf

    @classmethod
    def next_fast_len(
        cls,
        dim_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
        xp: pxt.ArrayModule = None,
    ) -> pxt.NDArrayShape:
        r"""
        Retrieve the next-best dimensions to perform an FFT.

        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) dimensions of the input :math:`\mathbf{x} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        axes: NDArrayAxis
            Axes over which to compute the FFT. If not given, all axes are used.
        xp: ArrayModule
            Which array module used to compute the FFT. (Default: NumPy.)

        Returns
        -------
        opt_shape: NDArrayShape
            FFT shape (N1,...,ND) >= (M1,...,MD).
        """
        xpf = cls.fft_backend(xp)

        dim_shape = pxu.as_canonical_shape(dim_shape)
        if axes is None:
            axes = tuple(range(len(dim_shape)))
        axes = pxu.as_canonical_axes(axes, rank=len(dim_shape))

        opt_shape = list(dim_shape)
        for ax in axes:
            opt_shape[ax] = xpf.next_fast_len(dim_shape[ax])
        return tuple(opt_shape)

    # Helpers (internal) ------------------------------------------------------
    def _transform(self, x: pxt.NDArray, mode: str) -> pxt.NDArray:
        # Parameters
        # ----------
        # x: NDArray [real/complex]
        #     (..., M1,...,MD) array to transform.
        #    [(..., L1,...,LD), Lk <= Mk works too: will be zero-padded as required.]
        # mode: str
        #     Transform direction:
        #
        #     * 'fw':  fftn(norm="backward")
        #     * 'bw': ifftn(norm="forward")
        #
        # Returns
        # -------
        # y: NDArray [complex]
        #     (..., M1,...,MD) transformed array.
        N = pxd.NDArrayInfo  # shorthand
        ndi = N.from_obj(x)
        xp = ndi.module()

        axes = tuple(ax - (self.codim_rank - 1) for ax in self._axes)
        if ndi == N.DASK:
            # Entries must have right shape for CZT: pad if required.
            pad_width = [(0, 0)] * x.ndim
            for ax in axes:
                pad_width[ax] = (0, self.dim_shape[ax - 1] - x.shape[ax])

            y = xp.pad(x, pad_width)
            for ax in axes:
                y = self._chunked_transform1D(y, mode, ax)
        else:  # NUMPY/CUPY
            xpf = self.fft_backend(xp)

            func, norm = dict(  # ref: scipy.fft norm conventions
                fw=(xpf.fftn, "backward"),
                bw=(xpf.ifftn, "forward"),
            )[mode]

            # `self._kwargs()` contains parameters undersood by different FFT backends.
            # Need to drop all non-standard parameters.
            sig = inspect.Signature.from_callable(func)
            kwargs = {k: v for (k, v) in self._kwargs[ndi].items() if (k in sig.parameters)}

            N_FFT = tuple(self.dim_shape[ax] for ax in self._axes)
            y = func(
                x=x,
                s=N_FFT,
                axes=axes,
                norm=norm,
                **kwargs,
            )
        return y

    @staticmethod
    def _chunked_transform1D(x: pxt.NDArray, mode: str, axis: int) -> pxt.NDArray:
        # Same signature as _transform(), but:
        # * limited to DASK inputs;
        # * performs 1D transform along chosen axis.

        def _mod_czt(x, M, A, W, n0, k0, axis):
            # 1D Chirp Z-Transform, followed by modulation with W**(k0 * [n0:n0+M]).
            #
            # This is a stripped-down version for performing chunked FFTs: don't use for other purposes.
            #
            # Parameters
            # ----------
            # x : NDArray
            #     (..., N, ...) NUMPY/CUPY array.
            # M : int
            #     Length of the transform.
            # A : complex
            #     Circular offset from the positive real-axis.
            # W : complex
            #     Circular spacing between transform points.
            # k0, n0: int
            #     Modulation coefficients.
            # axis : int
            #     Dimension of `x` along which the samples are stored.
            #
            # Returns
            # -------
            # z: NDArray
            #     (..., M, ...) modulated CZT along the axis indicated by `axis`.
            #     The precision matches that of `x`.
            #     [Note that SciPy's CZT implementation does not guarantee this.]

            # set backend -------------------------------------
            xp = pxu.get_array_module(x)
            xpf = FFT.fft_backend(xp)

            # constants ---------------------------------------
            N = x.shape[axis]
            N_FFT = xpf.next_fast_len(N + M - 1)
            swap = np.arange(x.ndim)
            swap[[axis, -1]] = [-1, axis]

            # filters -----------------------------------------
            k = xp.arange(max(M, N))
            Wk2 = W ** ((k**2) / 2)
            AWk2 = (A ** -k[:N]) * Wk2[:N]
            FWk2 = xpf.fft(
                xp.r_[Wk2[(N - 1) : 0 : -1], Wk2[:M]].conj(),
                n=N_FFT,
            )
            Wk2 = Wk2[:M]

            # transform inputs --------------------------------
            x = x.transpose(*swap).copy()
            x *= AWk2
            y = xpf.fft(x, n=N_FFT)
            y *= FWk2
            z = xpf.ifft(y)[..., (N - 1) : (N + M - 1)]
            z *= Wk2

            # modulate CZT ------------------------------------
            z *= W ** (xp.arange(n0, n0 + M) * k0)
            return z.transpose(*swap)

        def block_ip(x: list[pxt.NDArray], k: pxt.NDArray, sign: int, axis: int) -> pxt.NDArray:
            # Block-defined inner-product.
            #
            # y[:,...,:,k,:,...,:] = \sum_{n} x[:,...,:,n,:,...,:] W^{nk}
            #
            # `x`: list of chunks along transformed dimension.
            # `k`: consecutive sequence of output frequencies along transformed dimension.
            # `sign`: sign of the exponent.
            # `axis`: transformed axis.
            ndi = pxd.NDArrayInfo.from_obj(k)
            xp = ndi.module()

            chunks = tuple(_x.shape[axis] for _x in x)
            N, M = sum(chunks), len(k)
            W = xp.exp(sign * 2j * np.pi / N)
            S = xp.cumsum(xp.r_[0, chunks])

            y = 0
            for idx_n, _x in enumerate(x):
                y += _mod_czt(
                    x=_x,
                    M=M,
                    A=W ** (-k[0]),
                    W=W,
                    n0=k[0],
                    k0=S[idx_n],
                    axis=axis,
                )
            return y

        sign = dict(fw=-1, bw=1)[mode]
        xp = pxd.NDArrayInfo.DASK.module()
        try:  # `x` complex-valued
            cdtype = pxrt.CWidth(x.dtype).value
        except Exception:  # `x` was real-valued
            cdtype = pxrt.Width(x.dtype).complex.value

        ip_ind = tuple(range(x.ndim))
        x_ind = list(range(x.ndim))
        x_ind[axis] = -1  # block_ip() along this axis
        k = xp.arange(x.shape[axis], chunks=x.chunks[axis])
        k_ind = (ip_ind[axis],)

        y = xp.blockwise(
            *(block_ip, ip_ind),
            *(x, x_ind, k, k_ind),
            dtype=cdtype,
            align_arrays=False,
            concatenate=False,
            meta=x._meta,
            # extra kwargs for block_ip()
            sign=sign,
            axis=axis,
        )
        return y
