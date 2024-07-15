import collections.abc as cabc
import functools

import numpy as np
import scipy.special as sps

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "CZT",
]


class CZT(pxa.LinOp):
    r"""
    Multi-dimensional Chirp Z-Transform (CZT) :math:`C: \mathbb{C}^{N_{1} \times\cdots\times N_{D}} \to
    \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.

    The 1D CZT of parameters :math:`(A, W, M)` is defined as:

    .. math::

       (C \, \mathbf{x})[k]
       =
       \sum_{n=0}^{N-1} \mathbf{x}[n] A^{-n} W^{nk},

    where :math:`\mathbf{x} \in \mathbb{C}^{N}`, :math:`A, W \in \mathbb{C}`, and :math:`k = \{0, \ldots, M-1\}`.

    A D-dimensional CZT corresponds to taking a 1D CZT along each transform axis.

    .. rubric:: Implementation Notes

    For stability reasons, this implementation assumes :math:`A, W \in \mathbb{C}` lie on the unit circle.

    See Also
    --------
    :py:class:`~pyxu.operator.FFT`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis,
        M,
        A,
        W,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (N1,...,ND) dimensions of the input :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times\cdots\times N_{D}}`.
        axes: NDArrayAxis
            Axes over which to compute the CZT. If not given, all axes are used.
        M : int, list(int)
            Length of the transform per axis.
        A : complex, list(complex)
            Circular offset from the positive real-axis per axis.
        W : complex, list(complex)
            Circular spacing between transform points per axis.
        kwargs: dict
            Extra kwargs passed to :py:class:`~pyxu.operator.FFT`.
        """
        dim_shape = pxu.as_canonical_shape(dim_shape)
        if axes is None:
            axes = tuple(range(len(dim_shape)))
        self._axes = pxu.as_canonical_axes(axes, len(dim_shape))
        _M, self._A, self._W = self._canonical_repr(self._axes, M, A, W)

        codim_shape = list(dim_shape)
        for i, ax in enumerate(self._axes):
            codim_shape[ax] = _M[i]
        super().__init__(
            dim_shape=(*dim_shape, 2),
            codim_shape=(*codim_shape, 2),
        )
        self._kwargs = kwargs

        # We know a crude Lipschitz bound by default. Since computing it takes (code) space,
        # the estimate is computed as a special case of estimate_lipschitz()
        self.lipschitz = self.estimate_lipschitz(__rule=True)

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N1,...,ND,2) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times\cdots\times N_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M1,...,MD,2) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}` viewed
            as a real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))
        y = self.capply(x)
        out = pxu.view_as_real(pxu.require_viewable(y))
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N1,...,ND) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{M_{1} \times\cdots\times M_{D}}`.
        """
        AWk2, FWk2, Wk2, extract, fft = self._get_meta(arr)
        arr = arr.copy()  # for in-place updates
        for _AWk2 in AWk2:
            arr *= _AWk2
        y = fft.capply(arr)
        for _FWk2 in FWk2:
            y *= _FWk2
        out = fft.cpinv(y, damp=0)[extract]
        for _Wk2 in Wk2:
            out *= _Wk2
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
            (..., N1,...,ND,2) outputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times\cdots\times N_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))
        y = self.cadjoint(x)
        out = pxu.view_as_real(pxu.require_viewable(y))
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
            (..., N1,...,ND) outputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times\cdots\times N_{D}}`.
        """
        # CZT^{*}(y,M,A,W)[n] = CZT(y,N,A=1,W=W*)[n] * A^{n}
        czt = CZT(
            dim_shape=self.codim_shape[:-1],
            axes=self._axes,
            M=[self.dim_shape[ax] for ax in self._axes],
            A=1,
            W=np.conj(self._W),
            **self._kwargs,
        )
        out = czt.capply(arr)

        # Re-scale outputs per axis.
        xp = pxu.get_array_module(out)
        cdtype = pxrt.CWidth(out.dtype).value
        for i, ax in enumerate(self._axes):
            A = self._A[i]
            N = self.dim_shape[ax]
            expand = (np.newaxis,) * (self.dim_rank - 2 - ax)

            An = A ** xp.arange(N)[..., *expand]
            out *= An.astype(cdtype)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        no_eval = "__rule" in kwargs
        if no_eval:
            # We know that
            #     L^{2} = \sigma_{max}^{2}(C) = \lambda_{max}(C.gram) = \lambda_{max}(C.cogram)
            # We know that C.[co]gram correspond to linear convolution of the input with a Dirichlet kernel, i.e.
            #     ( Gx)[n] = \sum_{q=0}^{N-1} x[q] h1[n-q],  h1[n] = A^{n} W^{-(M-1)/2 n} \sin[p M/2 n] / \sin[p 1/2 n]
            #     (CGy)[n] = \sum_{q=0}^{M-1} y[q] h2[n-q],  h2[n] =       W^{ (N-1)/2 n} \sin[p N/2 n] / \sin[p 1/2 n]
            #                                                    p = \arg{W}
            # From Young's convolution inequality, we have the upper bound
            #     \norm{x \ast h}{2} \le \norm{x}{2}\norm{h}{1}
            # Therefore
            #     L^{2} <= max(\norm{h1}{1}, \norm{h2}{1})
            L2 = 1
            for i, ax in enumerate(self._axes):
                N = self.dim_shape[ax]
                M = self.codim_shape[ax]
                p = np.angle(self._W[i])

                h1 = sps.diric(p * np.arange(-N, N + 1), n=M) * M
                norm1 = np.fabs(h1).sum()

                h2 = sps.diric(p * np.arange(-M, M + 1), n=N) * N
                norm2 = np.fabs(h2).sum()

                L2 *= max(norm1, norm2)
            L = np.sqrt(L2)
        else:
            L = super().estimate_lipschitz(**kwargs)
        return L

    def asarray(self, **kwargs) -> pxt.NDArray:
        # We compute 1D transforms per axis, then Kronecker product them.

        # Since non-NP backends may be faulty, we do everything in NUMPY ...
        A_1D = [None] * (D := self.dim_rank - 1)
        i = 0
        for ax in range(D):
            N = self.dim_shape[ax]
            M = self.codim_shape[ax]
            if ax in self._axes:
                n = np.arange(N)
                m = np.arange(M)
                _A, _W = self._A[i], self._W[i]
                A_1D[ax] = (_W ** np.outer(m, n)) * (_A ** (-n))
                i += 1
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

    # Helper routines (internal) ----------------------------------------------
    @staticmethod
    def _canonical_repr(axes, M, A, W):
        # Create canonical representations
        #   * `_M`: tuple(int)
        #   * `_A`: tuple(complex)
        #   * `_W`: tuple(complex)
        #
        # `axes` is already assumed in tuple-form.
        def as_seq(x, N, _type):
            if isinstance(x, cabc.Iterable):
                _x = tuple(x)
            else:
                _x = (x,)
            if len(_x) == 1:
                _x *= N  # broadcast
            assert len(_x) == N

            return tuple(map(_type, _x))

        _M = as_seq(M, len(axes), int)
        _A = as_seq(A, len(axes), complex)
        _W = as_seq(W, len(axes), complex)
        assert all(m > 0 for m in _M)
        assert np.allclose(np.abs(_A), 1)
        assert np.allclose(np.abs(_W), 1)
        return _M, _A, _W

    def _get_meta(self, x: pxt.NDArray):
        # x: (..., M1,...,MD) [complex]
        #
        # Computes/Initializes (per axis):
        # * `AWk2`: list[NDArray] pre-FFT modulation vectors.
        # * `FWk2`: list[NDArray] FFT of convolution filters.
        # * `Wk2`: list[NDArray] post-FFT modulation vectors.
        # * `extract`: tuple[slice] FFT interval to extract.
        # * `fft`: FFT object to transform the input.
        from pyxu.operator import FFT

        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi == pxd.NDArrayInfo.DASK:
            xp = pxu.get_array_module(x._meta)
        else:
            xp = ndi.module()
        xpf = FFT.fft_backend(xp)
        cdtype = pxrt.CWidth(x.dtype).value

        # Initialize FFT to transform inputs.
        fft_shape = list(self.dim_shape[:-1])
        for i, ax in enumerate(self._axes):
            fft_shape[ax] += self.codim_shape[ax] - 1
        fft_shape = FFT.next_fast_len(fft_shape)
        fft = FFT(
            dim_shape=fft_shape,
            axes=self._axes,
            **self._kwargs,
        )

        # Build modulation vectors (Wk2, AWk2, FWk2).
        Wk2, AWk2, FWk2 = [], [], []
        for i, ax in enumerate(self._axes):
            A = self._A[i]
            W = self._W[i]
            N = self.dim_shape[ax]
            M = self.codim_shape[ax]
            L = fft.dim_shape[ax]

            k = xp.arange(max(M, N))
            _Wk2 = W ** ((k**2) / 2)
            _AWk2 = (A ** -k[:N]) * _Wk2[:N]
            _FWk2 = xpf.fft(
                xp.concatenate([_Wk2[(N - 1) : 0 : -1], _Wk2[:M]]).conj(),
                n=L,
            )
            _Wk2 = _Wk2[:M]

            expand = (np.newaxis,) * (self.dim_rank - 2 - ax)
            Wk2.append(_Wk2.astype(cdtype)[..., *expand])
            AWk2.append(_AWk2.astype(cdtype)[..., *expand])
            FWk2.append(_FWk2.astype(cdtype)[..., *expand])

        # Build (extract,)
        N_stack = x.ndim - (self.dim_rank - 1)
        extract = [slice(None)] * x.ndim
        for ax in self._axes:
            N = self.dim_shape[ax]
            M = self.codim_shape[ax]
            L = fft.dim_shape[ax]

            extract[N_stack + ax] = slice(N - 1, N + M - 1)
        extract = tuple(extract)

        return AWk2, FWk2, Wk2, extract, fft
