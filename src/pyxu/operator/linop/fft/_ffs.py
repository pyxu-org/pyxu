# Helper classes/functions related to the FFS transform.
#
# These are low-level routines NOT meant to be imported by default via `import pyxu.operator`.
# Import this module when/where needed only.

import collections.abc as cabc

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.math as pxm
import pyxu.runtime as pxrt
import pyxu.util as pxu

Period = tuple[float]
PeriodCenter = tuple[float]


class _FFS(pxa.NormalOp):
    r"""
    Fast Fourier Series Transform.

    This class allows one to compute FS coefficients of a D-dimensional band-limited signal :math:`\phi: \mathbb{R}^{D}
    \to \mathbb{C}` from its samples.

    A D-dimensional FFS corresponds to taking a 1D FFS along each transform axis.

    .. rubric:: Implementation Notes

    This is a modified version of FFS where both the input/output samples to/from FFS are linearly ordered.
    Reason: _FFS() is designed to be used with NUFFT[123](), where linear ordering makes more sense.
    """

    def __init__(
        self,
        T: Period,
        Tc: PeriodCenter,
        Nfs: pxt.NDArrayShape,
        Ns: pxt.NDArrayShape,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        T: list[float]
            Period :math:`T \in \mathbb{R}_{+}^{D}` of :math:`\phi`.
        Tc: list[float]
            Period mid-point :math:`T_{c} \in \mathbb{R}^{D}` of :math:`\phi`.
        Nfs: list[int]
            Bandwidth :math:`N_{FS} \in \mathbb{O}_{+}^{D}` of :math:`\phi`.
        Ns: list[int]
            FFT length :math:`N_{s} \in \mathbb{N}^{D}`.
            This parameter fixes the dimensionality :math:`D` of the transform.
        kwargs: dict
            kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        """
        from pyxu.operator import FFT

        # Put all internal variables in canonical form ------------------------
        #   T: (D,)-float
        #   Tc: (D,)-float
        #   Nfs: (D,)-int
        #   Ns: (D,)-int
        Ns = pxu.as_canonical_shape(Ns)
        D = len(Ns)
        T = self._as_seq(T, D, float)
        Tc = self._as_seq(Tc, D, float)
        Nfs = self._as_seq(Nfs, D, int)
        for d in range(D):
            assert T[d] > 0
            assert Ns[d] >= 3

            assert Nfs[d] % 2 == 1
            assert 3 <= Nfs[d] <= Ns[d]

        # Object Initialization -----------------------------------------------
        super().__init__(
            dim_shape=(*Ns, 2),
            codim_shape=(*Ns, 2),
        )
        self._Ns = Ns
        self._Nfs = Nfs
        self._T = T
        self._Tc = Tc

        self.lipschitz = self.estimate_lipschitz()
        self._fft = FFT(dim_shape=Ns, axes=None, **kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., Ns1,...,NsD, 2) samples of :math:`\phi \in \mathbb{C}^{N_{s_{1}} \times\cdots\times N_{s_{D}}}` at
            locations specified by ``_FFS.sample_points()``, viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., Ns1,...,NsD, 2) FS coefficients :math:`\{\phi_{\mathbb{k}}^{FS}\}_{k=-\mathbf{N}}^{\mathbf{N}} \in
            \mathbb{C}` in increasing order, viewed as a real array. (See :py:func:`~pyxu.util.view_as_real`.)  Trailing
            entries are 0.

        See Also
        --------
        :py:meth:`~pyxu.operator.linop.fft._ffs._FFS.sample_points`
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., Ns1,...,NsD)
        y = self.capply(x)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., Ns1,...,NsD, 2)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., Ns1,...,NsD) samples of :math:`\phi \in \mathbb{C}^{N_{s_{1}} \times\cdots\times N_{s_{D}}}` at
            locations specified by ``_FFS.sample_points()``.

        Returns
        -------
        out: NDArray
            (..., Ns1,...,NsD) FS coefficients :math:`\{\phi_{\mathbb{k}}^{FS}\}_{k=-\mathbf{N}}^{\mathbf{N}} \in
            \mathbb{C}` in increasing order. Trailing entries are 0.

        See Also
        --------
        :py:meth:`~pyxu.operator.linop.fft._ffs._FFS.sample_points`
        """
        xp = pxu.get_array_module(arr)
        dtype = arr.dtype
        axes = tuple(range(-(self.dim_rank - 1), 0))

        B1, B2, E1, nE2 = self._mod_params(xp, dtype)
        mod1 = [(B1[d] ** (-E1[d])) / self._Ns[d] for d in range(self.dim_rank - 1)]
        mod2 = [B2[d] ** (-nE2[d]) for d in range(self.dim_rank - 1)]

        x = pxm.hadamard_outer(xp.fft.ifftshift(arr, axes), *mod2)
        out = pxm.hadamard_outer(self._fft.capply(x), *mod1)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., Ns1,...,NsD, 2) FS coefficients :math:`\{\phi_{\mathbb{k}}^{FS}\}_{k=-\mathbf{N}}^{\mathbf{N}} \in
            \mathbb{C}` in increasing order, viewed as a real array. (See :py:func:`~pyxu.util.view_as_real`.)  Trailing
            entries are 0.

        Returns
        -------
        out: NDArray
            (..., Ns1,...,NsD, 2) samples of :math:`\phi \in \mathbb{C}^{N_{s_{1}} \times\cdots\times N_{s_{D}}}` at
            locations specified by ``_FFS.sample_points()``, viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        See Also
        --------
        :py:meth:`~pyxu.operator.linop.fft._ffs._FFS.sample_points`
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., Ns1,...,NsD)
        y = self.cadjoint(x)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., Ns1,...,NsD, 2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., Ns1,...,NsD) FS coefficients :math:`\{\phi_{\mathbb{k}}^{FS}\}_{k=-\mathbf{N}}^{\mathbf{N}} \in
            \mathbb{C}` in increasing order. Trailing entries are 0.

        Returns
        -------
        out: NDArray
            (..., Ns1,...,NsD) samples of :math:`\phi \in \mathbb{C}^{N_{s_{1}} \times\cdots\times N_{s_{D}}}` at
            locations specified by ``_FFS.sample_points()``.

        See Also
        --------
        :py:meth:`~pyxu.operator.linop.fft._ffs._FFS.sample_points`
        """
        xp = pxu.get_array_module(arr)
        dtype = arr.dtype
        axes = tuple(range(-(self.dim_rank - 1), 0))

        B1, B2, E1, nE2 = self._mod_params(xp, dtype)
        mod1 = [(B1[d] ** E1[d]) / self._Ns[d] for d in range(self.dim_rank - 1)]
        mod2 = [B2[d] ** nE2[d] for d in range(self.dim_rank - 1)]

        x = pxm.hadamard_outer(arr, *mod1)
        y = pxm.hadamard_outer(self._fft.cadjoint(x), *mod2)

        out = xp.fft.fftshift(y, axes)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        L = 1 / np.sqrt(np.prod(self._Ns))
        return L

    def gram(self) -> pxt.OpT:
        from pyxu.operator import HomothetyOp

        G = HomothetyOp(
            dim_shape=self.dim_shape,
            cst=self.lipschitz**2,
        )
        return G

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        out /= (self.lipschitz**2) + damp
        return out

    def cpinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        r"""
        iFFS transform.

        Parameters
        ----------
        arr: NDArray
            (..., Ns1,...,NsD) FS coefficients :math:`\{\phi_{\mathbb{k}}^{FS}\}_{k=-\mathbf{N}}^{\mathbf{N}} \in
            \mathbb{C}` in increasing order. Trailing entries are 0.

        Returns
        -------
        out: NDArray
            (..., Ns1,...,NsD) samples of :math:`\phi \in \mathbb{C}^{N_{s_{1}} \times\cdots\times N_{s_{D}}}` at
            locations specified by ``_FFS.sample_points()``.

        See Also
        --------
        :py:meth:`~pyxu.operator.linop.fft._ffs._FFS.sample_points`
        """
        out = self.cadjoint(arr)
        out /= (self.lipschitz**2) + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / ((self.lipschitz**2) + damp)
        return op

    @pxrt.enforce_precision()
    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxa.UnitOp.svdvals(self, **kwargs) * self.lipschitz
        return D

    def sample_points(
        self,
        xp: pxt.ArrayModule,
        dtype: pxt.DType,
        flatten: bool = True,
    ) -> list[pxt.NDArray]:
        """
        Sampling positions for FFS forward transform. (spatial samples -> FS coefficients.)

        Parameters
        ----------
        xp: ArrayModule
        dtype: DType
        flatten: bool

        Returns
        -------
        S1,...,SD : list[NDArray]
            (Ns_D,) mesh-points at which to sample a signal in the d-th dimension (in the right order).
            If `flatten` is False, then this is a sparse mesh.
        """
        S = [None] * (self.dim_rank - 1)
        for d, (ns, t, tc) in enumerate(zip(self._Ns, self._T, self._Tc)):
            if ns % 2 == 1:  # odd case
                m = (ns - 1) // 2
                idx = xp.array([*range(0, m + 1), *range(-m, 0)])
                _S = (tc + (t / ns) * idx).astype(dtype)
            else:  # even case
                m = ns // 2
                idx = xp.array([*range(0, m), *range(-m, 0)])
                _S = (tc + (t / ns) * (idx + 0.5)).astype(dtype)
            S[d] = xp.fft.fftshift(_S)

        if not flatten:
            S = xp.meshgrid(*S, indexing="ij", sparse=True)
        return S

    # Helper routines (internal) ----------------------------------------------
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

    def _mod_params(self, xp: pxt.ArrayModule, dtype: pxt.DType):
        """
        Returns
        -------
        B1, B2: list[float]
            (D,) base terms.
        E1, nE2: list[NDArray]
            (D,) exponent vectors (1D).
        """
        B1 = [None] * (self.dim_rank - 1)
        B2 = [None] * (self.dim_rank - 1)
        E1 = [None] * (self.dim_rank - 1)
        nE2 = [None] * (self.dim_rank - 1)

        for d in range(self.dim_rank - 1):
            t, tc, nfs, ns = [_[d] for _ in (self._T, self._Tc, self._Nfs, self._Ns)]
            q, n = ns - nfs, (nfs - 1) // 2
            if ns % 2 == 1:  # odd case
                m = (ns - 1) // 2
                B1[d] = np.exp((2j * np.pi / t) * tc)
                nE2[d] = n * xp.array([*range(0, m + 1), *range(-m, 0)], dtype=dtype)
            else:  # even case
                m = ns // 2
                B1[d] = np.exp((2j * np.pi / t) * (tc + 0.5 * t / ns))
                nE2[d] = n * xp.array([*range(0, m), *range(-m, 0)], dtype=dtype)
            E1[d] = xp.array([*range(-n, n + 1), *((0,) * q)], dtype=dtype)
            B2[d] = np.exp(-2j * np.pi / ns)

        return B1, B2, E1, nE2
