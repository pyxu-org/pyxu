import equinox as eqx
import jax.numpy as jnp
import numpy as np

import pyxu.abc as pxa
import pyxu.math as pxm
import pyxu.typing as pxt
import pyxu.util as pxu


class CZT(pxa.LinearOperator):
    r"""
    Multi-dimensional Chirp Z-Transform (CZT) :math:`C: \bC^{I_{1} \times\cdots\times I_{D}} \to
    \bC^{O_{1} \times\cdots\times O_{D}}`.

    The 1D CZT of parameters :math:`(I, A, W, O)` is defined as:

    .. math::

       (C \, \bbx)[k]
       =
       \bby[k]
       =
       \sum_{n=0}^{I-1} \bbx[n] A^{-n} W^{nk},

    where :math:`\bbx \in \bC^{I}`, :math:`A, W \in \bC`, and :math:`k = \{0, \ldots, O-1\}`.

    A D-dimensional CZT corresponds to taking a 1D CZT along each transform axis.
    """

    A: tuple[complex]
    W: tuple[complex]

    def __init__(
        self,
        dim_shape: tuple[int],
        codim_shape: tuple[int],
        A: tuple[complex],
        W: tuple[complex],
    ):
        r"""
        Parameters
        ----------
        dim_shape: tuple[int]
            (I1,...,ID) dimensions of the input :math:`\bbx`.
        codim_shape: tuple[int]
            (O1,...,OD) dimensions of the output :math:`\bby`.
        A: tuple[complex]
            (D,) circular offsets.
        W: tuple[complex]
            (D,) circular spacings between transform points.
        """
        self.dim_shape = pxu.ShapeStruct(dim_shape)
        self.codim_shape = pxu.ShapeStruct(codim_shape)
        assert self.dim_shape.ndim == self.codim_shape.ndim

        self.A = tuple(map(complex, pxu.broadcast_seq(A, self.dim_rank)))
        self.W = tuple(map(complex, pxu.broadcast_seq(W, self.dim_rank)))

    def apply(self, x: pxt.Array) -> pxt.Array:
        r"""
        Parameters
        ----------
        x: Array[float/complex]
            (I1,...,ID) inputs :math:`\bbx \in \bC^{I_{1} \times\cdots\times I_{D}}`.

        Returns
        -------
        y: Array[complex]
            (O1,...,OD) outputs :math:`\bby = (C \, \bbx) \in \bC^{O_{1} \times\cdots\times O_{D}}`.
        """
        L = [
            pxu.next_fast_len(lhs + rhs - 1)
            for (lhs, rhs) in zip(self.dim_shape, self.codim_shape)
        ]
        AWk2, FWk2, Wk2, extract = self._mod_params(L, x.dtype)

        _x = pxm.hadamard_outer(x, *AWk2)
        _x = jnp.fft.fftn(_x, s=L)
        _x = pxm.hadamard_outer(_x, *FWk2)
        _x = jnp.fft.ifftn(_x)
        y = pxm.hadamard_outer(_x[*extract], *Wk2)
        return y

    # Helper routines (internal) ----------------------------------------------
    def _mod_params(self, _L: tuple[int], dtype: pxt.DType):
        """
        Parameters
        ----------
        _L: tuple[int]
            (L1,...,LD) FFT length.
        dtype: float/complex

        Returns
        -------
        AWk2: list[Array]
            (I1,),...,(ID,) pre-FFT modulation vectors.
        FWk2: list[Array]
            (L1,),...,(LD,) FFT of convolution filters.
        Wk2: list[Array]
            (O1,),...,(OD,) post-FFT modulation vectors.
        extract: list[slice]
            (slice1,...,sliceD) FFT interval to extract.
        """
        c_dtype = pxu.TranslateDType(dtype).to_complex()

        # Build modulation vectors (Wk2, AWk2, FWk2).
        D = self.dim_rank
        Wk2 = [None] * D
        AWk2 = [None] * D
        FWk2 = [None] * D
        for d in range(D):
            A = self.A[d]
            W = self.W[d]
            N = self.dim_shape[d]
            M = self.codim_shape[d]
            L = _L[d]

            k = jnp.arange(max(M, N))
            _Wk2 = W ** ((k**2) / 2)
            _AWk2 = (A ** -k[:N]) * _Wk2[:N]
            _FWk2 = jnp.fft.fft(
                jnp.concatenate([_Wk2[(N - 1) : 0 : -1], _Wk2[:M]]).conj(),
                n=L,
            )
            _Wk2 = _Wk2[:M]

            Wk2[d] = _Wk2.astype(c_dtype)
            AWk2[d] = _AWk2.astype(c_dtype)
            FWk2[d] = _FWk2.astype(c_dtype)

        # Build (extract,)
        extract = [slice(None)] * D
        for d in range(D):
            N = self.dim_shape[d]
            M = self.codim_shape[d]
            L = _L[d]
            extract[d] = slice(N - 1, N + M - 1)

        return AWk2, FWk2, Wk2, extract


class Uniform2Uniform(pxa.LinearOperator):
    r"""
    Multi-dimensional Uniform-to-Uniform Fourier Transform.

    Given the Dirac stream

    .. math::

       f(\bbx) = \sum_{i} w_{i} \delta(\bbx - \bbx_{i}),

    computes samples of :math:`f^{F}`, i.e.,

    .. math::

       f^{F}(\bbv_{o}) = \bbz_{o} = \sum_{i} w_{i} \ee^{ -\cj 2\pi \innerProduct{\bbx_{i}}{\bbv_{o}} },

    where :math:`(\bbx_{i}, \bbv_{o})` lie on uniform lattices.
    """

    x_spec: pxu.UniformSpec
    v_spec: pxu.UniformSpec
    isign: int

    def __init__(
        self,
        x_spec: pxu.UniformSpec,
        v_spec: pxu.UniformSpec,
        isign: int,
    ):
        r"""
        Parameters
        ----------
        x_spec: UniformSpec
            :math:`\bbx_{i}` lattice specifier.
        v_spec: UniformSpec
            :math:`\bbv_{o}` lattice specifier.
        isign: +1, -1
            Sign of the exponent.
        """
        assert x_spec.ndim == v_spec.ndim

        self.dim_shape = pxu.ShapeStruct(x_spec.num)
        self.codim_shape = pxu.ShapeStruct(v_spec.num)

        self.x_spec = x_spec
        self.v_spec = v_spec
        self.isign = int(isign / abs(isign))

    def apply(self, w: pxt.Array) -> pxt.Array:
        r"""
        Parameters
        ----------
        w: Array[float/complex]
            (I1,...,ID) weights :math:`w_{i} \in \bC^{D}`.

        Returns
        -------
        z: Array[complex]
            (O1,...,OD) weights :math:`z_{o} \in \bC^{D}`.
        """
        czt, B = self._params(w.dtype)

        _w = czt.apply(w)
        z = pxm.hadamard_outer(_w, *B)
        return z

    # Helper routines (internal) ----------------------------------------------
    def _params(self, dtype: pxt.DType):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        czt: CZT
            CZT(I,A,W,O) instance.
        B: list[Array]
            (O1,),...,(OD,) post-CZT modulation vectors.
        """
        c_dtype = pxu.TranslateDType(dtype).to_complex()

        x0, dx = map(np.array, [self.x_spec.start, self.x_spec.step])
        v0, dv = map(np.array, [self.v_spec.start, self.v_spec.step])
        D = self.dim_rank

        # Build CZT operator
        A = np.exp(-1j * self.isign * 2 * np.pi * dx * v0)
        W = np.exp(+1j * self.isign * 2 * np.pi * dx * dv)
        czt = CZT(
            dim_shape=self.dim_shape,
            codim_shape=self.codim_shape,
            A=A,
            W=W,
        )

        # Build modulation vector (B,).
        B = [None] * D
        for d in range(D):
            phase_scale = self.isign * 2 * np.pi * x0[d]
            v = v0[d] + dv[d] * jnp.arange(self.v_spec.num[d])
            _B = jnp.exp(1j * phase_scale * v)

            B[d] = _B.astype(c_dtype)

        return czt, B


U2U = Uniform2Uniform  # alias
