import jax
import jax.numpy as jnp

from ..abc import LinearOperator, register_linop_vjp
from ..math import hadamard_outer
from ..typing import Array, DType
from ..util import TranslateDType, cdtype


@register_linop_vjp
class CZT(LinearOperator):
    r"""
    Multi-dimensional Chirp Z-Transform (CZT) :math:`C: \bC^{I_{1} \times\cdots\times I_{D}} \to
    \bC^{O_{1} \times\cdots\times O_{D}}`.

    The 1D CZT of parameters :math:`(A, W, O)` is defined as:

    .. math::

       (C \, \bbx)[k]
       =
       \bby[k]
       =
       \sum_{n=0}^{I-1} \bbx[n] A^{-n} W^{nk},

    where :math:`\bbx \in \bC^{I}`, :math:`A, W \in \bC`, and :math:`k = \{0, \ldots, O-1\}`.

    A D-dimensional CZT corresponds to taking a 1D CZT along each transform axis.
    """

    A: Array
    W: Array

    def __init__(
        self,
        dim_shape: tuple[int],
        codim_shape: tuple[int],
        A: tuple[complex],
        W: tuple[complex],
        dtype: DType = None,
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
        dtype: dtype
            FP precision of input/outputs.
            If not provided, uses JAX's default complex FP type.
        """
        if dtype is None:
            c_dtype = cdtype()
        else:
            c_dtype = TranslateDType(dtype).to_complex()

        self.dim_info = jax.ShapeDtypeStruct(dim_shape, c_dtype)
        self.codim_info = jax.ShapeDtypeStruct(codim_shape, c_dtype)

        D = len(dim_shape)
        assert all([len(_) == D for _ in (dim_shape, codim_shape, A, W)])

        self.A = jnp.asarray(A, dtype=c_dtype)
        self.W = jnp.asarray(W, dtype=c_dtype)

    def apply(self, x: Array) -> Array:
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
        L = jax.tree.map(
            lambda lhs, rhs: lhs + rhs - 1,
            # todo: increase FFT length (L >= I+O-1) to further optimize runtime?
            self.dim_shape,
            self.codim_shape,
        )
        AWk2, FWk2, Wk2, extract = self._mod_params_apply(L, x.dtype)

        _x = hadamard_outer(x, *AWk2)
        _x = jnp.fft.fftn(_x, s=L)
        _x = hadamard_outer(_x, *FWk2)
        _x = jnp.fft.ifftn(_x)
        y = hadamard_outer(_x[*extract], *Wk2)
        return y

    def adjoint(self, y: Array) -> Array:
        r"""
        Parameters
        ----------
        y: Array[float/complex]
            (..., O1,...,OD) outputs :math:`\bby = (C \, \bbx) \in \bC^{O_{1} \times\cdots\times O_{D}}`.

        Returns
        -------
        x: Array[complex]
            (..., I1,...,ID) inputs :math:`\bbx \in \bC^{I_{1} \times\cdots\times I_{D}}`.
        """
        # CZT^{*}(y,O,A,W)[n] = CZT(y,I,A=1,W=W*)[n] * A^{n}
        czt = CZT(
            dim_shape=self.codim_shape,
            codim_shape=self.dim_shape,
            A=(1,) * self.dim_rank,
            W=self.W.conj(),
            dtype=y.dtype,
        )
        An = self._mod_params_adjoint(y.dtype)

        _x = czt.apply(y)
        x = hadamard_outer(_x, *An)
        return x

    # Helper routines (internal) ----------------------------------------------
    def _mod_params_apply(self, _L: tuple[int], dtype: DType):
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
        c_dtype = TranslateDType(dtype).to_complex()

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

    def _mod_params_adjoint(self, dtype: DType):
        """
        Parameters
        ----------
        dtype: float/complex

        Returns
        -------
        An: list[Array]
            (I1,),...,(ID,) vectors.
        """
        c_dtype = TranslateDType(dtype).to_complex()

        D = self.dim_rank
        An = [None] * D
        for d in range(D):
            A = self.A[d]
            N = self.dim_shape[d]
            _An = A ** jnp.arange(N)

            An[d] = _An.astype(c_dtype)

        return An
