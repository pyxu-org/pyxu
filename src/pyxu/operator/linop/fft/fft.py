import inspect

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.util.operator import _dask_zip

__all__ = [
    "FFT",
]


class FFT(pxa.LinOp):  # Inherits from LinOp instead of NormalOp since operator not square if `real=True`.
    r"""
    Multi-dimensional Discrete Fourier Transform (DFT) :math:`A: \mathbb{C}^{N_{1} \times \cdots \times N_{D}} \to
    \mathbb{C}^{N_{1} \times \cdots \times N_{D}}`.

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

       (\mathbf{x}, \, \hat{\mathbf{x}}) \in \mathbb{C}^{N_{1} \times \cdots \times N_{D}},
       \quad
       (\mathbf{n}, \, \mathbf{k}) \in \{0, \ldots, N_{1}-1\} \times \cdots \times \{0, \ldots, N_{D}-1\}.

    The DFT is taken over any number of axes by means of the Fast Fourier Transform algorithm (FFT).


    .. rubric:: Implementation Notes

    * The CPU implementation uses `SciPy's FFT implementation <https://docs.scipy.org/doc/scipy/reference/fft.html>`_.
    * The GPU implementation uses cuFFT via `CuPy <https://docs.cupy.dev/en/latest/reference/scipy_fft.html>`_.


    Examples
    --------

    * 1D DFT of a cosine pulse.

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N = 10
         op = FFT(
             arg_shape=(N,),
             real=True,
         )

         x = np.cos(2 * np.pi / N * np.arange(N))
         y = pxu.view_as_complex(op.apply(x))
         # [0, N/2, 0, 0, 0, 0, 0, 0, 0, N/2]

         z = op.adjoint(op.apply(x))
         # np.allclose(z, N * x) -> True

    * 1D DFT of a complex exponential pulse.

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N = 10
         op = FFT(
             arg_shape=(N,),
             real=False,  # complex-valued inputs
         )

         x = np.exp(1j * 2 * np.pi / N * np.arange(N))
         y = pxu.view_as_complex(
                 op.apply(
                     pxu.view_as_real(x)
                 )
             )
         # [0, N, 0, 0, 0, 0, 0, 0, 0, 0]

         z = pxu.view_as_complex(
                 op.adjoint(
                     op.apply(
                         pxu.view_as_real(x)
                     )
                 )
             )
         # np.allclose(z, N * x) -> True

    * 2D DFT of an image

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N_h, N_w = 10, 8
         op = FFT(
             arg_shape=(N_h, N_w),
             real=True,
         )

         x = np.pad(  # an image
             np.ones((N_h//2, N_w//2)),
             pad_width=((0, N_h//2), (0, N_w//2)),
         )
         y = pxu.view_as_complex(  # sinc
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         z = op.adjoint(
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         # np.allclose(z, (N_h * N_w) * x) -> True

    * 1D DFT of an image's rows

      .. code-block:: python3

         from pyxu.operator import FFT
         import pyxu.util as pxu

         N_h, N_w = 10, 8
         op = FFT(
             arg_shape=(N_h, N_w),
             axes=-1,
             real=True,
         )

         x = np.pad(  # an image
             np.ones((N_h//2, N_w//2)),
             pad_width=((0, N_h//2), (0, N_w//2)),
         )
         y = pxu.view_as_complex(  # sinc
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         z = op.adjoint(
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         # np.allclose(z, N_w * x) -> True
    """

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
        real: bool = False,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        arg_shape: NDArrayShape
            (N_1, ..., N_D) shape of the input array :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times \cdots \times
            N_{D}}` or :math:`\mathbb{C}^{N_{1} \times \cdots \times N_{D}}`.
        axes: NDArrayAxis
            Axis or axes along which the DFT is performed.  The default, axes=None, will transform all dimensions of the
            input array.
        real: bool
            If ``True``, assumes :py:func:`~pyxu.operator.FFT.apply` takes (..., N.prod()) inputs in
            :math:`\mathbb{R}^{N}`.

            If ``False``, then :py:func:`~pyxu.operator.FFT.apply` takes (..., 2N.prod()) inputs, i.e.
            :math:`\mathbb{C}^{N}` vectors viewed as bijections with :math:`\mathbb{R}^{2N}`.
        kwargs: dict
            Extra kwargs passed to :py:func:`scipy.fft.fftn` or :py:func:`cupyx.scipy.fft.fftn`.

            Supported parameters for :py:func:`scipy.fft.fftn` are:

                * workers: int = 1

            Supported parameters for :py:func:`cupyx.scipy.fft.fftn` are:

                * NOT SUPPORTED FOR NOW

            Default values are chosen if unspecified.
        """
        arg_shape = pxu.as_canonical_shape(arg_shape)
        N_dim, N = len(arg_shape), np.prod(arg_shape)

        if axes is None:
            axes = tuple(range(N_dim))
        axes = np.unique(pxu.as_canonical_shape(axes))  # drop potential duplicates
        assert np.all((-N_dim <= axes) & (axes < N_dim))  # all axes in valid range
        axes = (axes + N_dim) % N_dim  # get rid of negative axes

        sh_op = [2 * N, 2 * N]
        sh_op[1] //= 2 if real else 1
        super().__init__(shape=sh_op)

        self._arg_shape = tuple(arg_shape)
        self._axes = tuple(axes)
        self._real = bool(real)
        self._kwargs = {
            pxd.NDArrayInfo.NUMPY: dict(
                workers=kwargs.get("workers", 1),
            ),
            pxd.NDArrayInfo.CUPY: dict(),
        }

        self.lipschitz = self.estimate_lipschitz()

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        arg_shape = np.array(self._arg_shape, dtype=int)
        axes = np.array(self._axes, dtype=int)

        L = np.sqrt(arg_shape[axes].prod())
        return L

    def gram(self) -> pxt.OpT:
        from pyxu.operator import HomothetyOp

        op_g = HomothetyOp(dim=self.dim, cst=self.lipschitz**2)
        return op_g

    def cogram(self) -> pxt.OpT:
        if self._real:
            # Enforcing real-valued adjoint() outputs implies a projection between A and A.T.
            # There is no simple closed form to compute the co-gram in this case.
            op_cg = super().cogram()
        else:
            from pyxu.operator import HomothetyOp

            op_cg = HomothetyOp(dim=self.codim, cst=self.lipschitz**2)
        return op_cg

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        N = self.lipschitz**2
        out = self.adjoint(arr)
        out /= N + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        N = self.lipschitz**2
        op = self.T / (N + damp)
        return op

    @pxrt.enforce_precision()
    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxa.UnitOp.svdvals(self, **kwargs) * self.lipschitz
        return D

    def _transform(self, arr: pxt.NDArray, mode: str) -> pxt.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # arr: pxt.NDArray [float/complex]
        #     (N_1, ..., N_D) array to transform.
        # mode: str
        #     Transform direction:
        #
        #     * 'fw': fftn(, norm="backward")
        #     * 'bw': ifftn(, norm="forward")
        #
        # Returns
        # -------
        # out: pxt.NDArray [complex]
        #     (N_1, ..., N_D) transformed array.
        N = pxd.NDArrayInfo  # shorthand
        ndi = N.from_obj(arr)

        if ndi == N.NUMPY:
            fft = pxu.import_module("scipy.fft")
        elif ndi == N.CUPY:
            fft = pxu.import_module("cupyx.scipy.fft")
        else:
            raise ValueError("Unknown NDArray category.")

        func, norm = dict(  # ref: scipy.fft norm conventions
            fw=(fft.fftn, "backward"),
            bw=(fft.ifftn, "forward"),
        )[mode]

        # `self._kwargs()` contains parameters undersood by different FFT backends.
        # If we must fallback to `scipy.fft`, need to drop all non-standard parameters.
        sig = inspect.Signature.from_callable(func)
        kwargs = {k: v for (k, v) in self._kwargs[ndi].items() if (k in sig.parameters)}

        out = func(
            x=arr,
            axes=self._axes,
            norm=norm,
            **kwargs,
        )
        return out

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            * (...,  N.prod()) inputs :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times \cdots \times N_{D}}`
              (``real=True``.)
            * (..., 2N.prod()) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times \cdots \times N_{D}}` viewed as a
              real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., 2N.prod()) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{N_{1} \times \cdots \times N_{D}}` viewed
            as a real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        if self._real:
            r_width = pxrt.Width(arr.dtype)
            c_dtype = r_width.complex.value
        else:
            arr = pxu.view_as_complex(arr)
            c_dtype = arr.dtype

        sh = arr.shape[:-1]
        N_trans = np.prod(sh, dtype=int)
        func = lambda _: self._transform(arr=_, mode="fw")
        blks = _dask_zip(
            func=(func,) * N_trans,
            data=arr.reshape(N_trans, *self._arg_shape),
            out_shape=(self._arg_shape,) * N_trans,
            out_dtype=(c_dtype,) * N_trans,
            parallel=True,
        )

        xp = pxu.get_array_module(arr)
        out = xp.stack(blks, axis=0).reshape(*sh, -1)

        return pxu.view_as_real(out)

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., 2N.prod()) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{N_{1} \times \cdots \times N_{D}}` viewed
            as a real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            * (...,  N.prod()) inputs :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times \cdots \times N_{D}}`
              (``real=True``.)
            * (..., 2N.prod()) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times \cdots \times N_{D}}` viewed as a
              real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        arr = pxu.view_as_complex(arr)
        c_dtype = arr.dtype

        sh = arr.shape[:-1]
        N_trans = np.prod(sh, dtype=int)
        func = lambda _: self._transform(arr=_, mode="bw")
        blks = _dask_zip(
            func=(func,) * N_trans,
            data=arr.reshape(N_trans, *self._arg_shape),
            out_shape=(self._arg_shape,) * N_trans,
            out_dtype=(c_dtype,) * N_trans,
            parallel=True,
        )

        xp = pxu.get_array_module(arr)
        out = xp.stack(blks, axis=0).reshape(*sh, -1)

        if self._real:
            return out.real
        else:
            return pxu.view_as_real(out)
