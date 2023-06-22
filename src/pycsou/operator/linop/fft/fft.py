import importlib
import inspect

import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
from pycsou.util.operator import _dask_zip

__all__ = [
    "FFT",
]


class FFT(pyca.LinOp):  # Inherits from LinOp instead of NormalOp since operator not square if `real=True`.
    r"""
    Multi-dimensional Discrete Fourier Transform (DFT) :math:`A: \mathbb{C}^{N_{1} \times \cdots
    \times N_{D}} \to \mathbb{C}^{N_{1} \times \cdots \times N_{D}}`:

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


    **Implementation Notes**

    * The CPU implementation uses `PyFFTW <https://pyfftw.readthedocs.io/en/latest/>`_ if installed;
      and otherwise relies on `SciPy's FFT implementation
      <https://docs.scipy.org/doc/scipy/reference/fft.html>`_.
    * The GPU implementation uses cuFFT via `CuPy
      <https://docs.cupy.dev/en/latest/reference/scipy_fft.html>`_.


    **Examples**

    * 1D DFT of a cosine pulse.

      .. code-block:: python3

         from pycsou.operator.linop import FFT
         import pycsou.util as pycu

         N = 10
         op = FFT(
             arg_shape=(N,),
             real=True,
         )

         x = np.cos(2 * np.pi / N * np.arange(N))
         y = pycu.view_as_complex(op.apply(x))
         # [0, N/2, 0, 0, 0, 0, 0, 0, 0, N/2]

         z = op.adjoint(op.apply(x))
         # np.allclose(z, N * x) -> True

    * 1D DFT of a complex exponential pulse.

      .. code-block:: python3

         from pycsou.operator.linop import FFT
         import pycsou.util as pycu

         N = 10
         op = FFT(
             arg_shape=(N,),
             real=False,  # complex-valued inputs
         )

         x = np.exp(1j * 2 * np.pi / N * np.arange(N))
         y = pycu.view_as_complex(
                 op.apply(
                     pycu.view_as_real(x)
                 )
             )
         # [0, N, 0, 0, 0, 0, 0, 0, 0, 0]

         z = pycu.view_as_complex(
                 op.adjoint(
                     op.apply(
                         pycu.view_as_real(x)
                     )
                 )
             )
         # np.allclose(z, N * x) -> True

    * 2D DFT of an image

      .. code-block:: python3

         from pycsou.operator.linop import FFT
         import pycsou.util as pycu

         N_h, N_w = 10, 8
         op = FFT(
             arg_shape=(N_h, N_w),
             real=True,
         )

         x = np.pad(  # an image
             np.ones((N_h//2, N_w//2)),
             pad_width=((0, N_h//2), (0, N_w//2)),
         )
         y = pycu.view_as_complex(  # sinc
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         z = op.adjoint(
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         # np.allclose(z, (N_h * N_w) * x) -> True

    * 1D DFT of an image's rows

      .. code-block:: python3

         from pycsou.operator.linop import FFT
         import pycsou.util as pycu

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
         y = pycu.view_as_complex(  # sinc
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         z = op.adjoint(
                 op.apply(x.reshape(-1))
             ).reshape(N_h, N_w)
         # np.allclose(z, N_w * x) -> True
    """

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        axes: pyct.NDArrayAxis = None,
        real: bool = False,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        arg_shape: pyct.NDArrayShape
            (N_1, ..., N_D) shape of the input array :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times
            \cdots \times N_{D}}` or :math:`\mathbb{C}^{N_{1} \times \cdots \times N_{D}}`.
        axes: pyct.NDArrayAxis
            Axis or axes along which the DFT is performed.
            The default, axes=None, will transform all dimensions of the input array.
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., N.prod()) inputs in
            :math:`\mathbb{R}^{N}`.

            If ``False``, then ``.apply()`` takes (..., 2N.prod()) inputs, i.e.
            :math:`\mathbb{C}^{N}` vectors viewed as bijections with :math:`\mathbb{R}^{2N}`.
        kwargs: dict
            Extra kwargs passed to
            :py:func:`pyfftw.interfaces.scipy_fft.fftn` or
            :py:func:`cupyx.scipy.fft.fftn`.

            Supported parameters for :py:func:`pyfftw.interfaces.scipy_fft.fftn` are:

                * workers: int = 1
                * auto_align_input: bool = True
                * auto_contiguous: bool = True

            Supported parameters for :py:func:`cupyx.scipy.fft.fftn` are:

                * NOT SUPPORTED FOR NOW

            Default values are chosen if unspecified.
        """
        arg_shape = pycu.as_canonical_shape(arg_shape)
        N_dim, N = len(arg_shape), np.prod(arg_shape)

        if axes is None:
            axes = tuple(range(N_dim))
        axes = np.unique(pycu.as_canonical_shape(axes))  # drop potential duplicates
        assert np.all((-N_dim <= axes) & (axes < N_dim))  # all axes in valid range
        axes = (axes + N_dim) % N_dim  # get rid of negative axes

        sh_op = [2 * N, 2 * N]
        sh_op[1] //= 2 if real else 1
        super().__init__(shape=sh_op)

        self._arg_shape = tuple(arg_shape)
        self._axes = tuple(axes)
        self._real = bool(real)
        self._kwargs = {
            pycd.NDArrayInfo.NUMPY: dict(
                workers=kwargs.get("workers", 1),
                planner_effort="FFTW_ESTIMATE",
                auto_align_input=kwargs.get("auto_align_input", True),
                auto_contiguous=kwargs.get("auto_contiguous", True),
            ),
            pycd.NDArrayInfo.CUPY: dict(),
        }

    @pycrt.enforce_precision()
    def lipschitz(self, **kwargs) -> pyct.Real:
        arg_shape = np.array(self._arg_shape, dtype=int)
        axes = np.array(self._axes, dtype=int)

        self._lipschitz = np.sqrt(arg_shape[axes].prod())
        return self._lipschitz

    def gram(self) -> pyct.OpT:
        from pycsou.operator.linop import HomothetyOp

        op_g = HomothetyOp(dim=self.dim, cst=self.lipschitz() ** 2)
        return op_g

    def cogram(self) -> pyct.OpT:
        if self._real:
            # Enforcing real-valued adjoint() outputs implies a projection between A and A.T.
            # There is no simple closed form to compute the co-gram in this case.
            op_cg = super().cogram()
        else:
            from pycsou.operator.linop import HomothetyOp

            op_cg = HomothetyOp(dim=self.codim, cst=self.lipschitz() ** 2)
        return op_cg

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        N = self.lipschitz() ** 2
        damp = kwargs.get("damp", 0)

        out = self.adjoint(arr)
        out /= N + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        N = self.lipschitz() ** 2
        damp = kwargs.get("damp", 0)

        op = self.T / (N + damp)
        return op

    @pycrt.enforce_precision()
    def svdvals(self, **kwargs) -> pyct.NDArray:
        D = pyca.UnitOp.svdvals(self, **kwargs) * self.lipschitz()
        return D

    def _transform(self, arr: pyct.NDArray, mode: str) -> pyct.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # arr: pyct.NDArray [float/complex]
        #     (N_1, ..., N_D) array to transform.
        # mode: str
        #     Transform direction:
        #
        #     * 'fw': fftn(, norm="backward")
        #     * 'bw': ifftn(, norm="forward")
        #
        # Returns
        # -------
        # out: pyct.NDArray [complex]
        #     (N_1, ..., N_D) transformed array.
        N = pycd.NDArrayInfo  # shorthand
        ndi = N.from_obj(arr)

        if ndi == N.NUMPY:
            PYFFTW_INSTALLED = importlib.util.find_spec("pyfftw") is not None
            if PYFFTW_INSTALLED:
                fft = pycu.import_module("pyfftw.interfaces.scipy_fft")
            else:  # fallback to SciPy's routines
                fft = pycu.import_module("scipy.fft")
        elif ndi == N.CUPY:
            fft = pycu.import_module("cupyx.scipy.fft")
        else:
            raise ValueError("Unknown NDArray category.")

        func, norm = dict(  # ref: scipy.fft norm conventions
            fw=(fft.fftn, "backward"),
            bw=(fft.ifftn, "forward"),
        )[mode]

        # `self._kwargs()` contains parameters undersood by pyFFTW.
        # If we must fallback to `scipy.fft`, need to drop all pyFFTW extra parameters.
        sig = inspect.Signature.from_callable(func)
        kwargs = {k: v for (k, v) in self._kwargs[ndi].items() if (k in sig.parameters)}

        out = func(
            x=arr,
            axes=self._axes,
            norm=norm,
            **kwargs,
        )
        return out

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray
            * (...,  N.prod()) inputs :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times \cdots \times
              N_{D}}` (``real=True``.)
            * (..., 2N.prod()) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times \cdots \times
              N_{D}}` viewed as a real array. (See :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray
            (..., 2N.prod()) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{N_{1} \times \cdots
            \times N_{D}}` viewed as a real array. (See :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        if self._real:
            r_width = pycrt.Width(arr.dtype)
            c_dtype = r_width.complex.value
        else:
            arr = pycu.view_as_complex(arr)
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

        xp = pycu.get_array_module(arr)
        out = xp.stack(blks, axis=0).reshape(*sh, -1)

        return pycu.view_as_real(out)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray
            (..., 2N.prod()) outputs :math:`\hat{\mathbf{x}} \in \mathbb{C}^{N_{1} \times \cdots
            \times N_{D}}` viewed as a real array. (See :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray
            * (...,  N.prod()) inputs :math:`\mathbf{x} \in \mathbb{R}^{N_{1} \times \cdots \times
              N_{D}}` (``real=True``.)
            * (..., 2N.prod()) inputs :math:`\mathbf{x} \in \mathbb{C}^{N_{1} \times \cdots \times
              N_{D}}` viewed as a real array. (See :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        arr = pycu.view_as_complex(arr)
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

        xp = pycu.get_array_module(arr)
        out = xp.stack(blks, axis=0).reshape(*sh, -1)

        if self._real:
            return out.real
        else:
            return pycu.view_as_real(out)
