import math
import types
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.math.stencil as pycstencil
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "Correlation",
    "Convolution",
]


def asarray(self, **kwargs) -> pyct.NDArray:
    r"""
    Make a matrix representation of the operator based on the stencil coefficients backend and dtype.
    """
    dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
    xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())
    dtype_ = self.stencil_coefs.dtype
    xp_ = pycu.get_array_module(self.stencil_coefs)
    E = xp_.eye(self.dim, dtype=dtype_)
    A = self.apply(E).T
    A = pycu.to_NUMPY(A) if xp_.__name__ == "cupy" and xp.__name__ != "cupy" else A

    return xp.array(A, dtype=dtype)


def lipschitz(self, **kwargs) -> pyct.Real:
    r"""
    Compute a Lipschitz constant of the stencil operator based on the stencil coefficients backend.
    """
    N = pycd.NDArrayInfo
    info = N.from_obj(self.stencil_coefs)
    kwargs.update(
        xp=info.module(),
        gpu=info == N.CUPY,
    )
    return pyca.LinOp.lipschitz(self, **kwargs)


class _Stencil(pyca.SquareOp):
    r"""
    Base class for NDArray computing functions that operate only on a local region of the NDArray through a
    multidimensional kernel, namely through correlation and convolution.
    This class leverages the :py:func:`numba.stencil` decorator, which allows to JIT (Just-In-Time) compile these
    functions to run more quickly.

    Parameters
    ----------
    stencil_coefs: NDArray
        Stencil coefficients. Must have the same number of dimensions as the input array's arg_shape (i.e., without the
        stacking dimension).
    center: NDArray
        Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``stencil_coefs``.
    arg_shape: tuple
        Shape of the input array.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mismatch issues.
    """

    def __init__(
        self,
        stencil_coefs: pyct.NDArray,
        center: pyct.NDArray,
        arg_shape: pyct.NDArrayShape,
        enable_warnings: bool = True,
    ):
        size = np.prod(arg_shape).item()

        super().__init__((size, size))

        self.arg_shape = arg_shape
        self.ndim = len(arg_shape)
        self._sanitize_inputs(stencil_coefs, center)
        self._make_stencils(self.stencil_coefs)
        self._lipschitz = 2 * abs(self.stencil_coefs).max()
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr", o=True)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be correlated with the kernel.
        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, correlated with kernel.
        """
        if (arr.dtype != self.stencil_coefs.dtype) and self._enable_warnings:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)

        return self._apply_dispatch(arr)

    @pycrt.enforce_precision(i="arr", o=True)
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be convolved with the kernel.
        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, convolved with kernel.
        """
        if (arr.dtype != self.stencil_coefs.dtype) and self._enable_warnings:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)

        return self._adjoint_dispatch(arr)

    def asarray(self, **kwargs) -> pyct.NDArray:
        # need to overwrite as stencils are not (hardware/precision) agnostic
        return asarray(self, **kwargs)

    def lipschitz(self, **kwargs) -> pyct.Real:
        # need to overwrite as stencils are not (hardware/precision) agnostic
        return lipschitz(self, **kwargs)

    def _apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.stencil(arr.reshape(-1, *self.arg_shape)).reshape(*arr.shape)

    def _apply_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return (
            arr.reshape(-1, *self.arg_shape)
            .map_overlap(self.stencil, depth=self.width, dtype=arr.dtype)
            .reshape(arr.shape)
        )

    def _apply_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr_shape = arr.shape
        arr = arr.reshape(-1, *self.arg_shape)
        out = xp.zeros_like(arr)
        # Cuda grid cannot have more than 3D. In the case of arg_shape with 3D, the cuda grid loops across the 3D and
        # looping over stacking dimension is done within the following Python list comprehension.
        tbp, bpg = self._get_gpu_config(arr)
        self.stencil[bpg, tbp](arr, out) if len(self.arg_shape) < 3 else [
            self.stencil[bpg, tbp](arr[i], out[i]) for i in range(arr.shape[0])
        ]
        return out.reshape(arr_shape)

    def _adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.stencil_adjoint(arr.reshape(-1, *self.arg_shape)).reshape(arr.shape)

    def _adjoint_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return (
            arr.reshape(-1, *self.arg_shape)
            .map_overlap(
                self.stencil_adjoint,
                depth=self.width,
                dtype=arr.dtype,
            )
            .reshape(arr.shape)
        )

    def _adjoint_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr_shape = arr.shape
        arr = arr.reshape(-1, *self.arg_shape)
        out = xp.zeros_like(arr)
        tbp, bpg = self._get_gpu_config(arr)
        self.stencil_adjoint[bpg, tbp](arr, out) if len(self.arg_shape) < 3 else [
            self.stencil_adjoint[bpg, tbp](arr[i], out[i]) for i in range(arr.shape[0])
        ]
        return out.reshape(arr_shape)

    @pycu.redirect("arr", DASK=_apply_dask, CUPY=_apply_cupy)
    def _apply_dispatch(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._apply(arr)

    @pycu.redirect("arr", DASK=_adjoint_dask, CUPY=_adjoint_cupy)
    def _adjoint_dispatch(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._adjoint(arr)

    def _make_stencils_cpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        # Create numba JIT-ted stencil functions for apply and adjoint methods.
        self.stencil = pycstencil.make_nd_stencil(coefficients=self.stencil_coefs, center=self.center)
        self.stencil_adjoint = pycstencil.make_nd_stencil(
            coefficients=self.stencil_coefs_adjoint, center=self.center_adjoint
        )

    def _make_stencils_gpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        # Create numba.cuda JIT-ted functions for apply and adjoint methods.
        self.stencil = pycstencil.make_nd_stencil_gpu(
            coefficients=self.stencil_coefs, center=self.center, func_name="apply"
        )
        self.stencil_adjoint = pycstencil.make_nd_stencil_gpu(
            coefficients=self.stencil_coefs_adjoint, center=self.center_adjoint, func_name="adjoint"
        )

    @pycu.redirect("stencil_coefs", CUPY=_make_stencils_gpu)
    def _make_stencils(self, stencil_coefs: pyct.NDArray) -> None:
        self._make_stencils_cpu(stencil_coefs)

    def _sanitize_inputs(self, stencil_coefs: pyct.NDArray, center: pyct.NDArray):
        # Check that inputs have the correct shape and correctly handle the boundary conditions.
        assert len(center) == stencil_coefs.ndim == self.ndim, (
            "The stencil coefficients should have the same"
            " number of dimensions as `arg_shape` and the "
            "same length as `center`."
        )
        self.xp = xp = pycu.get_array_module(stencil_coefs)
        self.stencil_coefs = stencil_coefs
        self.center = np.atleast_1d(center)
        self.stencil_coefs_adjoint = xp.flip(stencil_coefs)
        self.center_adjoint = np.array(stencil_coefs.shape) - 1 - np.atleast_1d(center)
        self.width = self._set_width(stencil_coefs.ndim)

    def _set_width(self, ndim):
        # set appropriate padding depth for different backends
        width_right = np.atleast_1d(self.stencil_coefs.shape) - self.center - 1
        return tuple([(0, 0)] + [(self.center[i].item(), width_right[i].item()) for i in range(ndim)])

    def _get_gpu_config(self, arr):
        # Get max number of threads in device
        t_max = arr.device.attributes["MaxThreadsPerBlock"]
        # Set at least as many threads as kernel elements per dimension
        _next_power_of_2 = lambda x: 1 if x == 0 else 2 ** (x - 1).bit_length()
        kernel_shape = self.stencil_coefs.shape
        tpb = [int(_next_power_of_2(kernel_shape[d])) for d in range(len(kernel_shape))]
        # Set maximum number of threads in the row-major order
        tpb[-1] = int(t_max / (np.prod(tpb) / tpb[-1]))
        # If kernel has less than 3D, add stacking dimension
        if len(self.arg_shape) < 3:
            tpb = [1] + tpb
        # If nthreads larger than a given array dimension size, use threads in other dimensions
        # This maximizes locality of cached memory (row-major order) to improve performance
        for i in range(len(tpb) - 1, -1, -1):
            while tpb[i] > self.arg_shape[i - 1] + np.sum(self.width[i]):
                tpb[i] = int(tpb[i] / 2)
                if i > 0:
                    tpb[i - 1] = int(tpb[i - 1] * 2)

        threadsperblock = tuple(tpb)

        # Define blockspergrid based on input array shape and threadsperblock
        aux_stacking = 0 if len(self.arg_shape) < 3 else 1
        blockspergrid = tuple([math.ceil(arr.shape[i + aux_stacking] / tpb) for i, tpb in enumerate(threadsperblock)])
        return threadsperblock, blockspergrid


def Correlation(
    stencil_coefs: pyct.NDArray,
    center: pyct.NDArray,
    arg_shape: pyct.NDArrayShape,
    mode: pycl.Pad.ModeSpec = "constant",
    enable_warnings: bool = True,
):
    r"""
    Parameters
    ----------
    stencil_coefs: NDArray
        Stencil coefficients. Must have the same number of dimensions as the input array's arg_shape (i.e., without the
        stacking dimension).
    center: NDArray
        Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``stencil_coefs``.
    arg_shape: tuple
        Shape of the input array.
    mode: str | list(str)
        Padding mode.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (zero-padding)
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: pad dimension[k] using `mode[k]`.

        (See :py:func:`numpy.pad` for details.)
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mismatch issues.


    Examples
    --------
    The following example creates a Correlation operator based on a 2-dimensional kernel.

    .. code-block:: python3

       from pycsou.operator.linop.stencil import Correlation
       import numpy as np
       import cupy as cp
       import dask.array as da
       nsamples = 2
       data_shape = (500, 100)
       da_blocks = (50, 10)
       # Numpy
       data_np = np.ones((nsamples, *data_shape)).reshape(nsamples, -1)
       # Cupy
       data_cp = cp.ones((nsamples, *data_shape)).reshape(nsamples, -1)
       # Dask
       data_da = da.from_array(data, chunks=da_blocks).reshape(nsamples, -1)
       kernel = np.array([[0.5, 0.0, 0.5],
                          [0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.5]])
       center = np.array([1, 0])
       stencil = StencilOp(stencil_coefs=kernel, center=center, arg_shape=data_shape, boundary=0.)
       stencil_cp = StencilOp(stencil_coefs=cp.asarray(kernel), center=center, arg_shape=data_shape, boundary=0.)
       # Correlate images with kernels
       out_np = stencil(data_np).reshape(nsamples, *data_shape)
       out_da = stencil(data_da).reshape(nsamples, *data_shape).compute()
       out_cp = stencil_cu(data_cp).reshape(nsamples, *data_shape).get()


    Notes
    -----
    Note that to perform correlation operations on GPU NDArrays, the operator has to be instantiated with GPU kernel
    coefficients.

    - **Remark 1**. When instantiated with a multi-dimensional kernel, the
    :py:class:`~pycsou.operator.linop.stencil.Correlate` performs correlation operations as non-separable filters.
    When possible, the user can decide whether to separate the filtering operation by composing different operators for
    different axis to accelerate performance. This approach is not guaranteed to improve performance due to the repeated
    copying of arrays associated to internal padding operations.


    - **Remark 2**. By default, for GPU computing, the ``threadsperblock`` argument is set according to the following criteria:

        - Number of the  GPU's threads per block (:math:`c`), i.e.,:

            .. math::
                \prod_{i=0}^{D-1} t_{i} \leq c

            where :math:`t_{i}` is the number of threads per block in dimension :math:`i`, :math:`D` is the number of dimensions
            of the kernel.

        - Maximum number of contiguous threads as possible:
            Because arrays are stored in row-major order, a larger number of threads per block in the last axis of the CuPy
            array benefits the spatial locality in memory caching. For this reason ``threadsperblock`` is set to the maximum
            number in the last axis, and to the minimum possible (respecting the kernel shape) in the other axes.

            .. math::
               t_{i} = 2^{j} \leq k_{i}, s.t., 2^{j+1} > k_{i} \quad \text{for} \quad i\in[0, \dots, D-2],


    .. warning::
       Due to code compilation the stencil methods assume arrays are in row-major or C order. If the input array is in
       Fortran or F order, a copy in C order is created automatically, which can lead to increased time and memory
       usage.
    """

    width_right = np.atleast_1d(stencil_coefs.shape) - center - 1
    widths = tuple([(max(center[i].item(), width_right[i].item()),) * 2 for i in range(len(arg_shape))])
    pad_shape = list(arg_shape)
    for i, (lhs, rhs) in enumerate(widths):
        pad_shape[i] += lhs + rhs
    pad_shape = tuple(pad_shape)
    padder = pycl.Pad(arg_shape=arg_shape, pad_width=widths, mode=mode)
    stencil = _Stencil(stencil_coefs=stencil_coefs, center=center, arg_shape=pad_shape, enable_warnings=enable_warnings)
    trimmer = pycl.Trim(arg_shape=pad_shape, trim_width=widths)

    op = trimmer * stencil * padder
    op.asarray = types.MethodType(asarray, op)
    op.lipschitz = types.MethodType(lipschitz, op)
    op._name = "Correlation"
    # store for testing
    op.arg_shape = arg_shape
    op.stencil_coefs = stencil.stencil_coefs
    op.center = stencil.center
    op._mode = padder._mode
    return op


def Convolution(
    stencil_coefs: pyct.NDArray,
    center: pyct.NDArray,
    arg_shape: pyct.NDArrayShape,
    mode: pycl.Pad.ModeSpec = "constant",
    enable_warnings: bool = True,
):
    r"""
    Parameters
    ----------
    stencil_coefs: NDArray
        Stencil coefficients. Must have the same number of dimensions as the input array's arg_shape (i.e., without the
        stacking dimension).
    center: NDArray
        Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``stencil_coefs``.
    arg_shape: tuple
        Shape of the input array.
    mode: str | list(str)
        Padding mode.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (zero-padding)
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: pad dimension[k] using `mode[k]`.

        (See :py:func:`numpy.pad` for details.)
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mismatch issues.


    Examples
    --------
    The following example creates a Convolution operator based on a 2-dimensional kernel.

    .. code-block:: python3

       from pycsou.operator.linop.stencil import Convolution
       import numpy as np
       import cupy as cp
       import dask.array as da
       nsamples = 2
       data_shape = (500, 100)
       da_blocks = (50, 10)
       # Numpy
       data_np = np.ones((nsamples, *data_shape)).reshape(nsamples, -1)
       # Cupy
       data_cp = cp.ones((nsamples, *data_shape)).reshape(nsamples, -1)
       # Dask
       data_da = da.from_array(data, chunks=da_blocks).reshape(nsamples, -1)
       kernel = np.array([[0.5, 0.0, 0.5],
                          [0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.5]])
       center = np.array([1, 0])
       stencil = StencilOp(stencil_coefs=kernel, center=center, arg_shape=data_shape, boundary=0.)
       stencil_cp = StencilOp(stencil_coefs=cp.asarray(kernel), center=center, arg_shape=data_shape, boundary=0.)
       # Correlate images with kernels
       out_np = stencil(data_np).reshape(nsamples, *data_shape)
       out_da = stencil(data_da).reshape(nsamples, *data_shape).compute()
       out_cp = stencil_cu(data_cp).reshape(nsamples, *data_shape).get()


    Notes
    -----
    Note that to perform convolution operations on GPU NDArrays, the operator has to be instantiated with GPU kernel
    coefficients.

    - **Remark 1**. When instantiated with a multi-dimensional kernel, the
    :py:class:`~pycsou.operator.linop.stencil.Convolve` performs convolution operations as non-separable filters.
    When possible, the user can decide whether to separate the filtering operation by composing different operators for
    different axis to accelerate performance. This approach is not guaranteed to improve performance due to the repeated
    copying of arrays associated to internal padding operations.


    - **Remark 2**. By default, for GPU computing, the ``threadsperblock`` argument is set according to the following criteria:

        - Number of the  GPU's threads per block (:math:`c`), i.e.,:

            .. math::
                \prod_{i=0}^{D-1} t_{i} \leq c

            where :math:`t_{i}` is the number of threads per block in dimension :math:`i`, :math:`D` is the number of dimensions
            of the kernel.

        - Maximum number of contiguous threads as possible:
            Because arrays are stored in row-major order, a larger number of threads per block in the last axis of the CuPy
            array benefits the spatial locality in memory caching. For this reason ``threadsperblock`` is set to the maximum
            number in the last axis, and to the minimum possible (respecting the kernel shape) in the other axes.


    .. warning::
       Due to code compilation the stencil methods assume arrays are in row-major or C order. If the input array is in
       Fortran or F order, a copy in C order is created automatically, which can lead to increased time and memory
       usage.
    """
    width_right = np.atleast_1d(stencil_coefs.shape) - center - 1
    widths = tuple([(max(center[i].item(), width_right[i].item()),) * 2 for i in range(len(arg_shape))])
    pad_shape = list(arg_shape)
    for i, (lhs, rhs) in enumerate(widths):
        pad_shape[i] += lhs + rhs
    pad_shape = tuple(pad_shape)
    padder = pycl.Pad(arg_shape=arg_shape, pad_width=widths, mode=mode)
    stencil = _Stencil(stencil_coefs=stencil_coefs, center=center, arg_shape=pad_shape, enable_warnings=enable_warnings)
    trimmer = pycl.Trim(arg_shape=pad_shape, trim_width=widths)

    op = trimmer * stencil.T * padder
    op.asarray = types.MethodType(asarray, op)
    op.lipschitz = types.MethodType(lipschitz, op)
    op._name = "Convolution"
    # store for testing
    op.arg_shape = arg_shape
    op.stencil_coefs = stencil.stencil_coefs
    op.center = stencil.center
    op._mode = padder._mode
    return op
