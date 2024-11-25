import collections.abc as cabc
import itertools

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
import pyxu.operator.linop.diff as pxld
import pyxu.operator.linop.pad as pxlp
import pyxu.operator.linop.stencil.stencil as pxls
import pyxu.runtime as pxrt
import pyxu.util as pxu

try:
    import scipy.ndimage._filters as scif
except ImportError:
    import scipy.ndimage.filters as scif

import functools
import typing as typ

import numpy as np

IndexSpec = cabc.Sequence[pxt.Integer]
KernelSpec = pxls.Stencil.KernelSpec
ModeSpec = pxlp.Pad.ModeSpec


__all__ = [
    "MovingAverage",
    "GaussianFilter",
    "DifferenceOfGaussians",
    "DoG",
    "Laplace",
    "Sobel",
    "Prewitt",
    "Scharr",
    "StructureTensor",
]


def _to_canonical_form(_, dim_shape):
    if not isinstance(_, cabc.Sequence):
        _ = (_,) * len(dim_shape)
    else:
        assert len(_) == len(dim_shape)
    return _


def _get_axes(axis, ndim):
    if axis is None:
        axes = list(range(ndim))
    elif np.isscalar(axis):
        axes = [axis]
    else:
        axes = axis
    return axes


def _sanitize_inputs(dim_shape, dtype, gpu):
    ndim = len(dim_shape)

    if dtype is None:
        dtype = pxrt.Width.DOUBLE.value

    if gpu:
        assert pxd.CUPY_ENABLED
        import cupy as xp
    else:
        import numpy as xp
    return ndim, dtype, xp


def MovingAverage(
    dim_shape: pxt.NDArrayShape,
    size: typ.Union[typ.Tuple, pxt.Integer],
    center: typ.Optional[IndexSpec] = None,
    mode: ModeSpec = "constant",
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional moving average or uniform filter.

    Notes
    -----
    This operator performs a convolution between the input :math:`D`-dimensional NDArray :math:`\mathbf{x} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` and a uniform :math:`D`-dimensional filter :math:`\mathbf{h} \in
    \mathbb{R}^{\text{size} \times \cdots \times \text{size}}` that computes the `size`-point local mean values using
    separable kernels for improved performance.

    .. math::

       y_{i} = \frac{1}{|\mathcal{N}_{i}|}\sum_{j \in \mathcal{N}_{i}} x_{j}

    Where :math:`\mathcal{N}_{i}` is the set of elements neighbouring the :math:`i`-th element of the input array, and
    :math:`\mathcal{N}_{i}` denotes its cardinality, i.e. the total number of neighbors.


    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    size: int, tuple
        Size of the moving average kernel.

        If a single integer value is provided, then the moving average filter will have as many dimensions as the input
        array.  If a tuple is provided, it should contain as many elements as `dim_shape`.  For example, the ``size=(1,
        3)`` will convolve the input image with the filter ``[[1, 1, 1]] / 3``.

    center: ~pyxu.operator.linop.filter.IndexSpec
        (i_1, ..., i_D) index of the kernel's center.

        `center` defines how a kernel is overlaid on inputs to produce outputs.
        For odd `size`, it defaults to the central element (``center=size//2``).
        For even `size` the desired center indices must be provided.

    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        MovingAverage

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import MovingAverage

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       ma = MovingAverage(dim_shape, size=5)
       out = ma(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.GaussianFilter`
    """
    size = _to_canonical_form(size, dim_shape)
    if center is None:
        assert all([s % 2 == 1 for s in size]), (
            "Can only infer center for odd `size`s. For even `size`s, please " "provide the desired `center`s."
        )
        center = [s // 2 for s in size]

    ndim, dtype, xp = _sanitize_inputs(dim_shape, dtype, gpu)

    kernel = [xp.ones(s, dtype=dtype) for s in size]  # use separable filters
    scale = 1 / np.prod(size)

    op = scale * pxls.Stencil(dim_shape=dim_shape, kernel=kernel, center=center, mode=mode)
    op._name = "MovingAverage"
    return op


def GaussianFilter(
    dim_shape: pxt.NDArrayShape,
    sigma: typ.Union[typ.Tuple[pxt.Real], pxt.Real] = 1.0,
    truncate: typ.Union[typ.Tuple[pxt.Real], pxt.Real] = 3.0,
    order: typ.Union[typ.Tuple[pxt.Integer], pxt.Integer] = 0,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Gaussian filter.

    Notes
    -----
    This operator performs a convolution between the input :math:`D`-dimensional NDArray :math:`\mathbf{x} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` and a Gaussian :math:`D`-dimensional filter :math:`\mathbf{h} \in
    \mathbb{R}^{\text{size} \times \cdots \times \text{size}}` using separable kernels for improved performance.

    .. math::

       y_{i} = \sum_{j \in \mathcal{N}_{i}} a_{j} x_{j} \exp(\frac{d_{ij}^{2}}{\sigma^{2}})

    Where :math:`\mathcal{N}_{i}` is the set of elements neighbouring the :math:`i`-th element of the input array, and
    :math:`a_{j} = \sum_{j \in \mathcal{N}_{i}} a_{j} \exp(\frac{d_{ij}^{2}}{\sigma^{2}})` normalizes the kernel to sum
    to one.


    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    sigma: float, tuple
        Standard deviation of the Gaussian kernel.

        If a scalar value is provided, then the Gaussian filter will have as many dimensions as the input array.
        If a tuple is provided, it should contain as many elements as `dim_shape`.
        Use ``0`` to prevent filtering in a given dimension.
        For example, the ``sigma=(0, 3)`` will convolve the input image in its last dimension.
    truncate: float, tuple
        Truncate the filter at this many standard deviations.
        Defaults to 3.0.
    order: int, tuple
        Gaussian derivative order.

        Use ``0`` for the standard Gaussian kernel.
    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
        Sampling step (i.e. distance between two consecutive elements of an array).
        Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU).
        Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        GaussianFilter

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import GaussianFilter

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       gaussian = GaussianFilter(dim_shape, sigma=3)
       out = gaussian(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.MovingAverage`,
    :py:class:`~pyxu.operator.DifferenceOfGaussians`
    """

    ndim, dtype, xp = _sanitize_inputs(dim_shape, dtype, gpu)
    sigma = _to_canonical_form(sigma, dim_shape)
    truncate = _to_canonical_form(truncate, dim_shape)
    order = _to_canonical_form(order, dim_shape)
    sampling = _to_canonical_form(sampling, dim_shape)

    kernel = [
        xp.array([1], dtype=dtype),
    ] * len(dim_shape)
    center = [0 for _ in range(len(dim_shape))]

    for i, (sigma_, truncate_, order_, sampling_) in enumerate(zip(sigma, truncate, order, sampling)):
        if sigma_:
            sigma_pix = sigma_ / sampling_
            radius = int(truncate_ * float(sigma_pix) + 0.5)
            kernel[i] = xp.asarray(np.flip(scif._gaussian_kernel1d(sigma_pix, order_, radius)), dtype=dtype)
            kernel[i] /= sampling_**order_
            center[i] = radius

    op = pxls.Stencil(dim_shape=dim_shape, kernel=kernel, center=center, mode=mode)
    op._name = "GaussianFilter"
    return op


def DifferenceOfGaussians(
    dim_shape: pxt.NDArrayShape,
    low_sigma=1.0,
    high_sigma=None,
    low_truncate=3.0,
    high_truncate=3.0,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Difference of Gaussians filter.

    Notes
    -----

    This operator uses the Difference of Gaussians (DoG) method to a :math:`D`-dimensional NDArray :math:`\mathbf{x} \in
    \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` using separable kernels for improved performance.
    The DoG method blurs the input image with two Gaussian kernels with different sigma, and subtracts the more-blurred
    signal from the less-blurred image.
    This creates an output signal containing only the information from the original signal at the spatial scale
    indicated by the two sigmas.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    low_sigma: float, tuple
        Standard deviation of the Gaussian kernel with smaller sigmas across all axes.

        If a scalar value is provided, then the Gaussian filter will have as many dimensions as the input array.
        If a tuple is provided, it should contain as many elements as `dim_shape`.
        Use ``0`` to prevent filtering in a given dimension.
        For example, the ``low_sigma=(0, 3)`` will convolve the input image in its last dimension.
    high_sigma: float, tuple, None
        Standard deviation of the Gaussian kernel with larger sigmas across all axes.
        If ``None`` is given (default), sigmas for all axes are calculated as ``1.6 * low_sigma``.
    low_truncate: float, tuple
        Truncate the filter at this many standard deviations.
        Defaults to 3.0.
    high_truncate: float, tuple
        Truncate the filter at this many standard deviations.
        Defaults to 3.0.
    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
        Sampling step (i.e. distance between two consecutive elements of an array).
        Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        DifferenceOfGaussians

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import DoG

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       dog = DoG(dim_shape, low_sigma=3)
       out = dog(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.Gaussian`,
    :py:class:`~pyxu.operator.Sobel`,
    :py:class:`~pyxu.operator.Prewitt`,
    :py:class:`~pyxu.operator.Scharr`,
    :py:class:`~pyxu.operator.StructureTensor`
    """

    low_sigma = _to_canonical_form(low_sigma, dim_shape)
    if high_sigma is None:
        high_sigma = tuple(s * 1.6 for s in low_sigma)

    high_sigma = _to_canonical_form(high_sigma, dim_shape)
    low_truncate = _to_canonical_form(low_truncate, dim_shape)
    high_truncate = _to_canonical_form(high_truncate, dim_shape)

    kwargs = {
        "dim_shape": dim_shape,
        "order": 0,
        "mode": mode,
        "gpu": gpu,
        "dtype": dtype,
        "sampling": sampling,
    }
    op_low = GaussianFilter(sigma=low_sigma, truncate=low_truncate, **kwargs)
    op_high = GaussianFilter(sigma=high_sigma, truncate=high_truncate, **kwargs)
    op = op_low - op_high
    op._name = "DifferenceOfGaussians"
    return op


DoG = DifferenceOfGaussians  #: Alias of :py:func:`~pyxu.operator.DifferenceOfGaussians`.


def Laplace(
    dim_shape: pxt.NDArrayShape,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Laplace filter.

    The implementation is based on second derivatives approximated via finite differences.

    Notes
    -----
    This operator uses the applies the Laplace kernel :math:`[1 -2 1]` to a :math:`D`-dimensional NDArray
    :math:`\mathbf{x} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` using separable kernels for improved
    performance.  The Laplace filter is commonly used to find high-frequency components in the signal, such as for
    example, the edges in an image.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        DifferenceOfGaussians

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import Laplace

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       laplace = Laplace(dim_shape)
       out = laplace(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.Sobel`,
    :py:class:`~pyxu.operator.Prewitt`,
    :py:class:`~pyxu.operator.Scharr`,
    """

    ndim, dtype, xp = _sanitize_inputs(dim_shape, dtype, gpu)
    sampling = _to_canonical_form(sampling, dim_shape)
    centers = [[1 if i == dim else 0 for i in range(ndim)] for dim in range(ndim)]
    kernels = [
        xp.array([1.0, -2.0, 1.0], dtype=dtype).reshape([-1 if i == dim else 1 for i in range(ndim)]) / sampling[dim]
        for dim in range(ndim)
    ]
    ops = [pxls.Stencil(dim_shape=dim_shape, kernel=k, center=c, mode=mode) for (k, c) in zip(kernels, centers)]
    op = functools.reduce(lambda x, y: x + y, ops)
    op._name = "Laplace"
    return op


def Sobel(
    dim_shape: pxt.NDArrayShape,
    axis: typ.Optional[typ.Tuple] = None,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Sobel filter.

    Notes
    -----

    This operator uses the applies the multi-dimensional Sobel filter to a :math:`D`-dimensional NDArray
    :math:`\mathbf{x} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` using separable kernels for improved
    performance.  The Sobel filter applies the following edge filter in the dimensions of interest: ``[1, 0, -1]`` and
    the smoothing filter on the rest of dimensions: ``[1, 2, 1] / 4``.  The Sobel filter is commonly used to find
    high-frequency components in the signal, such as for example, the edges in an image.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    axis: int, tuple
        Compute the edge filter along this axis. If not provided, the edge magnitude is computed.

        This is defined as: ``np.sqrt(sum([sobel(array, axis=i)**2 for i in range(array.ndim)]) / array.ndim)`` The
        magnitude is also computed if axis is a sequence.

    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        Sobel

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import Sobel

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       sobel = Sobel(dim_shape)
       out = sobel(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.Prewitt`,
    :py:class:`~pyxu.operator.Scharr`,
    """
    smooth_kernel = np.array([1, 2, 1]) / 4
    return _EdgeFilter(
        dim_shape=dim_shape,
        smooth_kernel=smooth_kernel,
        filter_name="SobelFilter",
        axis=axis,
        mode=mode,
        sampling=sampling,
        gpu=gpu,
        dtype=dtype,
    )


def Prewitt(
    dim_shape: pxt.NDArrayShape,
    axis: typ.Optional[typ.Tuple] = None,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Prewitt filter.

    Notes
    -----

    This operator uses the applies the multi-dimensional Prewitt filter to a :math:`D`-dimensional NDArray
    :math:`\mathbf{x} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` using separable kernels for improved
    performance.  The Prewitt filter applies the following edge filter in the dimensions of interest: ``[1, 0, -1]``,
    and the smoothing filter on the rest of dimensions: ``[1, 1, 1] / 3``.  The Prewitt filter is commonly used to find
    high-frequency components in the signal, such as for example, the edges in an image.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    axis: int, tuple
        Compute the edge filter along this axis. If not provided, the edge magnitude is computed. This is defined as:

        ``np.sqrt(sum([prewitt(array, axis=i)**2 for i in range(array.ndim)]) / array.ndim)`` The magnitude is also
        computed if axis is a sequence.

    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        Prewitt

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import Prewitt

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       prewitt = Prewitt(dim_shape)
       out = prewitt(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.Sobel`,
    :py:class:`~pyxu.operator.Scharr`,
    """
    smooth_kernel = np.full((3,), 1 / 3)
    return _EdgeFilter(
        dim_shape=dim_shape,
        smooth_kernel=smooth_kernel,
        filter_name="Prewitt",
        axis=axis,
        mode=mode,
        sampling=sampling,
        gpu=gpu,
        dtype=dtype,
    )


def Scharr(
    dim_shape: pxt.NDArrayShape,
    axis: typ.Optional[typ.Tuple] = None,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    r"""
    Multi-dimensional Scharr filter.

    Notes
    -----

    This operator uses the applies the multi-dimensional Scharr filter to a :math:`D`-dimensional NDArray
    :math:`\mathbf{x} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` using separable kernels for improved
    performance.  The Scharr filter applies the following edge filter in the dimensions of interest: ``[1, 0, -1]``, and
    the smoothing filter on the rest of dimensions: ``[3, 10, 3] / 16``.  The Scharr filter is commonly used to find
    high-frequency components in the signal, such as for example, the edges in an image.

    Parameters
    ----------
    dim_shape: NDArrayShape
        Shape of the input array.
    axis: int, tuple
        Compute the edge filter along this axis. If not provided, the edge magnitude is computed. This is defined as:

        ``np.sqrt(sum([scharr(array, axis=i)**2 for i in range(array.ndim)]) / array.ndim)`` The magnitude is also
        computed if axis is a sequence.
    mode: str, list[str]
        Boundary conditions.
        Multiple forms are accepted:

        * str: unique mode shared amongst dimensions.
          Must be one of:

          * 'constant' (default): zero-padding
          * 'wrap'
          * 'reflect'
          * 'symmetric'
          * 'edge'
        * tuple[str, ...]: the `d`-th dimension uses `mode[d]` as boundary condition.

        (See :py:func:`numpy.pad` for details.)
    sampling: Real, list[Real]
            Sampling step (i.e. distance between two consecutive elements of an array).
            Defaults to 1.
    gpu: bool
        Input NDArray type (`True` for GPU, `False` for CPU). Defaults to `False`.
    dtype: DType
        Working precision of the linear operator.

    Returns
    -------
    op: OpT
        Scharr

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pyxu.operator import Scharr

       dim_shape = (11, 11)
       image = np.zeros(dim_shape)
       image[5, 5] = 1.

       scharr = Scharr(dim_shape)
       out = scharr(image)
       plt.figure(figsize=(10, 5))
       plt.subplot(121)
       plt.imshow(image)
       plt.colorbar()
       plt.subplot(122)
       plt.imshow(out)
       plt.colorbar()

    See Also
    --------
    :py:class:`~pyxu.operator.Sobel`,
    :py:class:`~pyxu.operator.Prewitt`,
    """
    smooth_kernel = np.array([3, 10, 3]) / 16
    return _EdgeFilter(
        dim_shape=dim_shape,
        smooth_kernel=smooth_kernel,
        filter_name="Scharr",
        axis=axis,
        mode=mode,
        sampling=sampling,
        gpu=gpu,
        dtype=dtype,
    )


def _EdgeFilter(
    dim_shape: pxt.NDArrayShape,
    smooth_kernel: KernelSpec,
    filter_name: str,
    axis: typ.Optional[typ.Tuple] = None,
    mode: ModeSpec = "constant",
    sampling: typ.Union[pxt.Real, cabc.Sequence[pxt.Real, ...]] = 1,
    gpu: bool = False,
    dtype: typ.Optional[pxt.DType] = None,
):
    from pyxu.operator import Sqrt, Square

    square = Square(dim_shape=dim_shape)
    sqrt = Sqrt(dim_shape=dim_shape)

    ndim, dtype, xp = _sanitize_inputs(dim_shape, dtype, gpu)
    sampling = _to_canonical_form(sampling, dim_shape)

    axes = _get_axes(axis, ndim)

    return_magnitude = len(axes) > 1

    op_list = []
    for edge_dim in axes:
        kernel = [xp.array(1, dtype=dtype)] * len(dim_shape)
        center = np.ones(len(dim_shape), dtype=int)
        # We define the kernel reversed compared to Scipy or Skimage because we use correlation instead of convolution
        kernel[edge_dim] = xp.array([-1, 0, 1], dtype=dtype) / sampling[edge_dim]
        smooth_axes = list(set(range(ndim)) - {edge_dim})
        for smooth_dim in smooth_axes:
            kernel[smooth_dim] = xp.asarray(smooth_kernel, dtype=dtype) / sampling[smooth_dim]

        if return_magnitude:
            op_list.append(square * pxls.Stencil(dim_shape=dim_shape, kernel=kernel, center=center, mode=mode))
        else:
            op_list.append(pxls.Stencil(dim_shape=dim_shape, kernel=kernel, center=center, mode=mode))

    op = functools.reduce(lambda x, y: x + y, op_list)
    if return_magnitude:
        op = (1 / np.sqrt(ndim)) * (sqrt * op)

    op._name = filter_name
    return op


class StructureTensor(pxa.DiffMap):
    r"""
    Structure tensor operator.

    Notes
    -----
    The Structure Tensor, also known as the second-order moment tensor or the inertia tensor, is a matrix derived from
    the gradient of a function. It describes the distribution of the gradient (i.e., its prominent directions) in a
    specified neighbourhood around a point, and the degree to which those directions are coherent.  The structure tensor
    of a :math:`D`-dimensional signal :math:`\mathbf{f} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` can be
    written as:

    .. math::

        \mathbf{S}_\sigma \mathbf{f} = \mathbf{g}_{\sigma} * \nabla\mathbf{f} (\nabla\mathbf{f})^{\top} = \mathbf{g}_{\sigma} *
        \begin{bmatrix}
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \right)^2 &  \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\,\partial x_{1} } & \cdots & \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \\
        \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } & \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{1} }\right)^2 & \cdots & \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \\
        \vdots & \vdots & \ddots & \vdots \\
        \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } & \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1} } \dfrac{ \partial\mathbf{f} }{ \partial x_{1} } & \cdots & \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1}} \right)^2
        \end{bmatrix},

    where :math:`\mathbf{g}_{\sigma} \in \mathbb{R}^{N_0 \times \cdots \times N_{D-1}}` is a discrete Gaussian filter
    with standard variation :math:`\sigma` with which a convolution is performed elementwise.

    However, due to the symmetry of the structure tensor, only the upper triangular part is computed in practice:

    .. math::

        \mathbf{H}_{\mathbf{v}_1, \ldots ,\mathbf{v}_m} \mathbf{f} = \mathbf{g}_{\sigma} * \begin{bmatrix}
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{0} } \right)^2 \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\,\partial x_{1} } \\
        \vdots \\
        \left( \dfrac{ \partial\mathbf{f} }{ \partial x_{D-1}} \right)^2
        \end{bmatrix} \mathbf{f} \in \mathbb{R}^{\frac{D (D-1)}{2} \times N_0 \times \cdots \times N_{D-1}}

    Remark
    ------
    In case of using the finite differences (`diff_type="fd"`), the finite difference scheme defaults to `central` (see
    :py:class:`~pyxu.operator.PartialDerivative`).

    Example
    -------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pyxu.operator import StructureTensor
       from pyxu.util.misc import peaks

       # Define input image
       n = 1000
       x = np.linspace(-3, 3, n)
       xx, yy = np.meshgrid(x, x)
       image = peaks(xx, yy)
       nsamples = 2
       dim_shape = image.shape  # (1000, 1000)
       images = np.tile(image, (nsamples, 1, 1))
       # Instantiate structure tensor operator
       structure_tensor = StructureTensor(dim_shape=dim_shape)

       outputs = structure_tensor(images)
       print(outputs.shape)  # (2, 3, 1000, 1000)
       # Plot
       plt.figure()
       plt.imshow(images[0])
       plt.colorbar()
       plt.title("Image")
       plt.axis("off")

       plt.figure()
       plt.imshow(outputs[0][0])
       plt.colorbar()
       plt.title(r"$\hat{S}_{xx}$")
       plt.axis("off")

       plt.figure()
       plt.imshow(outputs[0][1])
       plt.colorbar()
       plt.title(r"$\hat{S}_{xy}$")
       plt.axis("off")

       plt.figure()
       plt.imshow(outputs[0][2])
       plt.colorbar()
       plt.title(r"$\hat{S}_{yy}$")
       plt.axis("off")

    See Also
    --------
    :py:class:`~pyxu.operator.PartialDerivative`,
    :py:class:`~pyxu.operator.Gradient`,
    :py:class:`~pyxu.operator.Hessian`
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        diff_method="fd",
        smooth_sigma: typ.Union[pxt.Real, tuple[pxt.Real, ...]] = 1.0,
        smooth_truncate: typ.Union[pxt.Real, tuple[pxt.Real, ...]] = 3.0,
        mode: ModeSpec = "constant",
        sampling: typ.Union[pxt.Real, tuple[pxt.Real, ...]] = 1,
        gpu: bool = False,
        dtype: typ.Optional[pxt.DType] = None,
        parallel: bool = False,
        **diff_kwargs,
    ):
        ndim = len(dim_shape)
        ntriu = (ndim * (ndim + 1)) // 2
        super().__init__(dim_shape=dim_shape, codim_shape=(ntriu,) + dim_shape)
        self.directions = tuple(
            list(_) for _ in itertools.combinations_with_replacement(np.arange(len(dim_shape)).astype(int), 2)
        )

        if diff_method == "fd":
            diff_kwargs.update({"scheme": diff_kwargs.pop("scheme", "central")})

        self.grad = pxld.Gradient(
            dim_shape=dim_shape,
            directions=None,
            diff_method=diff_method,
            mode=mode,
            gpu=gpu,
            dtype=dtype,
            sampling=sampling,
            parallel=parallel,
            **diff_kwargs,
        )

        if smooth_sigma:
            self.smooth = GaussianFilter(
                dim_shape=dim_shape,
                sigma=smooth_sigma,
                truncate=smooth_truncate,
                order=0,
                mode=mode,
                sampling=sampling,
                gpu=gpu,
                dtype=dtype,
            )
        else:
            self.smooth = pxlb.IdentityOp(dim_shape=dim_shape)

    def apply(self, arr):
        xp = pxu.get_array_module(arr)
        sh = arr.shape[: -self.dim_rank]
        grad = self.grad(arr)

        def slice_(ax):
            return (slice(None),) * len(sh) + (slice(ax, ax + 1),)

        return xp.concatenate(
            [self.smooth((grad[slice_(i)] * grad[slice_(j)])) for i, j in self.directions],
            axis=len(sh),
        ).reshape(*sh, len(self.directions), *self.dim_shape)
