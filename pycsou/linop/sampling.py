# #############################################################################
# sampling.py
# ===========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Sampling operators.

This module provides sampling operators for discrete or continuous signals.
"""
import numpy as np
import pandas as pd
import pylops
import pygsp
from typing import Optional, Union, Tuple, Iterable, List, Callable
from pycsou.core.linop import PyLopLinearOperator, LinearOperator, IdentityOperator, DiagonalOperator, LinOpVStack, \
    SparseLinearOperator, PolynomialLinearOperator
from numbers import Number
from skimage.measure import block_reduce
from scipy.spatial import cKDTree


def SubSampling(size: int, sampling_indices: Union[np.ndarray, list], shape: Optional[tuple] = None, axis: int = 0,
                dtype: str = 'float64', inplace: bool = True):
    r"""
    Subsampling operator.

    Extract subset of values from input array at locations ``sampling_indices``
    in forward mode and place those values at locations ``sampling_indices``
    in an otherwise zero array in adjoint mode.

    Parameters
    ----------
    size : int
        Size of input array.
    sampling_indices : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of samples for data selection.
    shape : tuple
        Shape of input array
        (``None`` if only one dimension is available).
    axis : int
        When ``shape`` is not ``None``, axis along which subsampling is applied.
    dtype : str
        Type of elements in input array.
    inplace : bool
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).

    Returns
    -------
    :py:class:`~pycsou.core.linop.PyLopLinearOperator`
        The subsampling operator.

    Raises
    ------
    ValueError
        If shape and size do not match.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import SubSampling

    .. doctest::

       >>> x = np.arange(9).reshape(3,3)
       >>> sampling_indices = [0,2]
       >>> SamplingOp=SubSampling(size=x.size, sampling_indices=sampling_indices)
       >>> SamplingOp * x.reshape(-1)
       array([0, 2])
       >>> SamplingOp.adjoint(SamplingOp* x.reshape(-1)).reshape(x.shape)
       array([[0., 0., 2.],
              [0., 0., 0.],
              [0., 0., 0.]])
       >>> SamplingOp=SubSampling(size=x.size, sampling_indices=sampling_indices, shape=x.shape, axis=1)
       >>> (SamplingOp * x.reshape(-1)).reshape(x.shape[1], len(sampling_indices))
       array([[0, 2],
              [3, 5],
              [6, 8]])
       >>> SamplingOp.adjoint(SamplingOp* x.reshape(-1)).reshape(x.shape)
       array([[0., 0., 2.],
              [3., 0., 5.],
              [6., 0., 8.]])


    Notes
    -----
    Subsampling of a subset of :math:`L` values at locations
    ``sampling_indices`` from an input vector :math:`\mathbf{x}` of size
    :math:`N` can be expressed as:

    .. math::

        y_i = x_{n_i}  \quad \forall i=1,2,...,L,

    where :math:`\mathbf{n}=[n_1, n_2,..., n_L]` is a vector containing the indeces
    of the original array at which samples are taken.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`\mathbf{n}=[n_1, n_2,..., n_L]` in the model vector:

    .. math::

        x_{n_i} = y_i  \quad \forall i=1,2,...,L

    and :math:`x_{j}=0 \,\forall j \neq n_i` (i.e., at all other locations in input
    vector).


    See Also
    --------
    :py:class:`~pycsou.linop.sampling.Masking`, :py:class:`~pycsou.linop.sampling.Downsampling`
    """
    PyLop = pylops.Restriction(M=size, iava=sampling_indices, dims=shape, dir=axis, dtype=dtype, inplace=inplace)
    return PyLopLinearOperator(PyLop=PyLop, is_symmetric=False, is_dense=False, is_sparse=False)


class Masking(LinearOperator):
    r"""
    Masking operator.

    Extract subset of values from input array at locations marked as ``True`` in ``sampling_bool``.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import Masking, SubSampling

    .. doctest::

       >>> x = np.arange(9).reshape(3,3)
       >>> sampling_bool = np.zeros((3,3)).astype(bool)
       >>> sampling_bool[[1,2],[0,2]] = True
       >>> SamplingOp = Masking(size=x.size, sampling_bool=sampling_bool)
       >>> SamplingOp * x.reshape(-1)
       array([3, 8])
       >>> SamplingOp.adjoint(SamplingOp* x.reshape(-1)).reshape(x.shape)
       array([[0., 0., 0.],
              [3., 0., 0.],
              [0., 0., 8.]])
       >>> sampling_indices = np.nonzero(sampling_bool.reshape(-1))[0].astype(int)
       >>> SubSamplingOp=SubSampling(size=x.size, sampling_indices=sampling_indices)
       >>> np.allclose(SamplingOp * x.reshape(-1), SubSamplingOp * x.reshape(-1))
       True

    Notes
    -----
    For flattened arrays, the ``Masking`` operator is equivalent to the :py:func:`~pycsou.linop.sampling.SubSampling` operator,
    with the only difference that the sampling locations are specified in the form of a boolean array instead of indices.

    See Also
    --------
    :py:class:`~pycsou.linop.sampling.SubSampling`, :py:class:`~pycsou.linop.sampling.Downsampling`
    """

    def __init__(self, size: int, sampling_bool: Union[np.ndarray, list], dtype: type = np.float64):
        r"""

        Parameters
        ----------
        size : int
            Size of input array.
        sampling_bool : :obj:`list` or :obj:`numpy.ndarray`
            Boolean array for data selection. ``True`` values mark the positions of the samples.
        dtype : str
        Type of elements in input array.

        Raises
        ------
        ValueError
            If the size of ``sampling_bool`` differs from ``size``.
        """
        self.sampling_bool = np.asarray(sampling_bool).reshape(-1).astype(bool)
        self.input_size = size
        self.nb_of_samples = self.sampling_bool[self.sampling_bool == True].size
        if self.sampling_bool.size != size:
            raise ValueError('Invalid size of boolean sampling array.')
        super(Masking, self).__init__(shape=(self.nb_of_samples, self.input_size), dtype=dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[self.sampling_bool]

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        x = np.zeros(shape=self.input_size, dtype=self.dtype)
        x[self.sampling_bool] = y
        return x


class DownSampling(Masking):
    r"""
    Downsampling operator.

    Downsample an array in one of the three ways:

    - If ``shape`` is ``None``: The input array is flat and the downsampling selects one element every
      ``downsampling_factor`` elements.
    - If ``shape`` is not ``None`` and ``axis`` is ``None``: The input array is multidimensional, and each dimension is
      downsampled by a certain factor. Downsampling factors for each dimension are specified in the tuple
      ``downsampling_factor`` (if ``downsampling_factor`` is an integer, then the same factor is assumed for every dimension).
    - If ``shape`` is not ``None`` and ``axis`` is not ``None``: The input array is multidimensional, but only the dimension is
      specified by ``axis`` is downsampled by a ``downsampling_factor``.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import DownSampling

    .. doctest::

       >>> x = np.arange(10)
       >>> DownSamplingOp = DownSampling(size=x.size, downsampling_factor=3)
       >>> DownSamplingOp * x
       array([0, 3, 6, 9])
       >>> DownSamplingOp.adjoint(DownSamplingOp * x)
       array([0., 0., 0., 3., 0., 0., 6., 0., 0., 9.])
       >>> x = np.arange(20).reshape(4,5)
       >>> DownSamplingOp = DownSampling(size=x.size, shape=x.shape, downsampling_factor=3)
       >>> (DownSamplingOp * x.flatten()).reshape(DownSamplingOp.output_shape)
       array([[ 0,  3],
              [15, 18]])
       >>> DownSamplingOp = DownSampling(size=x.size, shape=x.shape, downsampling_factor=(2,4))
       >>> (DownSamplingOp * x.flatten()).reshape(DownSamplingOp.output_shape)
       array([[ 0,  4],
              [10, 14]])
       >>> DownSamplingOp.adjoint(DownSamplingOp * x.flatten()).reshape(x.shape)
       array([[ 0.,  0.,  0.,  0.,  4.],
              [ 0.,  0.,  0.,  0.,  0.],
              [10.,  0.,  0.,  0., 14.],
              [ 0.,  0.,  0.,  0.,  0.]])
       >>> DownSamplingOp = DownSampling(size=x.size, shape=x.shape, downsampling_factor=2, axis=-1)
       >>> (DownSamplingOp * x.flatten()).reshape(DownSamplingOp.output_shape)
       array([[ 0,  2,  4],
              [ 5,  7,  9],
              [10, 12, 14],
              [15, 17, 19]])
       >>> DownSamplingOp.adjoint(DownSamplingOp * x.flatten()).reshape(x.shape)
       array([[ 0.,  0.,  2.,  0.,  4.],
              [ 5.,  0.,  7.,  0.,  9.],
              [10.,  0., 12.,  0., 14.],
              [15.,  0., 17.,  0., 19.]])

    .. plot::

       import numpy as np
       from pycsou.linop.sampling import DownSampling
       import matplotlib.pyplot as plt
       import scipy.misc
       img = scipy.misc.face(gray=True).astype(float)
       DownSampOp = DownSampling(size=img.size, shape=img.shape, downsampling_factor=(3,6))
       down_sampled_img = (DownSampOp * img.flatten()).reshape(DownSampOp.output_shape)
       up_sampled_img = DownSampOp.adjoint(down_sampled_img.flatten()).reshape(img.shape)
       plt.figure()
       plt.imshow(img)
       plt.colorbar()
       plt.title('Original')
       plt.figure()
       plt.imshow(down_sampled_img)
       plt.colorbar()
       plt.title('Downsampling')
       plt.figure()
       plt.imshow(up_sampled_img)
       plt.colorbar()
       plt.title('Downsampling followed by Upsampling')

    Notes
    -----
    Downsampling by :math:`M` an input vector :math:`\mathbf{x}` of size
    :math:`N` can be performed as:

    .. math::

        y_i = x_{iM}  \quad  i=1,\ldots, \lfloor N/M \rfloor.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`n=iM` in the model vector:

    .. math::

        x_{iM} = y_i,\;\;x_{iM+1}=x_{iM+2}=\cdots=x_{(i+1)M-1}=0, \qquad i=1,\ldots, \lfloor N/M \rfloor.


    See Also
    --------
    :py:class:`~pycsou.linop.sampling.SubSampling`, :py:class:`~pycsou.linop.sampling.Masking`
    """

    def __init__(self, size: int, downsampling_factor: Union[int, tuple, list], shape: Optional[tuple] = None,
                 axis: Optional[int] = None, dtype: type = np.float64):
        """

        Parameters
        ----------
        size : int
            Size of input array.
        downsampling_factor : Union[int, tuple, list]
            Downsampling factor (possibly different for each dimension).
        shape: Optional[tuple]
            Shape of input array (default ``None``: the input array is 1D).
        axis: Optional[int]
            Axis along which to downsample for ND input arrays (default ``None``: downsampling is performed along each axis).
        dtype: type
            Type of input array.

        Raises
        ------
        ValueError
            If the set of parameters {``shape``, ``size``, ``sampling_factor``, ``axis``} is invalid.
        """
        if type(downsampling_factor) is int:
            if (shape is not None) and (axis is None):
                self.downsampling_factor = len(shape) * (downsampling_factor,)
            else:
                self.downsampling_factor = (downsampling_factor,)
        else:
            self.downsampling_factor = tuple(downsampling_factor)
        if shape is not None:
            if size != np.prod(shape):
                raise ValueError(f'Array size {size} is incompatible with array shape {shape}.')
            if (axis is not None) and (axis > len(shape) - 1):
                raise ValueError(f'Array size {size} is incompatible with array shape {shape}.')
        if (shape is None) and (len(self.downsampling_factor) > 1):
            raise ValueError('Please specify an array shape for multidimensional downsampling.')
        elif (shape is not None) and (axis is None) and (len(shape) != len(self.downsampling_factor)):
            raise ValueError(f'Inconsistent downsampling factors {downsampling_factor} for array of shape {shape}.')
        self.input_size = size
        self.input_shape = shape
        self.axis = axis
        self.downsampling_mask = self.compute_downsampling_mask()
        if self.input_shape is None:
            self.output_shape = None
        else:
            if len(self.downsampling_factor) > 1:
                output_shape = []
                for ax in range(len(self.input_shape)):
                    axis_indices = np.arange(self.input_shape[ax])
                    downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                    output_shape.append(downsampled_axis_indices[downsampled_axis_indices == 0].size)
                self.output_shape = tuple(output_shape)
            else:
                output_shape = list(self.input_shape)
                downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                output_shape[self.axis] = downsampled_axis_indices[downsampled_axis_indices == 0].size
                self.output_shape = tuple(output_shape)

        super(DownSampling, self).__init__(size=self.input_size, sampling_bool=self.downsampling_mask, dtype=dtype)

    def compute_downsampling_mask(self) -> np.ndarray:
        """
        Compute the downsampling mask.

        Returns
        -------
        :py:class:`np.ndarray`
            The mask to apply to get the downsampled values.
        """
        if self.input_shape is None:
            indices = np.arange(self.input_size)
            downsampled_mask = (indices % self.downsampling_factor) == 0
        else:
            if len(self.downsampling_factor) > 1:
                downsampled_mask = True
                for ax in range(len(self.input_shape)):
                    axis_indices = np.arange(self.input_shape[ax])
                    downsampled_axis_indices = axis_indices % self.downsampling_factor[ax]
                    downsampled_axis_indices = downsampled_axis_indices.reshape(
                        downsampled_axis_indices.shape + (len(self.input_shape) - 1) * (1,))
                    downsampled_axis_indices = np.swapaxes(downsampled_axis_indices, 0, ax)
                    downsampled_mask = downsampled_mask * (downsampled_axis_indices == 0)
            else:
                downsampled_mask = np.zeros(shape=self.input_shape, dtype=bool)
                downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
                downsampled_axis_indices = np.arange(self.input_shape[self.axis])
                downsampled_axis_indices = downsampled_axis_indices % self.downsampling_factor
                downsampled_mask[downsampled_axis_indices == 0, ...] = True
                downsampled_mask = np.swapaxes(downsampled_mask, 0, self.axis)
        return downsampled_mask.reshape(-1)


class Pooling(LinearOperator):
    r"""
    Pooling operator.

    Pool an array by summing/averaging across constant size blocks tiling the array.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import Pooling

    .. doctest::

       >>> x = np.arange(24).reshape(4,6)
       >>> PoolingOp = Pooling(shape=x.shape, block_size=(2,3), pooling_func='mean')
       >>> (PoolingOp * x.flatten()).reshape(PoolingOp.output_shape)
       array([[ 4.,  7.],
              [16., 19.]])
       >>> PoolingOp.adjoint(PoolingOp * x.flatten()).reshape(x.shape)
       array([[ 4.,  4.,  4.,  7.,  7.,  7.],
              [ 4.,  4.,  4.,  7.,  7.,  7.],
              [16., 16., 16., 19., 19., 19.],
              [16., 16., 16., 19., 19., 19.]])
       >>> PoolingOp = Pooling(shape=x.shape, block_size=(2,3), pooling_func='sum')
       >>> (PoolingOp * x.flatten()).reshape(PoolingOp.output_shape)
       array([[ 24,  42],
              [ 96, 114]])
       >>> PoolingOp.adjoint(PoolingOp * x.flatten()).reshape(x.shape)
       array([[ 24,  24,  24,  42,  42,  42],
              [ 24,  24,  24,  42,  42,  42],
              [ 96,  96,  96, 114, 114, 114],
              [ 96,  96,  96, 114, 114, 114]])

    .. plot::

       import numpy as np
       from pycsou.linop.sampling import Pooling
       import matplotlib.pyplot as plt
       import scipy.misc
       img = scipy.misc.face(gray=True).astype(float)
       PoolingOp = Pooling(shape=img.shape, block_size=(10,20))
       pooled_img = (PoolingOp * img.flatten()).reshape(PoolingOp.output_shape)
       adjoint_img = PoolingOp.adjoint(pooled_img.flatten()).reshape(img.shape)
       plt.figure()
       plt.imshow(img)
       plt.colorbar()
       plt.title('Original')
       plt.figure()
       plt.imshow(pooled_img)
       plt.colorbar()
       plt.title('Mean Pooling')
       plt.figure()
       plt.imshow(adjoint_img)
       plt.colorbar()
       plt.title('Mean Pooling followed by Unpooling')

    Notes
    -----
    Pooling is performed via the function `skimage.measure.block_reduce <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce>`_
    from ``scikit-image``. If one dimension of the image is not perfectly divisible by the block size then it is zero padded.

    The adjoint (*unpooling*) is performed by assigning the value of the blocks through the pooling function (e.g. mean, sum) to each element
    of the blocks.

    Warnings
    --------
    Max, median or min pooling are not supported since the resulting `PoolingOperator` would then be non linear!

    See Also
    --------
    :py:class:`~pycsou.linop.sampling.SubSampling`, :py:class:`~pycsou.linop.sampling.Downsampling`
    """

    def __init__(self, shape: tuple, block_size: Union[tuple, list], pooling_func: str = 'mean',
                 dtype: type = np.float64):
        """

        Parameters
        ----------
        shape : tuple
            Shape of the input array.
        block_size : Union[tuple, list]
            Shape of the sub-blocks on which pooling is performed.
        pooling_func : str
            Specifies if the local blocks should be summed (`pooling_func='sum'`) or averaged (`pooling_func='mean'`).
        dtype :
            Type of input array.

        Raises
        ------
        ValueError
            If the block size is inconsistent with the input array shape or if the pooling function is not supported.

        """
        if len(shape) != len(block_size):
            raise ValueError(f'Inconsistent block size {block_size} for array of shape {shape}.')
        if pooling_func not in ['mean', 'sum', 'average']:
            raise ValueError(f'pooling_func must be one of: "mean" or "sum".')
        self.input_shape = shape
        self.block_size = block_size
        if pooling_func == 'mean':
            self.pooling_func = np.mean
            self.pooling_func_kwargs = None
        elif pooling_func == 'sum':
            self.pooling_func = np.sum
            self.pooling_func_kwargs = None
        else:
            self.pooling_func = None
            self.pooling_func_kwargs = None
        self.output_shape = self.get_output_shape()
        self.output_size = int(np.prod(self.output_shape).astype(int))
        self.input_size = int(np.prod(self.input_shape).astype(int))
        super(Pooling, self).__init__(shape=(self.output_size, self.input_size), dtype=dtype)

    def get_output_shape(self) -> tuple:
        """
        Get shape of the pooled array.

        Returns
        -------
        tuple
            Output array shape.
        """
        x = np.zeros(shape=self.input_shape, dtype=np.float)
        y = block_reduce(image=x, block_size=self.block_size, func=self.pooling_func, cval=0,
                         func_kwargs=self.pooling_func_kwargs)
        return y.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return block_reduce(image=x.reshape(self.input_shape), block_size=self.block_size, func=self.pooling_func,
                            func_kwargs=self.pooling_func_kwargs).reshape(-1)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        x = np.copy(y.reshape(self.output_shape))
        for ax in range(len(self.input_shape)):
            x = np.repeat(x, self.block_size[ax], axis=ax)
            x = np.swapaxes(x, 0, ax)
            x = x[:self.input_shape[ax], ...]
            x = np.swapaxes(x, 0, ax)
        return x.reshape(-1)


class NNSampling(LinearOperator):
    r"""
    Nearest neighbours sampling operator.

    Sample a gridded ND signal at on-grid nearest neighbours of off-grid sampling locations. This can be useful when piecewise
    constant priors are used to recover continuously-defined signals sampled non-uniformly (see [FuncSphere]_ Remark 6.9).

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import NNSampling
       rng = np.random.default_rng(seed=0)

    .. doctest::

       >>> x = np.arange(24).reshape(4,6)
       >>> grid = np.stack(np.meshgrid(np.arange(6),np.arange(4)), axis=-1)
       >>> samples = np.stack((5 * rng.random(size=6),3 * rng.random(size=6)), axis=-1)
       >>> print(samples)
       [[3.18480844 1.81990733]
        [1.34893357 2.18848968]
        [0.20486762 1.63087497]
        [0.08263818 2.80521727]
        [4.0663512  2.44756066]
        [4.56377789 0.0082155 ]]
       >>> NNSamplingOp = NNSampling(samples=samples, grid=grid)
       >>> (NNSamplingOp * x.flatten())
       array([15, 13, 12, 18, 16,  5])
       >>> NNSamplingOp.adjoint(NNSamplingOp * x.flatten()).reshape(x.shape)
       array([[ 0.,  0.,  0.,  0.,  0.,  5.],
              [ 0.,  0.,  0.,  0.,  0.,  0.],
              [12., 13.,  0., 15., 16.,  0.],
              [18.,  0.,  0.,  0.,  0.,  0.]])

    .. plot::

       import numpy as np
       from pycsou.linop.sampling import NNSampling
       import matplotlib.pyplot as plt
       rng = np.random.default_rng(seed=0)

       rng = np.random.default_rng(seed=0)
       x = np.arange(24).reshape(4, 6)
       grid = np.stack(np.meshgrid(np.arange(6), np.arange(4)), axis=-1)
       samples = np.stack((5 * rng.random(size=12), 3 * rng.random(size=12)), axis=-1)
       NNSamplingOp = NNSampling(samples=samples, grid=grid)
       grid = grid.reshape(-1, 2)
       x = x.reshape(-1)
       y = (NNSamplingOp * x.flatten())
       x_samp = NNSamplingOp.adjoint(y).reshape(x.shape)
       plt.scatter(grid[..., 0].reshape(-1), grid[..., 1].reshape(-1), s=64, c=x.reshape(-1), marker='s', vmin=np.min(x),
            vmax=np.max(x))
       plt.scatter(samples[:, 0], samples[:, 1], c='r', s=64)
       plt.plot(np.stack((grid[NNSamplingOp.nn_indices, 0], samples[:, 0]), axis=0),
         np.stack((grid[NNSamplingOp.nn_indices, 1], samples[:, 1]), axis=0), '--r')
       plt.colorbar()
       plt.title('Nearest-neighbours Sampling')
       plt.figure()
       plt.scatter(grid[..., 0].reshape(-1), grid[..., 1].reshape(-1), s=64, c=x_samp.reshape(-1), marker='s',
            vmin=np.min(x),
            vmax=np.max(x))
       plt.scatter(samples[:, 0], samples[:, 1], c='r', s=64)
       plt.plot(np.stack((grid[NNSamplingOp.nn_indices, 0], samples[:, 0]), axis=0),
         np.stack((grid[NNSamplingOp.nn_indices, 1], samples[:, 1]), axis=0), '--r')
       plt.colorbar()
       plt.title('Sampling followed by adjoint')

    Notes
    -----
    Consider a signal defined over a mesh :math:`f:\{\mathbf{n}_1,\ldots, \mathbf{n}_M\}\to \mathbb{C}`, with
    :math:`\{\mathbf{n}_1,\ldots, \mathbf{n}_M\}\subset \mathbb{R}^N`. Consider moreover sampling locations
    :math:`\{\mathbf{z}_1,\ldots, \mathbf{z}_L\}\subset \mathbb{R}^N` which do not necessarily lie on the mesh.
    Then, nearest-neighbours sampling is defined as:

    .. math::

       y_i=f\left[\arg\min_{k=1,\ldots, M}\|\mathbf{z}_i-\mathbf{n}_k\|_2\right], \qquad i=1,\ldots, L.

    Note that in practice every sample locations has exactly *one* nearest neighbour (ties have probability zero) and hence
    this equation is well-defined.

    Given a vector :math:`\mathbf{y}=[y_1,\ldots, y_L]\in\mathbb{C}^N`, the adjoint of the ``NNSampling`` operator is defined as

    .. math::

       f[\mathbf{n}_k]=\mbox{mean}\left\{y_i :\; \mathbf{n}_k=\arg\min_{j=1,\ldots,M}\|\mathbf{z}_i-\mathbf{n}_j\|_2,\, i=1,\ldots, L\right\},\quad k=1,\ldots, M,

    where :math:`\mbox{mean}\{B\}=|B|^{-1}\sum_{z\in B} z,\; \forall B\subset\mathbb{C}` and with the convention that :math:`\mbox{mean}\{\emptyset\}=0.`
    The mean is used to handle cases where many sampling locations are mapped to a common nearest neighbour on the mesh.

    Warnings
    --------
    The grid needs not be uniform! Think of it as a mesh. It can happen that more than one sampling location is mapped to
    the same nearest neighbour.

    See Also
    --------
    :py:class:`~pycsou.linop.sampling.ContinuousSampling`
    """

    def __init__(self, samples: np.ndarray, grid: np.ndarray, dtype: type = np.float64):
        """

        Parameters
        ----------
        samples : np.ndarray
            Off-grid sampling locations with shape (M,N).
        grid :
            Grid points with shape (L,N).
        dtype : type
            Type of the input array.

        Raises
        ------
        ValueError
            If the dimension of the sample locations and the grid points do not match.
        """
        if samples.shape[-1] != grid.shape[-1]:
            raise ValueError(
                f'The samples have dimension {samples.shape[-1]} but the grid has dimension {grid.shape[-1]}.')
        self.samples = samples
        self.grid = grid.reshape(-1, grid.shape[-1])
        self.input_shape = self.grid.shape[:-1]
        self.input_size = int(np.prod(self.input_shape).astype(int))
        self.compute_nn()
        super(NNSampling, self).__init__(shape=(self.samples.size, self.input_size), dtype=dtype)

    def compute_nn(self):
        """
        Compute the on-grid nearest neighbours to the sampling locations.
        """
        self.grid_tree = cKDTree(data=self.grid, compact_nodes=False, balanced_tree=False)
        self.nn_distances, self.nn_indices = self.grid_tree.query(self.samples, k=1, p=2, n_jobs=-1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x[self.nn_indices]
        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        y_series = pd.Series(data=y, index=self.nn_indices)
        y_series = y_series.groupby(by=y_series.index).mean() # Average the samples associated to a common nearest neighbour.
        y = y_series.loc[self.nn_indices].values
        x = np.zeros(shape=self.input_size, dtype=self.dtype)
        x[self.nn_indices] = y
        return x


class ContinuousSampling(LinearOperator):
    pass


if __name__ == '__main__':
    pass
