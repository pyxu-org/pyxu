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
import dask.array as da
import scipy.sparse as sparse
import pylops
from typing import Optional, Union, List, Callable, Tuple
from pycsou.core.linop import LinearOperator
from pycsou.linop.base import PyLopLinearOperator, ExplicitLinearOperator, DenseLinearOperator
from skimage.measure import block_reduce
from scipy.spatial import cKDTree
import joblib as job


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
    :py:class:`~pycsou.linop.base.PyLopLinearOperator`
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

    .. doctest::

       >>> rng = np.random.default_rng(seed=0)
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
    :py:class:`~pycsou.linop.sampling.GeneralisedVandermonde`
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
        y_series = y_series.groupby(
            by=y_series.index).mean()  # Average the samples associated to a common nearest neighbour.
        y = y_series.loc[self.nn_indices].values
        x = np.zeros(shape=self.input_size, dtype=self.dtype)
        x[self.nn_indices] = y
        return x


class GeneralisedVandermonde(DenseLinearOperator):
    r"""
    Generalised Vandermonde matrix.

    Given sampling locations :math:`\{\mathbf{z}_1,\ldots,\mathbf{z}_L\}\subset\mathbb{R}^N`, and a family of `continuous` functions :math:`\{\varphi_1, \ldots, \varphi_K\}\subset \mathcal{C}(\mathbb{R}^N, \mathbb{C})`,
    this function forms the generalised Vandermonde matrix:

    .. math::

       \left[\begin{array}{ccc} \varphi_1(\mathbf{z}_1) & \cdots & \varphi_K(\mathbf{z}_1)\\\vdots & \ddots & \vdots\\\varphi_1(\mathbf{z}_L) & \cdots & \varphi_K(\mathbf{z}_L)\end{array}\right]\in\mathbb{C}^{L\times K}.

    This matrix is useful for sampling functions in the span of :math:`\{\varphi_1, \ldots, \varphi_K\}`. Indeed, if :math:`f=\sum_{k=1}^K\alpha_k\varphi_k`, then we have

    .. math::

       \left[\begin{array}{c}f(\mathbf{z}_1)\\\vdots\\ f(\mathbf{z}_L) \end{array}\right]=\left[\begin{array}{ccc} \varphi_1(\mathbf{z}_1) & \cdots & \varphi_K(\mathbf{z}_1)\\\vdots & \ddots & \vdots\\\varphi_1(\mathbf{z}_L) & \cdots & \varphi_K(\mathbf{z}_L)\end{array}\right]\left[\begin{array}{c}\alpha_1\\\vdots\\ \alpha_K \end{array}\right].

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import GeneralisedVandermonde

    .. doctest::

       >>> samples = np.arange(10)
       >>> func1 = lambda t: t**2;  func2 = lambda t: t**3; funcs = [func1, func2]
       >>> VOp = GeneralisedVandermonde(funcs=funcs, samples=samples)
       >>> alpha=np.ones((2,))
       >>> VOp.mat
       array([[  0.,   0.],
              [  1.,   1.],
              [  4.,   8.],
              [  9.,  27.],
              [ 16.,  64.],
              [ 25., 125.],
              [ 36., 216.],
              [ 49., 343.],
              [ 64., 512.],
              [ 81., 729.]])
       >>> np.allclose(VOp * alpha, samples ** 2 + samples ** 3)
       True

    See Also
    --------
    :py:class:`~pycsou.linop.sampling.MappedDistanceMatrix`
    """

    def __init__(self, samples: np.ndarray, funcs: List[Callable], dtype: type = np.float64):
        """

        Parameters
        ----------
        samples : np.ndarray
            Sampling locations with shape (L,N).
        funcs : list
            List of functions.
        dtype : type
            Type of input array.
        """
        self.samples = samples.reshape(-1)
        self.funcs = list(funcs)
        gen_vandermonde_mat = self.get_generalised_vandermonde_matrix().astype(dtype)
        super(GeneralisedVandermonde, self).__init__(ndarray=gen_vandermonde_mat, is_symmetric=False)

    def _map_func(self, f: Callable) -> np.ndarray:
        return f(self.samples)

    def get_generalised_vandermonde_matrix(self) -> np.ndarray:
        """
        Construct the generalised Vandermonde matrix.

        Returns
        -------
        np.ndarray
            The generalised Vandermonde matrix.
        """
        return np.stack(list(map(self._map_func, self.funcs)), axis=-1)


class MappedDistanceMatrix(ExplicitLinearOperator):
    r"""
    Transformed Distance Matrix.

    Given two point sets :math:`\{\mathbf{z}_1,\ldots,\mathbf{z}_L\}\subset\mathbb{R}^N`, :math:`\{\mathbf{x}_1,\ldots,\mathbf{x}_K\}\subset\mathbb{R}^N`
    and a `continuous` function :math:`\varphi:\mathbb{R}_+\to\mathbb{C}`, this function forms the following matrix:

    .. math::

       \left[\begin{array}{ccc} \varphi(d(\mathbf{z}_1,\mathbf{x}_1)) & \cdots & \varphi(d(\mathbf{z}_1,\mathbf{x}_K))\\\vdots & \ddots & \vdots\\\varphi(d(\mathbf{z}_L,\mathbf{x}_1)) & \cdots & \varphi(d(\mathbf{z}_1,\mathbf{x}_K))\end{array}\right]\in\mathbb{C}^{L\times K},

    where :math:`d:\mathbb{R}^N\times \mathbb{R}^N\to \mathbb{R}` is a (signed) distance defined in one of the following two ways:

    * `Radial:` :math:`d(\mathbf{z},\mathbf{x})=\|\mathbf{z}-\mathbf{x}\|_p, \; \forall \mathbf{z},\mathbf{x}\in\mathbb{R}^N,` with :math:`p\in [1, +\infty]`.
    * `Zonal:` :math:`d(\mathbf{z},\mathbf{x})=\langle\mathbf{z},\mathbf{x}\rangle, \; \forall \mathbf{z},\mathbf{x}\in\mathbb{S}^{N-1}.`
      Note that in this case the two point sets must be on the hypersphere :math:`\mathbb{S}^{N-1}.`

    This matrix is useful for sampling sums of radial/zonal functions. Indeed, if :math:`f(\mathbf{z})=\sum_{k=1}^K\alpha_k\varphi(d(\mathbf{z},\mathbf{x}_k))`, then we have

    .. math::

       \left[\begin{array}{c}f(\mathbf{z}_1)\\\vdots\\ f(\mathbf{z}_L) \end{array}\right]=\left[\begin{array}{ccc} \varphi(d(\mathbf{z}_1,\mathbf{x}_1)) & \cdots & \varphi(d(\mathbf{z}_1,\mathbf{x}_K))\\\vdots & \ddots & \vdots\\\varphi(d(\mathbf{z}_L,\mathbf{x}_1)) & \cdots & \varphi(d(\mathbf{z}_1,\mathbf{x}_K))\end{array}\right]\left[\begin{array}{c}\alpha_1\\\vdots\\ \alpha_K \end{array}\right].

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.sampling import MappedDistanceMatrix

    .. doctest::

       >>> rng = np.random.default_rng(seed=1)
       >>> sigma = 1 / 12
       >>> func = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))
       >>> max_distance = 3 * sigma
       >>> samples1 = np.linspace(0, 2, 10); samples2 = rng.random(size=3)
       >>> MDMOp1 = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func,max_distance=max_distance, operator_type='sparse', n_jobs=-1)
       >>> MDMOp2 = MappedDistanceMatrix(samples1=samples1, samples2=None, function=func,max_distance=max_distance, operator_type='sparse', n_jobs=-1)
       >>> MDMOp3 = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func, operator_type='dense')
       >>> MDMOp4 = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func, operator_type='dask')
       >>> type(MDMOp1.mat), type(MDMOp3.mat), type(MDMOp4.mat)
       (<class 'scipy.sparse.csr.csr_matrix'>, <class 'numpy.ndarray'>, <class 'dask.array.core.Array'>)
       >>> MDMOp1.mat.shape, MDMOp2.mat.shape
       ((10, 3), (10, 10))
       >>> MDMOp3.mat
       array([[6.43689834e-009, 5.64922608e-029, 2.23956473e-001],
              [2.38517543e-003, 2.61112241e-017, 6.44840995e-001],
              [7.21186665e-001, 9.84802612e-009, 1.51504434e-003],
              [1.77933914e-001, 3.03078296e-003, 2.90456923e-009],
              [3.58222996e-005, 7.61104282e-001, 4.54382719e-018],
              [5.88480228e-012, 1.55961419e-001, 5.80023464e-030],
              [7.88849171e-022, 2.60779757e-005, 6.04161441e-045],
              [8.62858845e-035, 3.55806811e-012, 5.13504354e-063],
              [7.70139186e-051, 3.96130538e-022, 3.56138512e-084],
              [5.60896038e-070, 3.59870356e-035, 2.01547509e-108]])
       >>> alpha = rng.lognormal(size=samples2.size, sigma=0.5)
       >>> beta = rng.lognormal(size=samples1.size, sigma=0.5)
       >>> MDMOp1 * alpha
       array([0.27995783, 0.8060865 , 0.37589858, 0.09274313, 1.19684992,
              0.24525208, 0.        , 0.        , 0.        , 0.        ])
       >>> MDMOp2 * beta
       array([0.80274037, 1.39329178, 1.27124579, 1.22168244, 1.08494933,
              1.36310926, 0.7558366 , 0.96398702, 0.85066284, 1.37152693])
       >>> MDMOp3 * alpha
       array([2.79957838e-01, 8.07329704e-01, 3.77792488e-01, 9.75090904e-02,
              1.19686859e+00, 2.45252084e-01, 4.10080770e-05, 5.59512490e-12,
              6.22922262e-22, 5.65902486e-35])
       >>> MDMOp4 * alpha
       array([2.79957838e-01, 8.07329704e-01, 3.77792488e-01, 9.75090904e-02,
              1.19686859e+00, 2.45252084e-01, 4.10080770e-05, 5.59512490e-12,
              6.22922262e-22, 5.65902486e-35])
       >>> np.allclose(MDMOp3 * alpha, MDMOp4 * alpha)
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.sampling import MappedDistanceMatrix

       t = np.linspace(0, 2, 256)
       rng = np.random.default_rng(seed=2)
       x,y = np.meshgrid(t,t)
       samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
       samples2 = np.stack((2 * rng.random(size=4), 2 * rng.random(size=4)), axis=-1)
       alpha = np.ones(samples2.shape[0])
       sigma = 1 / 12
       func = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))
       MDMOp = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func, operator_type='dask')
       plt.contourf(x,y,(MDMOp * alpha).reshape(t.size, t.size), 50)
       plt.title('Sum of 4 (radial) Gaussians')
       plt.colorbar()
       plt.xlabel('$x$')
       plt.ylabel('$y$')

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.sampling import MappedDistanceMatrix

       rng = np.random.default_rng(seed=2)
       phi = np.linspace(0,2*np.pi, 256)
       theta = np.linspace(-np.pi/2,np.pi/2, 256)
       phi, theta = np.meshgrid(phi, theta)
       x,y,z = np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)
       samples1 = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
       phi2 = 2 * np.pi * rng.random(size=4)
       theta2 = np.pi * rng.random(size=4) - np.pi/2
       x2,y2,z2 = np.cos(phi2)*np.cos(theta2), np.sin(phi2)*np.cos(theta2), np.sin(theta2)
       samples2 = np.stack((x2,y2,z2), axis=-1)
       alpha = np.ones(samples2.shape[0])
       sigma = 1 / 9
       func = lambda x: np.exp(-np.abs(1-x) / (sigma ** 2))
       MDMOp = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func,mode='zonal', operator_type='sparse', max_distance=3*sigma)
       plt.contourf(phi, theta, (MDMOp * alpha).reshape(phi.shape), 50)
       plt.title('Sum of 4 (zonal) wrapped Gaussians')
       plt.colorbar()
       plt.xlabel('$\\phi$ (radians)')
       plt.ylabel('$\\theta$ (radians)')

    See Also
    --------
    :py:class:`~pycsou.linop.sampling.GeneralisedVandermonde`
    """

    def __init__(self, samples1: np.ndarray, function: Callable, samples2: Optional[np.ndarray] = None,
                 mode: str = 'radial', max_distance: Optional[np.float] = None, dtype: type = np.float64,
                 chunks: Union[str, int, tuple, None] = 'auto', operator_type: str = 'dask', verbose: bool = True,
                 n_jobs: int = -1, joblib_backend: str = 'loky', ord: float = 2., eps: float = 0):
        r"""

        Parameters
        ----------
        samples1 : np.ndarray
            First point set with shape (L,N).
        function : Callable
            Function :math:`\varphi: \mathbb{R}_+\to \mathbb{C}` to apply to each entry of the distance matrix.
        samples2 : Optional[np.ndarray]
            Optional second point set with shape (K,N). If ``None``, ``samples2`` is equal to ``samples1`` and the operator symmetric.
        mode : str
            How to compute the distances. If ``'radial'``, the Euclidean distance with respect to the Minkowski norm :math:`\|\cdot\|_p` is used. :math:`p` is specified via the parameter ``ord``  and has to meet the condition `1 <= p <= infinity`. If ``'zonal'`` the spherical geodesic distance is used.
        max_distance : Optional[np.float]
            Support of the function :math:`\varphi`. Must be specified for sparse representation.
        dtype : type
            Type of input array.
        chunks : Union[str, int, tuple, None]
            Chunk sizes for Dask representation.
        operator_type : str
            Internal representation of the operator: ``'dense'`` represents the operator as a Numpy array, ``'dask'`` as a `Dask array <https://docs.dask.org/en/latest/array.html>`_,
            ``'sparse'`` as a `sparse matrix <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_. Dask arrays or sparse matrices can be more memory efficient for very large point sets.
        verbose : bool
            Verbosity for parallel computations (only for sparse representations).
        n_jobs : int
            Number of cores for parallel computations (only for sparse representations). ``n_jobs=-1`` uses all available cores.
        joblib_backend : str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).
        ord: float
            Order of the norm for ``mode='radial'``. ``ord`` must satisfy `1<= ord <=infty`.
        eps: float
            Parameter for approximate nearest neighbours search when ``mode='sparse'``.

        Raises
        ------
        ValueError
            If ``mode`` is invalid or if ``operator_type`` is ``'sparse'`` but ``max_distance=None``.

        """

        if mode not in ['radial', 'zonal']:
            raise ValueError('Supported modes are "radial" or "zonal".')
        if (operator_type is 'sparse') and (max_distance is None):
            raise ValueError('Specify a maximal distance for sparse format.')
        self.max_distance = max_distance
        self.mode = mode
        self.function = function
        self.samples1 = samples1 if samples1.ndim > 1 else samples1[:, None]
        if samples2 is None:
            self.is_symmetric = True
            self.samples2 = samples1 if samples1.ndim > 1 else samples1[:, None]
        else:
            self.is_symmetric = False
            self.samples2 = samples2 if samples2.ndim > 1 else samples2[:, None]
        self.dtype = dtype
        self.chunks = chunks
        self.operator_type = operator_type
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend
        self.ord = ord
        self.eps = eps
        if self.operator_type is 'sparse':
            mapped_distance_matrix = self.get_sparse_mdm()
        elif self.operator_type is 'dask':
            mapped_distance_matrix = self.get_dask_mdm()
        elif self.operator_type is 'dense':
            mapped_distance_matrix = self.get_dense_mdm()
        else:
            raise ValueError(f'Unsupported operator type {self.operator_type}.')
        super(MappedDistanceMatrix, self).__init__(array=mapped_distance_matrix, is_symmetric=self.is_symmetric)

    def get_sparse_mdm(self) -> sparse.csr_matrix:
        r"""
        Form the Mapped Distance Matrix as a sparse matrix.

        Returns
        -------
        sparse.csr_matrix
            Sparse Mapped Distance Matrix.
        """
        samples_tree1 = cKDTree(data=self.samples1, compact_nodes=False, balanced_tree=False)
        if self.is_symmetric is False:
            samples_tree2 = cKDTree(data=self.samples2, compact_nodes=False, balanced_tree=False)
        else:
            samples_tree2 = samples_tree1
        if self.samples2.shape[0] < self.samples1.shape[0]:
            mapped_distance_matrix = self._sparse_mdm(big_tree=samples_tree1, small_tree=samples_tree2, iter_over='col')
        else:
            mapped_distance_matrix = self._sparse_mdm(big_tree=samples_tree2, small_tree=samples_tree1, iter_over='row')
        return mapped_distance_matrix

    def _sparse_mdm(self, big_tree: cKDTree, small_tree: cKDTree, iter_over: str) -> sparse.csr_matrix:
        neighbours = small_tree.query_ball_tree(big_tree, self.max_distance, p=self.ord, eps=self.eps)
        with job.Parallel(backend=self.joblib_backend, n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            results = parallel(job.delayed(self._apply_function)
                               (small_tree.data[i], big_tree.data[neighbours[i]], i, neighbours[i], iter_over)
                               for i in range(small_tree.data.shape[0]))
        coo_data = np.concatenate([results[i][0] for i in range(len(results))])
        coo_i = np.concatenate([results[i][1] for i in range(len(results))])
        coo_j = np.concatenate([results[i][2] for i in range(len(results))])
        sparse_mdm = sparse.coo_matrix((coo_data, (coo_i, coo_j)),
                                       shape=(self.samples1.shape[0], self.samples2.shape[0]),
                                       dtype=self.dtype).tocsr()
        return sparse_mdm

    def _apply_function(self, coord_center: np.ndarray, coord_neighbours: np.ndarray, index_center: int,
                        index_neighbours: list, iter_over: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == 'radial':
            distances = np.linalg.norm(coord_center[None, :] - coord_neighbours, axis=-1, ord=self.ord)
        else:
            distances = np.clip(np.sum(coord_center[None, :] * coord_neighbours, axis=-1), a_min=-1, a_max=1)
        values = self.function(distances)
        if iter_over == 'row':
            return values, np.repeat(index_center, len(index_neighbours)), np.array(index_neighbours)
        elif iter_over == 'col':
            return values, np.array(index_neighbours), np.repeat(index_center, len(index_neighbours))

    def get_dask_mdm(self) -> da.core.Array:
        r"""
        Form the Mapped Distance Matrix as a dask array.

        Returns
        -------
        da.core.Array
            Mapped Distance Matrix as a Dask array.
        """
        samples_da = da.from_array(self.samples1[:, None, :], chunks=self.chunks)
        knots_da = da.from_array(self.samples2[None, ...], chunks=self.chunks)
        if self.mode == 'radial':
            distances_da = da.linalg.norm(samples_da - knots_da, axis=-1, ord=self.ord)
        elif self.mode == 'zonal':
            distances_da = da.clip(da.sum(samples_da * knots_da, axis=-1), -1, 1)
        else:
            raise ValueError(f'Unsupported mode {self.mode}.')
        mapped_distance_matrix = da.map_blocks(self.function, distances_da, dtype=self.dtype)
        return mapped_distance_matrix

    def get_dense_mdm(self) -> np.ndarray:
        r"""
        Form the Mapped Distance Matrix as a dense Numpy array.

        Returns
        -------
        np.ndarray
            Mapped Distance Matrix as a Numpy array.
        """
        if self.mode == 'radial':
            distances = np.linalg.norm(self.samples1[:, None, :] - self.samples2[None, ...], axis=-1, ord=self.ord)
        elif self.mode == 'zonal':
            distances = np.clip(np.sum(self.samples1[:, None, :] * self.samples2[None, ...], axis=-1), a_min=-1,
                                a_max=1)
        else:
            raise ValueError(f'Unsupported mode {self.mode}.')
        mapped_distance_matrix = self.function(distances)
        return mapped_distance_matrix.astype(self.dtype)


if __name__ == '__main__':
    pass
