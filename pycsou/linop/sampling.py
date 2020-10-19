# masking, binning, sampling, downsampling
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
import pylops
import pygsp
from typing import Optional, Union, Tuple, Iterable, List
from pycsou.core.linop import PyLopLinearOperator, LinearOperator, IdentityOperator, DiagonalOperator, LinOpVStack, \
    SparseLinearOperator, PolynomialLinearOperator
from numbers import Number


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
        self.size = size
        self.nb_of_samples = self.sampling_bool[self.sampling_bool == True].size
        if self.sampling_bool.size != size:
            raise ValueError('Invalid size of boolean sampling array.')
        super(Masking, self).__init__(shape=(self.nb_of_samples, self.size), dtype=dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[self.sampling_bool]

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        x = np.zeros(shape=self.size, dtype=self.dtype)
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
       plt.figure()
       plt.imshow(down_sampled_img)
       plt.figure()
       plt.imshow(up_sampled_img)

    Notes
    -----
    Downsampling by :math:`M` an input vector :math:`\mathbf{x}` of size
    :math:`N` can be performed as:

    .. math::

        y_i = x_{n}  \quad  n=iM, \,i=1,\ldots, \lfloor N/M \rfloor.

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
            If the set of parameters {`shape`, `size`, `sampling_factor`, `axis`} is invalid.
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
        self.size = size
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

        super(DownSampling, self).__init__(size=self.size, sampling_bool=self.downsampling_mask, dtype=dtype)

    def compute_downsampling_mask(self) -> np.ndarray:
        """
        Compute the downsampling mask.

        Returns
        -------
        :py:class:`np.ndarray`
            The mask to apply to get the downsampled values.
        """
        if self.input_shape is None:
            indices = np.arange(self.size)
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
    pass


class NNSampling(LinearOperator):
    pass


class ContinuousSampling(LinearOperator):
    pass


if __name__ == '__main__':
    import numpy as np
    from pycsou.linop.sampling import DownSampling
    import matplotlib.pyplot as plt
    import scipy.misc

    img = scipy.misc.face(gray=True).astype(float)
    DownSampOp = DownSampling(size=img.size, shape=img.shape, downsampling_factor=(5, 9))
    down_sampled_img = (DownSampOp * img.flatten()).reshape(DownSampOp.output_shape)
    up_sampled_img = DownSampOp.adjoint(down_sampled_img.flatten()).reshape(img.shape)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(down_sampled_img)
    plt.figure()
    plt.imshow(up_sampled_img)
