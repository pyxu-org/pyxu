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


class DownSampling(LinearOperator):
    pass


class Pooling(LinearOperator):
    pass


class OffGridSampling(LinearOperator):
    pass

class ContinuousSampling(LinearOperator):
    pass
