# #############################################################################
# linop.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Abstract classes for multidimensional linear maps.
"""

from pycsou.core.map import DifferentiableMap, DiffMapSum, DiffMapComp, Map, MapSum, MapComp, DiffMapStack
import numpy as np
from typing import Union, Tuple, Optional, Iterable
from abc import abstractmethod
from numbers import Number
import pylops
from pylops.optimization.leastsquares import NormalEquationsInversion
import scipy.sparse.linalg as spls
import scipy.sparse as sparse
import dask.array as da


class LinearOperator(DifferentiableMap):
    r"""
    Base class for linear operators.

    Any instance/subclass of this class must at least implement the abstract methods ``__call__`` and ``adjoint``.

    Notes
    -----
    This class supports the following arithmetic operators ``+``, ``-``, ``*``, ``@``, ``**`` and ``/``, implemented with the
    class methods ``__add__``/``__radd__``, ``__sub__``/``__neg__``, ``__mul__``/``__rmul__``, ``__matmul__``, ``__pow__``, ``__truediv__``.
    Such arithmetic operators can be used to *add*, *substract*, *scale*, *compose*, *exponentiate* or *evaluate* ``LinearOperator`` instances.

    """

    def __init__(self, shape: Tuple[int, int], dtype: Optional[type] = None,
                 is_explicit: bool = False, is_dense: bool = False, is_sparse: bool = False, is_dask: bool = False,
                 is_symmetric: bool = False, lipschitz_cst: float = np.infty):
        r"""
        Parameters
        ----------
        shape: Tuple[int, int]
            Shape of the linear operator.
        dtype: Optional[type]
            Data type of the linear operator.
        is_explicit: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Numpy/Scipy/Dask array.
        is_dense: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Numpy array.
        is_sparse: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Scipy sparse matrix.
        is_dask: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Dask array.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        lipschitz_cst: float
            Lipschitz constant (maximal singular value) of the linear operator if known. Default to :math:`+\infty`.
        """
        DifferentiableMap.__init__(self, shape=shape, is_linear=True, lipschitz_cst=lipschitz_cst,
                                   diff_lipschitz_cst=lipschitz_cst)
        self.dtype = dtype
        self.is_explicit = is_explicit
        self.is_dense = is_dense
        self.is_sparse = is_sparse
        self.is_dask = is_dask
        self.is_symmetric = is_symmetric
        self.is_square = True if shape[0] == shape[1] else False

    def matvec(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        r"""Alias for ``self.__call__`` to comply with Scipy's interface."""
        return self.__call__(x)

    @abstractmethod
    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        r"""
        Evaluates the adjoint of the operator at a point.

        Parameters
        ----------
        y: Union[Number, np.ndarray]
            Point at which the adjoint should be evaluated.

        Returns
        -------
        Union[Number, np.ndarray]
            Evaluation of the adjoint at ``y``.
        """
        pass

    def jacobianT(self, arg: Union[Number, np.ndarray, None] = None) -> 'LinearOperator':
        return self.get_adjointOp()

    def get_adjointOp(self) -> Union['LinearOperator', 'AdjointLinearOperator']:
        r"""
        Return the adjoint operator as a ``LinearOperator`` instance.

        Returns
        -------
        Union['LinearOperator', 'AdjointLinearOperator']
            The adjoint operator.
        """
        if self.is_symmetric is True:
            return self
        else:
            return AdjointLinearOperator(self)

    @property
    def H(self):
        r"""Alias for ``self.get_adjointOp``."""
        return self.get_adjointOp()

    @property
    def RangeGram(self):
        r"""
        Range-Gram operator, obtained by the composition ``self * self.H`` (adjoint followed by operator).

        Returns
        -------
        :py:class:`~pycsou.core.linop.SymmetricLinearOperator`
            The Range-Gram operator.
        """
        return SymmetricLinearOperator(self * self.H)

    @property
    def DomainGram(self):
        r"""
        Domain-Gram operator, obtained by the composition ``self.H * self`` (operator followed by adjoint).

        Returns
        -------
        :py:class:`~pycsou.core.linop.SymmetricLinearOperator`
            The Domain-Gram operator.
        """
        return SymmetricLinearOperator(self.H * self)

    def eigenvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        r"""
        Find ``k`` eigenvalues of a square operator.

        Parameters
        ----------
        k: int
            The number of eigenvalues and eigenvectors desired. ``k`` must be strictly smaller than the dimension of the operator.
            It is not possible to compute all eigenvectors of a matrix.
        which: str, [‘LM’ | ‘SM’ | ‘LR’ | ‘SR’ | ‘LI’ | ‘SI’]
            Which ``k`` eigenvalues to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude
                * ‘LR’ : largest real part
                * ‘SR’ : smallest real part
                * ‘LI’ : largest imaginary part
                * ‘SI’ : smallest imaginary part

        kwargs: dict
            A dict of additional keyword arguments values accepted by Scipy's functions :py:func:`scipy.sparse.linalg.eigs` and :py:func:`scipy.sparse.linalg.eigsh`.

        Returns
        -------
        np.ndarray
            Array containing the ``k`` requested eigenvalues.

        Raises
        ------
        NotImplementedError
            If the linear operator is not square.

        Notes
        -----
        This function calls one of the two Scipy's functions: :py:func:`scipy.sparse.linalg.eigs` and :py:func:`scipy.sparse.linalg.eigsh`.
        See the documentation of these two functions for more information on their behaviour and the underlying ARPACK functions they rely on.

        See Also
        --------
        :py:meth:`~pycsou.core.linop.LinearOperator.svds`
        """
        if self.is_symmetric is True:
            eigen_values = spls.eigsh(A=self.SciOp, k=k, which=which, return_eigenvectors=False, **kwargs)
        elif self.is_square is True:
            eigen_values = spls.eigs(A=self.SciOp, k=k, which=which, return_eigenvectors=False, **kwargs)
        else:
            raise NotImplementedError(
                'The function eigenvals is only for square linear operator. For non square linear operators, use the method singularvals.')
        return eigen_values

    def singularvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        r"""
        Compute the largest or smallest ``k`` singular values of an operator.
        The order of the singular values is not guaranteed.

        Parameters
        ----------
        k: int
            Number of singular values to compute. Must be ``1 <= k < min(self.shape)``.

        which: str, [‘LM’ | ‘SM’]
            Which ``k`` singular values to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude

        kwargs: dict
            A dict of additional keyword arguments values accepted by Scipy's function :py:func:`scipy.sparse.linalg.svds`.

        Returns
        -------
        np.ndarray
            Array containing the ``k`` requested singular values.

        Examples
        --------
        .. testsetup::

           import numpy as np
           from pycsou.linop.conv import Convolve1D
           from scipy import signal

        .. doctest::

           >>> sig = np.repeat([0., 1., 0.], 10)
           >>> filter = signal.hann(5); filter[filter.size//2:] = 0
           >>> ConvOp = Convolve1D(size=sig.size, filter=filter)
           >>> np.round((ConvOp.singularvals(k=3, which='LM', tol=1e-3)), 2)
           array([0.5, 0.5, 0.5])

        Notes
        -----
        This function calls the Scipy's function: :py:func:`scipy.sparse.linalg.svds`.
        See the documentation of this function for more information on its behaviour and the underlying ARPACK/LOBPCG functions it relies on.

        See Also
        --------
        :py:meth:`~pycsou.core.linop.LinearOperator.eigenvals`
        """
        return spls.svds(A=self.SciOp, k=k, which=which, return_singular_vectors=False, **kwargs)

    def compute_lipschitz_cst(self, **kwargs):
        r"""
        Compute the Lipschitz constant of the operator.

        Parameters
        ----------
        kwargs: dict
            A dict of additional keyword arguments values accepted by Scipy's functions :py:func:`scipy.sparse.linalg.eigs`,
            :py:func:`scipy.sparse.linalg.eigsh`, :py:func:`scipy.sparse.linalg.svds`.

        Returns
        -------
        None
            Nothing: The Lipschitz constant is stored in the attribute ``self.lipschitz_cst``.

        Examples
        --------

        .. doctest::

           >>> sig = np.repeat([0., 1., 0.], 10)
           >>> filter = signal.hann(5); filter[filter.size//2:] = 0
           >>> ConvOp = Convolve1D(size=sig.size, filter=filter)
           >>> ConvOp.compute_lipschitz_cst(tol=1e-2); np.round(ConvOp.lipschitz_cst,1)
           0.4

        Notes
        -----
        The Lipschtiz constant of a linear operator is its largest singular value / eigenvalue (in terms of magnitude). This function therefore calls
        the methods  ``self.singularvals`` or ``self.eigenvals`` with ``k=1`` and ``which='LM'`` to perform this computation.

        Warnings
        --------
        For high-dimensional linear operators this method can be very time-consuming. Reducing the computation accuracy with
        the optional argument ``tol: float`` may help reduce the computational burden. See Scipy's functions :py:func:`scipy.sparse.linalg.eigs`,
        :py:func:`scipy.sparse.linalg.eigsh`, :py:func:`scipy.sparse.linalg.svds` for more on this parameter.

        """
        if self.is_square is True:
            self.lipschitz_cst = float(np.abs(self.eigenvals(k=1, **kwargs)))
        else:
            self.lipschitz_cst = float(self.singularvals(k=1, **kwargs))
        self.diff_lipschitz_cst = self.lipschitz_cst

    def todense(self) -> 'DenseLinearOperator':
        r"""
        Convert the operator to a :py:class:`~pycsou.core.linop.DenseLinearOperator`.

        Returns
        -------
        :py:class:`~pycsou.core.linop.DenseLinearOperator`
            The dense linear operator representation.
        """
        return DenseLinearOperator(self.PyLop.todense())

    def tosparse(self) -> 'SparseLinearOperator':
        r"""
        Convert the operator to a :py:class:`~pycsou.core.linop.SparseLinearOperator`.

        Returns
        -------
        :py:class:`~pycsou.core.linop.SparseLinearOperator`
            The sparse linear operator representation.
        """
        return SparseLinearOperator(self.PyLop.tosparse())

    def tosciop(self) -> spls.LinearOperator:
        r"""
        Convert the operator to a Scipy :py:class:`scipy.sparse.linalg.LinearOperator`.

        Returns
        -------
        :py:class:`scipy.sparse.linalg.LinearOperator`
            The Scipy linear operator representation.
        """
        return spls.LinearOperator(dtype=self.dtype, shape=self.shape, matvec=self.matvec, rmatvec=self.adjoint)

    @property
    def SciOp(self):
        r"""Alias for method ``self.tosciop``."""
        return self.tosciop()

    def topylop(self) -> pylops.LinearOperator:
        r"""
        Convert the operator to a Pylops :py:class:`pylops.LinearOperator`.

        Returns
        -------
        :py:class:`pylops.LinearOperator`
            The Pylops linear operator representation.
        """
        return pylops.LinearOperator(Op=self.SciOp, explicit=self.is_explicit)

    @property
    def PyLop(self):
        r"""Alias for method ``self.topylop``."""
        return self.topylop()

    def cond(self, **kwargs) -> float:
        r"""
        Compute the condition number of the operator.

        Parameters
        ----------
        kwargs: dict
            A dict of additional keyword arguments values accepted by Pylops' method :py:meth:`pylops.LinearOperator.cond`.

        Returns
        -------
        float
            Condition number.
        """
        return self.PyLop.cond(**kwargs)

    def pinv(self, y: Union[Number, np.ndarray], eps: Number = 0, **kwargs) -> Union[Number, np.ndarray]:
        r"""
        Evaluate the pseudo-inverse of the operator at ``y``.

        Parameters
        ----------
        y: Union[Number, np.ndarray]
            Point at which the pseudo-inverse is evaluated.
        eps: Number
            Tikhonov damping.
        kwargs:
            Arbitrary keyword arguments accepted by the function: :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`.

        Returns
        -------
        numpy.ndarray
            Evaluation of the pseudo-inverse of the operator at ``y``.

        Notes
        -----
        This is a wrapper around the function :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`. Additional
        information can be found in the help of this function.
        """
        return NormalEquationsInversion(Op=self.PyLop, Regs=None, data=y, epsI=eps, **kwargs, returninfo=False)

    @property
    def PinvOp(self) -> 'LinOpPinv':
        r"""Return the pseudo-inverse of the operator as a ``LinearOperator``."""
        return LinOpPinv(self)

    @property
    def dagger(self) -> 'LinOpPinv':
        r"""Alias for ``self.PinvOp``."""
        return self.PinvOp

    def RowProjector(self):
        r"""Orthogonal projection operator onto the rows of ``self``. It is given by ``self * self.dagger``."""
        return SymmetricLinearOperator(self * self.dagger)

    def ColProjector(self):
        r"""Orthogonal projection operator onto the columns of ``self``. It is given by ``self.dagger * self``."""
        return SymmetricLinearOperator(self.dagger * self)

    def __add__(self, other: Union['Map', 'DifferentiableMap', 'LinearOperator', np.ndarray]) -> Union[
        'MapSum', 'DiffMapSum', 'LinOpSum']:
        if isinstance(other, LinearOperator):
            return LinOpSum(self, other)
        elif isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union['Map', 'DifferentiableMap', 'LinearOperator', Number, np.ndarray]) -> Union[
        'MapSum', 'DiffMapSum', 'LinOpSum', np.ndarray]:
        if isinstance(other, Number):
            other = HomothetyMap(constant=other, size=self.shape[1])

        if isinstance(other, np.ndarray):
            return self(other)
        elif isinstance(other, LinearOperator):
            return LinOpComp(self, other)
        elif isinstance(other, DifferentiableMap):
            return DiffMapComp(self, other)
        elif isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other: Union['Map', 'DifferentiableMap', 'LinearOperator', Number]) -> Union[
        'MapSum', 'DiffMapSum', 'LinOpSum']:
        if isinstance(other, Number):
            other = HomothetyMap(constant=other, size=self.shape[0])

        if isinstance(other, LinearOperator):
            return LinOpComp(other, self)
        elif isinstance(other, DifferentiableMap):
            return DiffMapComp(other, self)
        elif isinstance(other, Map):
            return MapComp(other, self)
        else:
            raise NotImplementedError


class AdjointLinearOperator(LinearOperator):
    def __init__(self, LinOp: LinearOperator):
        super(AdjointLinearOperator, self).__init__(shape=(LinOp.shape[1], LinOp.shape[0]), dtype=LinOp.dtype,
                                                    is_explicit=LinOp.is_explicit, is_dask=LinOp.is_dask,
                                                    is_dense=LinOp.is_dense, is_sparse=LinOp.is_sparse,
                                                    is_symmetric=LinOp.is_symmetric)
        self.Linop = LinOp

    def __call__(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Linop.adjoint(y)

    def adjoint(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Linop.matvec(x)

    def compute_lipschitz_cst(self, **kwargs: dict):
        if self.Linop.lipschitz_cst != np.infty:
            self.lipschitz_cst = self.Linop.lipschitz_cst
        else:
            LinearOperator.compute_lipschitz_cst(self, **kwargs)


class LinOpSum(LinearOperator, DiffMapSum):
    def __init__(self, LinOp1: LinearOperator, LinOp2: LinearOperator, dtype: Optional[type] = None):
        dtype = LinOp1.dtype if LinOp1.dtype is LinOp2.dtype else dtype
        DiffMapSum.__init__(self, map1=LinOp1, map2=LinOp2)
        LinearOperator.__init__(self, shape=self.shape, dtype=dtype,
                                is_explicit=LinOp1.is_explicit & LinOp2.is_explicit,
                                is_dask=LinOp1.is_dask & LinOp2.is_dask, is_dense=LinOp1.is_dense & LinOp2.is_dense,
                                is_sparse=LinOp1.is_sparse & LinOp2.is_sparse,
                                is_symmetric=LinOp1.is_symmetric & LinOp2.is_symmetric,
                                lipschitz_cst=self.lipschitz_cst)
        self.LinOp1, self.LinOp2 = LinOp1, LinOp2

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.LinOp1.adjoint(y) + self.LinOp2.adjoint(y)


class LinOpComp(LinearOperator, DiffMapComp):
    def __init__(self, LinOp1: LinearOperator, LinOp2: LinearOperator, dtype: Optional[type] = None):
        dtype = LinOp1.dtype if LinOp1.dtype is LinOp2.dtype else dtype
        DiffMapComp.__init__(self, map1=LinOp1, map2=LinOp2)
        LinearOperator.__init__(self, shape=self.shape, dtype=dtype,
                                is_explicit=LinOp1.is_explicit & LinOp2.is_explicit,
                                is_dask=LinOp1.is_dask & LinOp2.is_dask, is_dense=LinOp1.is_dense & LinOp2.is_dense,
                                is_sparse=LinOp1.is_sparse & LinOp2.is_sparse,
                                is_symmetric=LinOp1.is_symmetric & LinOp2.is_symmetric,
                                lipschitz_cst=self.lipschitz_cst)
        self.LinOp1, self.LinOp2 = LinOp1, LinOp2

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.LinOp2.adjoint(self.LinOp1.adjoint(y))


class SymmetricLinearOperator(LinearOperator):
    def __init__(self, LinOp: LinearOperator):
        if LinOp.shape[0] != LinOp.shape[1]:
            raise TypeError('The input linear operator is not symmetric.')
        super(SymmetricLinearOperator, self).__init__(shape=LinOp.shape, dtype=LinOp.dtype,
                                                      is_explicit=LinOp.is_explicit,
                                                      is_dask=LinOp.is_dask, is_dense=LinOp.is_dense,
                                                      is_sparse=LinOp.is_sparse, is_symmetric=True,
                                                      lipschitz_cst=LinOp.lipschitz_cst)
        self.LinOp = LinOp

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.LinOp.__call__(x)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.__call__(y)


class UnitaryOperator(LinearOperator):
    def __init__(self, size: int, dtype: Optional[type] = None,
                 is_explicit: bool = False, is_dense: bool = False, is_sparse: bool = False, is_dask: bool = False,
                 is_symmetric: bool = False):
        super(UnitaryOperator, self).__init__(shape=(size, size), dtype=dtype, is_explicit=is_explicit,
                                              is_dense=is_dense,
                                              is_sparse=is_sparse, is_dask=is_dask, is_symmetric=is_symmetric)
        self.size = size
        self.lipschitz_cst = self.diff_lipschitz_cst = 1

    @property
    def RangeGram(self):
        return IdentityOperator(size=self.size, dtype=self.dtype)

    @property
    def DomainGram(self):
        return IdentityOperator(size=self.size, dtype=self.dtype)

    def eigenvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        return self.singularvals(k=k, which='LM', **kwargs)

    def singularvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        if k > np.fmin(self.shape[0], self.shape[1]):
            raise ValueError('The number of singular values must not exceed the smallest dimension size.')
        return np.ones(shape=(k,))

    def compute_lipschitz_cst(self, **kwargs: dict):
        self.lipschitz_cst = self.diff_lipschitz_cst = 1

    def pinv(self, y: Union[Number, np.ndarray], eps: Number = 0, **kwargs) -> Union[Number, np.ndarray]:
        return self.adjoint(y)

    @property
    def PinvOp(self):
        return self.H

    def cond(self, **kwargs):
        return 1


class LinOpPinv(LinearOperator):
    def __init__(self, LinOp: LinearOperator, eps: Number = 0):
        self.LinOp = LinOp
        self.eps = eps
        super(LinOpPinv, self).__init__(shape=LinOp.H.shape, dtype=LinOp.dtype, is_explicit=False, is_dense=False,
                                        is_dask=False, is_symmetric=LinOp.is_symmetric)

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return NormalEquationsInversion(Op=self.PyLop, Regs=None, data=x, epsI=self.eps, returninfo=False)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return LinOpPinv(LinOp=self.LinOp.H, eps=self.eps).__call__(y)


class PyLopLinearOperator(LinearOperator):
    r"""
    Construct a linear operator from a :py:class:`pylops.LinearOperator` instance.
    """

    def __init__(self, PyLop: pylops.LinearOperator, is_symmetric: bool = False, is_dense: bool = False,
                 is_sparse: bool = False):
        r"""
        Parameters
        ----------
        PyLop: pylops.LinearOperator
            Pylops linear operator.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        is_dense: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Numpy array.
        is_sparse: bool
            If  ``True``, the linear operator is specified explicitly in terms of a Scipy sparse matrix.
        """
        super(PyLopLinearOperator, self).__init__(shape=PyLop.shape, dtype=PyLop.dtype, is_explicit=PyLop.explicit,
                                                  is_dense=is_dense, is_sparse=is_sparse, is_dask=False,
                                                  is_symmetric=is_symmetric)
        self.Op = PyLop

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Op.matvec(x)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Op.rmatvec(y)


class ExplicitLinearOperator(LinearOperator):
    r"""
    Construct an explicit linear operator.

    Explicit operators can be built from a Numpy array/Scipy sparse matrix/Dask array.
    The array is stored in the attribute ``self.mat``.
    """

    def __init__(self, array: Union[np.ndarray, sparse.spmatrix, da.core.Array], is_symmetric: bool = False):
        r"""
        Parameters
        ----------
        array: Union[np.ndarray, sparse.spmatrix, da.core.Array]
            Numpy array, Scipy sparse matrix or Dask array from which to construct the linear operator.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        """
        if isinstance(array, np.ndarray):
            is_dense, is_sparse, is_dask = True, False, False
        elif isinstance(array, sparse.spmatrix):
            is_dense, is_sparse, is_dask = False, True, False
        elif isinstance(array, da.core.Array):
            is_dense, is_sparse, is_dask = False, False, True
        else:
            raise TypeError('Invalid input type.')
        super(ExplicitLinearOperator, self).__init__(shape=array.shape, dtype=array.dtype, is_explicit=True,
                                                     is_dask=is_dask, is_dense=is_dense, is_sparse=is_sparse,
                                                     is_symmetric=is_symmetric)
        self.mat = array

    def __call__(self, x: Union[Number, np.ndarray, da.core.Array]) -> Union[Number, np.ndarray]:
        if self.is_dask:
            x = da.from_array(x) if not isinstance(x, da.core.Array) else x
            return (self.mat.dot(x)).compute()
        else:
            return self.mat.dot(x)

    def adjoint(self, y: Union[Number, np.ndarray, da.core.Array]) -> Union[Number, np.ndarray]:
        if self.is_dask:
            y = da.from_array(y) if not isinstance(y, da.core.Array) else y
            return (self.mat.conj().transpose().dot(y)).compute()
        else:
            return self.mat.conj().transpose().dot(y)


class DenseLinearOperator(ExplicitLinearOperator):
    r"""
    Construct a linear operator from a Numpy array.

    The array is stored in the attribute ``self.mat``.
    """

    def __init__(self, ndarray: np.ndarray, is_symmetric: bool = False):
        r"""
        Parameters
        ----------
        ndarray: numpy.ndarray
            Numpy array from which to construct the linear operator.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        """
        super(DenseLinearOperator, self).__init__(array=ndarray, is_symmetric=is_symmetric)


class SparseLinearOperator(ExplicitLinearOperator):
    r"""
    Construct a linear operator from a sparse Scipy matrix (:py:class:`scipy.sparse.spmatrix`).

    The array is stored in the attribute ``self.mat``.
    """

    def __init__(self, spmatrix: sparse.spmatrix, is_symmetric: bool = False):
        r"""
        Parameters
        ----------
        spmatrix: scipy.sparse.spmatrix
            Scipy sparse matrix from which to construct the linear operator.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        """
        super(SparseLinearOperator, self).__init__(array=spmatrix, is_symmetric=is_symmetric)


class DaskLinearOperator(ExplicitLinearOperator):
    r"""
    Construct a linear operator from a Dask array (:py:class:`dask.array.core.Array`).

    The array is stored in the attribute ``self.mat``.
    """

    def __init__(self, dask_array: da.core.Array, is_symmetric: bool = False):
        r"""
        Parameters
        ----------
        dask_array: :py:class:`dask.array.core.Array`
            Dask array from which to construct the linear operator.
        is_symmetric: bool
            Whether the linear operator is symmetric or not.
        """
        super(DaskLinearOperator, self).__init__(array=dask_array, is_symmetric=is_symmetric)


class PolynomialLinearOperator(LinearOperator):
    def __init__(self, LinOp: LinearOperator, coeffs: Union[np.ndarray, list, tuple]):
        self.coeffs = np.asarray(coeffs).astype(LinOp.dtype)
        if LinOp.shape[0] != LinOp.shape[1]:
            raise ValueError('Input linear operator must be square.')
        else:
            self.Linop = LinOp
        super(PolynomialLinearOperator, self).__init__(shape=LinOp.shape, dtype=LinOp.dtype,
                                                       is_explicit=LinOp.is_explicit, is_dense=LinOp.is_dense,
                                                       is_sparse=LinOp.is_sparse,
                                                       is_dask=LinOp.is_dask,
                                                       is_symmetric=LinOp.is_symmetric)

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        z = x.astype(self.dtype)
        y = self.coeffs[0] * x
        for i in range(1, len(self.coeffs)):
            z = self.Linop(z)
            y += self.coeffs[i] * z
        return y

    def adjoint(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.is_symmetric:
            return self(x)
        else:
            z = x.astype(self.dtype)
            y = np.conj(self.coeffs[0]) * x
            for i in range(1, len(self.coeffs)):
                z = self.Linop.adjoint(z)
                y += np.conj(self.coeffs[i]) * z
            return y


class DiagonalOperator(LinearOperator):
    r"""
    Construct a diagonal operator.
    """

    def __init__(self, diag: Union[Number, np.ndarray]):
        """
        Parameters
        ----------
        diag:  Union[Number, np.ndarray]
            Diagonal of the operator.
        """
        self.diag = np.asarray(diag).reshape(-1)
        super(DiagonalOperator, self).__init__(shape=(self.diag.size, self.diag.size), dtype=self.diag.dtype,
                                               is_explicit=False, is_dense=False, is_sparse=False, is_dask=False,
                                               is_symmetric=np.alltrue(np.isreal(self.diag)))
        self.lipschitz_cst = self.diff_lipschitz_cst = np.max(diag)

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.shape[0] == 1:
            return np.asscalar(self.diag * x)
        else:
            return self.diag * x

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.diag.size == 1:
            return np.asscalar(self.diag.conj() * y)
        else:
            return self.diag.conj() * y


class IdentityOperator(DiagonalOperator):
    r"""
    Square identity operator.
    """

    def __init__(self, size: int, dtype: Optional[type] = None):
        r"""
        Parameters
        ----------
        size: int
            Dimension of the domain.
        dtype: Optional[type]
            Data type of the operator.
        """
        diag = np.ones(shape=(size,), dtype=dtype)
        super(IdentityOperator, self).__init__(diag)
        self.lipschitz_cst = self.diff_lipschitz_cst = 1


class HomothetyMap(DiagonalOperator):
    def __init__(self, size: int, constant: Number):
        self.cst = constant
        super(HomothetyMap, self).__init__(diag=self.cst)
        self.shape = (size, size)
        self.lipschitz_cst = self.diff_lipschitz_cst = constant

    def jacobianT(self, arg: Optional[Number] = None) -> Number:
        return self.cst


class NullOperator(LinearOperator):
    r"""
    Null operator.
    """

    def __init__(self, shape: Tuple[int, int], dtype: Optional[type] = np.float):
        super(NullOperator, self).__init__(shape=shape, dtype=dtype,
                                           is_explicit=False, is_dense=False, is_sparse=False, is_dask=False,
                                           is_symmetric=True if (shape[0] == shape[1]) else False)
        self.lipschitz_cst = self.diff_lipschitz_cst = 0

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return np.zeros(shape=self.shape[0], dtype=self.dtype)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return np.zeros(shape=self.shape[1], dtype=self.dtype)

    def eigenvals(self, k: int, which='LM', **kwargs) -> np.ndarray:
        return np.zeros(shape=(k,), dtype=self.dtype)

    def singularvals(self, k: int, which='LM', **kwargs) -> np.ndarray:
        return np.zeros(shape=(k,), dtype=self.dtype)


class LinOpStack(LinearOperator, DiffMapStack):
    r"""
    Stack linear operators together.

    This class constructs a linear operator by stacking multiple linear operators together, either **vertically** (``axis=0``) or **horizontally** (``axis=1``):

    - **Vertical stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N}\to \mathbb{R}^{M_i}, i=1,\ldots, k\}`
      of linear operators. Their vertical stacking is defined as the operator

      .. math::

         V:\begin{cases}\mathbb{R}^{N}\to \mathbb{R}^{M_1}\times \cdots \times\mathbb{R}^{M_k}\\
         \mathbf{x}\mapsto (L_1\mathbf{x},\ldots, L_k\mathbf{x}).
         \end{cases}

      The adjoint of :math:`V` is moreover given by:

      .. math::

         V^\ast(\mathbf{y}_1, \ldots, \mathbf{y}_k)=\sum_{i=1}^k L_i^\ast \mathbf{y}_i, \quad \forall (\mathbf{y}_1, \ldots, \mathbf{y}_k)\in \mathbb{R}^{M_1}\times \cdots \times\mathbb{R}^{M_k}.



    - **Horizontal stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N_i}\to \mathbb{R}^{M}, i=1,\ldots, k\}`
      of linear operators. Their horizontal stacking is defined as the operator

      .. math::

         H:\begin{cases}\mathbb{R}^{N_1}\times \cdots \times\mathbb{R}^{N_k}\to \mathbb{R}^{M}\\
         (\mathbf{x}_1,\ldots, \mathbf{x}_k)\mapsto \sum_{i=1}^k L_i \mathbf{x}_i.
         \end{cases}

      The adjoint of :math:`H` is moreover given by:

      .. math::

         H^\ast(\mathbf{y})=(L_1^\ast \mathbf{y},\ldots, L_k^\ast \mathbf{y}) \quad \forall \mathbf{y}\in \mathbb{R}^{M}.

    Examples
    --------

    We can form the 2D gradient operator by stacking two 1D derivative operators:

    .. testsetup::

       import numpy as np
       from pycsou.core.linop import LinOpStack
       from pycsou.util.misc import peaks

    .. doctest::

       >>> from pycsou.linop.diff import FirstDerivative, Gradient
       >>> x = np.linspace(-2.5, 2.5, 100)
       >>> X,Y = np.meshgrid(x,x)
       >>> Z = peaks(X, Y)
       >>> D1 = FirstDerivative(size=Z.size, shape=Z.shape, axis=0, kind='centered')
       >>> D2 = FirstDerivative(size=Z.size, shape=Z.shape, axis=1, kind='centered')
       >>> G1 = LinOpStack(D1, D2, axis=0)
       >>> G2 = Gradient(shape=Z.shape, kind='centered')
       >>> Z_d = D2*Z.flatten()
       >>> np.allclose(G1*Z.flatten(), G2 * Z.flatten())
       True
       >>> np.allclose(G1.adjoint(G1*Z.flatten()), G2.adjoint(G2 * Z.flatten()))
       True
       >>> G3 = LinOpStack(D1.H, D2.H, axis=1)
       >>> np.allclose(G1.adjoint(G1*Z.flatten()), (G3 * G1) * Z.flatten())
       True

    See Also
    --------
    :py:class:`~pycsou.core.linop.LinOpVStack`, :py:class:`~pycsou.core.linop.LinOpHStack`
    """

    def __init__(self, *linops, axis: int):
        DiffMapStack.__init__(self, *linops, axis=axis)
        self.linops = self.maps
        self.is_explicit_list = [linop.is_explicit for linop in self.linops]
        self.is_dense_list = [linop.is_dense for linop in self.linops]
        self.is_sparse_list = [linop.is_sparse for linop in self.linops]
        self.is_dask_list = [linop.is_dask for linop in self.linops]
        self.is_symmetric_list = [linop.is_symmetric for linop in self.linops]
        LinearOperator.__init__(self, shape=self.shape,
                                is_explicit=bool(np.prod(self.is_explicit_list).astype(bool)),
                                is_dense=bool(np.prod(self.is_dense_list).astype(bool)),
                                is_sparse=bool(np.prod(self.is_sparse_list).astype(bool)),
                                is_dask=bool(np.prod(self.is_dask_list).astype(bool)),
                                is_symmetric=bool(np.prod(self.is_symmetric_list).astype(bool)),
                                lipschitz_cst=self.lipschitz_cst)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.axis == 0:
            y_split = np.split(y, self.sections)
            result = 0
            for i, linop in enumerate(self.linops):
                result += linop.adjoint(y_split[i])
            return result
        else:
            out_list = [linop.adjoint(y).flatten() for linop in self.linops]
            return np.concatenate(out_list, axis=0)


class LinOpVStack(LinOpStack):
    r"""
    Alias for vertical stacking, equivalent to ``LinOpStack(*linops, axis=0)``.
    """

    def __init__(self, *linops):
        super(LinOpVStack, self).__init__(*linops, axis=0)


class LinOpHStack(LinOpStack):
    r"""
    Alias for horizontal stacking, equivalent to ``LinOpStack(*linops, axis=1)``.
    """

    def __init__(self, *linops):
        super(LinOpHStack, self).__init__(*linops, axis=1)

class DiagonalBlock(LinearOperator):
    pass

class LinOpBlock(LinearOperator):
    pass