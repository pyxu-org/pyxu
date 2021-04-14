# #############################################################################
# linop.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Abstract classes for multidimensional linear maps.
"""

from pycsou.core.map import DifferentiableMap, DiffMapSum, DiffMapComp, Map, MapSum, MapComp
import numpy as np
from typing import Union, Tuple, Optional
from abc import abstractmethod
from numbers import Number
import pylops
from pylops.optimization.leastsquares import NormalEquationsInversion
import scipy.sparse.linalg as spls


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

    def transpose(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        r"""
        Evaluates the tranpose of the operator at a point.

        Parameters
        ----------
        y: Union[Number, np.ndarray]
            Point at which the transpose should be evaluated.

        Returns
        -------
        Union[Number, np.ndarray]
            Evaluation of the transpose at ``y``.

        Notes
        -----
        For real-valued operators, the adjoint and the transpose coincide. For complex-valued operators, we can define the
        transpose as :math:`\mathbf{A}^T \mathbf{y}=\overline{\mathbf{A}^\ast \overline{\mathbf{y}}}`.

        """
        return self.adjoint(y.conj()).conj()

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

    def get_transposeOp(self) -> 'TransposeLinearOperator':
        r"""
        Return the transpose operator as a ``LinearOperator`` instance.

        Returns
        -------
        TransposeLinearOperator
            The transpose operator.

        Notes
        -----
        For real-valued operators, the adjoint and the transpose coincide. For complex-valued operators, we can define the
        transpose as :math:`\mathbf{A}^T \mathbf{y}=\overline{\mathbf{A}^\ast \overline{\mathbf{y}}}`.
        """
        return TransposeLinearOperator(self)

    @property
    def T(self):
        r"""Alias for ``self.get_adjointOp``."""
        return self.get_transposeOp()

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
           0.5

        Notes
        -----
        The Lipschtiz constant of a linear operator is its largest singular value. This function therefore calls
        the methods  ``self.singularvals`` with ``k=1`` and ``which='LM'`` to perform this computation.

        Warnings
        --------
        For high-dimensional linear operators this method can be very time-consuming. Reducing the computation accuracy with
        the optional argument ``tol: float`` may help reduce the computational burden. See Scipy's functions :py:func:`scipy.sparse.linalg.eigs`,
        :py:func:`scipy.sparse.linalg.eigsh`, :py:func:`scipy.sparse.linalg.svds` for more on this parameter.

        """
        if self.is_symmetric is True:
            self.lipschitz_cst = float(np.abs(self.eigenvals(k=1, **kwargs)))
        else:
            self.lipschitz_cst = float(self.singularvals(k=1, **kwargs))
        self.diff_lipschitz_cst = self.lipschitz_cst

    def todense(self) -> 'DenseLinearOperator':
        r"""
        Convert the operator to a :py:class:`~pycsou.linop.base.DenseLinearOperator`.

        Returns
        -------
        :py:class:`~pycsou.linop.base.DenseLinearOperator`
            The dense linear operator representation.
        """
        from pycsou.linop.base import DenseLinearOperator

        return DenseLinearOperator(self.PyLop.todense())

    def tosparse(self) -> 'SparseLinearOperator':
        r"""
        Convert the operator to a :py:class:`~pycsou.linop.base.SparseLinearOperator`.

        Returns
        -------
        :py:class:`~pycsou.linop.base.SparseLinearOperator`
            The sparse linear operator representation.
        """
        from pycsou.linop.base import SparseLinearOperator

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

    @property
    def RowProjector(self):
        r"""Orthogonal projection operator onto the rows of ``self``. It is given by ``self * self.dagger``."""
        return SymmetricLinearOperator(self * self.dagger)

    @property
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
            from pycsou.linop.base import HomothetyMap

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
            from pycsou.linop.base import HomothetyMap

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


class TransposeLinearOperator(LinearOperator):
    def __init__(self, LinOp: LinearOperator):
        super(TransposeLinearOperator, self).__init__(shape=(LinOp.shape[1], LinOp.shape[0]), dtype=LinOp.dtype,
                                                      is_explicit=LinOp.is_explicit, is_dask=LinOp.is_dask,
                                                      is_dense=LinOp.is_dense, is_sparse=LinOp.is_sparse,
                                                      is_symmetric=LinOp.is_symmetric)
        self.Linop = LinOp

    def __call__(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Linop.adjoint(y.conj()).conj()

    def adjoint(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Linop.matvec(x)


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
        from pycsou.linop.base import IdentityOperator

        return IdentityOperator(size=self.size, dtype=self.dtype)

    @property
    def DomainGram(self):
        from pycsou.linop.base import IdentityOperator

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
        return NormalEquationsInversion(Op=self.LinOp.PyLop, Regs=None, data=x, epsI=self.eps, returninfo=False)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return LinOpPinv(LinOp=self.LinOp.H, eps=self.eps).__call__(y)
