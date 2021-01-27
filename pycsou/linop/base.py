# #############################################################################
# base.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Interface classes for constructing linear operators.
"""

from numbers import Number
from typing import Union, Optional, Tuple

import numpy as np
import pylops
from dask import array as da
from scipy import sparse as sparse

from pycsou.core.linop import LinearOperator
from pycsou.core.map import DiffMapStack


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
       from pycsou.linop.base import LinOpStack
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
    :py:class:`~pycsou.linop.base.LinOpVStack`, :py:class:`~pycsou.linop.base.LinOpHStack`
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


class HomothetyMap(DiagonalOperator):
    def __init__(self, size: int, constant: Number):
        self.cst = constant
        super(HomothetyMap, self).__init__(diag=self.cst)
        self.shape = (size, size)
        self.lipschitz_cst = self.diff_lipschitz_cst = constant

    def jacobianT(self, arg: Optional[Number] = None) -> Number:
        return self.cst