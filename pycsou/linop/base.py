# #############################################################################
# base.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Classes for constructing linear operators.
"""

from numbers import Number
from typing import Union, Optional, Tuple, List

import numpy as np
import pylops
import joblib as job
from dask import array as da
from scipy import sparse as sparse

from pycsou.core.linop import LinearOperator
from pycsou.core.map import DiffMapStack


class PyLopLinearOperator(LinearOperator):
    r"""
    Construct a linear operator from a :py:class:`pylops.LinearOperator` instance.
    """

    def __init__(self, PyLop: pylops.LinearOperator, is_symmetric: bool = False, is_dense: bool = False,
                 is_sparse: bool = False, lipschitz_cst: float = np.infty):
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
        lipschitz_cst: float
            Lipschitz constant of the operator.
        """
        super(PyLopLinearOperator, self).__init__(shape=PyLop.shape, dtype=PyLop.dtype, is_explicit=PyLop.explicit,
                                                  is_dense=is_dense, is_sparse=is_sparse, is_dask=False,
                                                  is_symmetric=is_symmetric, lipschitz_cst=lipschitz_cst)
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

      The Lipschitz constant of the vertically stacked operator can be bounded by :math:`\sqrt{\sum_{i=1}^k \|L_i\|_2^2}`.


    - **Horizontal stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N_i}\to \mathbb{R}^{M}, i=1,\ldots, k\}`
      of linear operators. Their horizontal stacking is defined as the operator

      .. math::

         H:\begin{cases}\mathbb{R}^{N_1}\times \cdots \times\mathbb{R}^{N_k}\to \mathbb{R}^{M}\\
         (\mathbf{x}_1,\ldots, \mathbf{x}_k)\mapsto \sum_{i=1}^k L_i \mathbf{x}_i.
         \end{cases}

      The adjoint of :math:`H` is moreover given by:

      .. math::

         H^\ast(\mathbf{y})=(L_1^\ast \mathbf{y},\ldots, L_k^\ast \mathbf{y}) \quad \forall \mathbf{y}\in \mathbb{R}^{M}.

      The Lipschitz constant of the horizontally stacked operator can be bounded by :math:`{\max_{i=1}^k \|L_i\|_2}`.

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
       >>> parG1 = LinOpStack(D1, D2, axis=0, n_jobs=-1)
       >>> parG3 = LinOpStack(D1.H, D2.H, axis=1, n_jobs=-1)
       >>> np.allclose(G1.adjoint(G1*Z.flatten()), parG1.adjoint(parG1*Z.flatten()))
       True
       >>> np.allclose((G3 * G1) * Z.flatten(), (parG3 * parG1) * Z.flatten())
       True

    See Also
    --------
    :py:class:`~pycsou.linop.base.LinOpVStack`, :py:class:`~pycsou.linop.base.LinOpHStack`
    """

    def __init__(self, *linops, axis: int, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
       Parameters
       ----------
       linops: LinearOperator
           List of linear operators to stack.
       axis:
           Stacking direction: 0 for vertical and 1 for horizontal stacking.
       n_jobs: int
           Number of cores to be used for parallel evaluation of the linear operator stack and its adjoint.
           If ``n_jobs==1``, the operator stack and its adjoint are evaluated sequentially, otherwise they are
           evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
       joblib_backend: str
           Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

       """
        DiffMapStack.__init__(self, *linops, axis=axis, n_jobs=n_jobs, joblib_backend=joblib_backend)
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
            if self.n_jobs == 1:
                result = 0
                for i, linop in enumerate(self.linops):
                    result += linop.adjoint(y_split[i])
            else:
                with job.Parallel(backend=self.joblib_backend, n_jobs=self.n_jobs, verbose=False) as parallel:
                    out_list = parallel(job.delayed(linop.adjoint)(y_split[i])
                                        for i, linop in enumerate(self.linops))
                    result = np.sum(np.stack(out_list, axis=0), axis=0)
            return result
        else:
            if self.n_jobs == 1:
                out_list = [linop.adjoint(y).flatten() for linop in self.linops]
            else:
                with job.Parallel(backend=self.joblib_backend, n_jobs=self.n_jobs, verbose=False) as parallel:
                    out_list = parallel(job.delayed(linop.adjoint)(y) for linop in self.linops)
                out_list = [y.flatten() for y in out_list]
            return np.concatenate(out_list, axis=0)


class LinOpVStack(LinOpStack):
    r"""
    Alias for vertical stacking, equivalent to ``LinOpStack(*linops, axis=0)``.
    """

    def __init__(self, *linops, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
       Parameters
       ----------
       linops: LinearOperator
           List of linear operators to stack.
       n_jobs: int
           Number of cores to be used for parallel evaluation of the linear operator stack and its adjoint.
           If ``n_jobs==1``, the operator stack and its adjoint are evaluated sequentially, otherwise they are
           evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
       joblib_backend: str
           Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

       """
        super(LinOpVStack, self).__init__(*linops, axis=0, n_jobs=n_jobs, joblib_backend=joblib_backend)


class LinOpHStack(LinOpStack):
    r"""
    Alias for horizontal stacking, equivalent to ``LinOpStack(*linops, axis=1)``.
    """

    def __init__(self, *linops, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
       Parameters
       ----------
       linops: LinearOperator
           List of linear operators to stack.
       n_jobs: int
           Number of cores to be used for parallel evaluation of the linear operator stack and its adjoint.
           If ``n_jobs==1``, the operator stack and its adjoint are evaluated sequentially, otherwise they are
           evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
       joblib_backend: str
           Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

       """
        super(LinOpHStack, self).__init__(*linops, axis=1, n_jobs=n_jobs, joblib_backend=joblib_backend)


def BlockOperator(linops: List[List[LinearOperator]], n_jobs: int = 1) -> PyLopLinearOperator:
    r"""
    Construct a block operator from N lists of M linear operators each.

    Parameters
    ----------
    linops: List[List[LinearOperator]]
        List of lists of linear operators to be combined in block fashion.
        Alternatively, numpy.ndarray or scipy.sparse.spmatrix can be passed in place of one or more operators.
    n_jobs: int
        Number of processes used to evaluate the N operators in parallel using multiprocessing.
        If ``n_jobs=1`` (default), work in serial mode.


    Returns
    -------
    PyLopLinearOperator
        Block linear operator.

    Examples
    --------

    .. doctest::

        >>> from pycsou.linop.base import BlockOperator
        >>> from pycsou.linop.diff import SecondDerivative
        >>> Nv, Nh = 11, 21
        >>> D2hop = SecondDerivative(size=Nv * Nh, shape=(Nv,Nh), axis=1)
        >>> D2vop = SecondDerivative(size=Nv * Nh, shape=(Nv,Nh), axis=0)
        >>> Dblock = BlockOperator([[D2vop, 0.5 * D2vop, - D2hop], [D2hop, 2 * D2hop, D2vop]])
        >>> x = np.zeros((Nv, Nh)); x[int(Nv//2), int(Nh//2)] = 1; z = np.tile(x, (3,1)).flatten()
        >>> np.allclose(Dblock(z), np.concatenate(((D2vop + 0.5 * D2vop - D2hop)(x.flatten()), (D2hop + 2 * D2hop + D2vop)(x.flatten()))))
        True

    Notes
    -----
    In mathematics, a block or a partitioned matrix is a matrix that is
    interpreted as being broken into sections called blocks or submatrices.
    Similarly a block operator is composed of N sets of M linear operators
    each such that its application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_{1,1}}  & \mathbf{L_{1,2}} &  \cdots & \mathbf{L_{1,M}}  \\
            \mathbf{L_{2,1}}  & \mathbf{L_{2,2}} &  \cdots & \mathbf{L_{2,M}}  \\
            \vdots               & \vdots              &  \cdots & \vdots               \\
            \mathbf{L_{N,1}}  & \mathbf{L_{N,2}} &  \cdots & \mathbf{L_{N,M}}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{x}_{M}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_{1,1}} \mathbf{x}_{1} + \mathbf{L_{1,2}} \mathbf{x}_{2} +
            \mathbf{L_{1,M}} \mathbf{x}_{M} \\
            \mathbf{L_{2,1}} \mathbf{x}_{1} + \mathbf{L_{2,2}} \mathbf{x}_{2} +
            \mathbf{L_{2,M}} \mathbf{x}_{M} \\
            \vdots     \\
            \mathbf{L_{N,1}} \mathbf{x}_{1} + \mathbf{L_{N,2}} \mathbf{x}_{2} +
            \mathbf{L_{N,M}} \mathbf{x}_{M} \\
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_{1,1}}^\ast  & \mathbf{L_{2,1}}^\ast &  \cdots &
            \mathbf{L_{N,1}}^\ast  \\
            \mathbf{L_{1,2}}^\ast  & \mathbf{L_{2,2}}^\ast &  \cdots &
            \mathbf{L_{N,2}}^\ast  \\
            \vdots                 & \vdots                &  \cdots & \vdots \\
            \mathbf{L_{1,M}}^\ast  & \mathbf{L_{2,M}}^\ast &  \cdots &
            \mathbf{L_{N,M}}^\ast  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_{1,1}}^\ast \mathbf{y}_{1} +
            \mathbf{L_{2,1}}^\ast \mathbf{y}_{2} +
            \mathbf{L_{N,1}}^\ast \mathbf{y}_{N} \\
            \mathbf{L_{1,2}}^\ast \mathbf{y}_{1} +
            \mathbf{L_{2,2}}^\ast \mathbf{y}_{2} +
            \mathbf{L_{N,2}}^\ast \mathbf{y}_{N} \\
            \vdots     \\
            \mathbf{L_{1,M}}^\ast \mathbf{y}_{1} +
            \mathbf{L_{2,M}}^\ast \mathbf{y}_{2} +
            \mathbf{L_{N,M}}^\ast \mathbf{y}_{N} \\
        \end{bmatrix}

    The Lipschitz constant of the block operator can be bounded by :math:`\max_{j=1}^M\sqrt{\sum_{i=1}^N \|\mathbf{L}_{i,j}\|_2^2}`.


    Warnings
    --------
    The parameter ``n_jobs`` is currently unused and is there for compatibility with the future API of PyLops.
    The code should be updated when the next version on PyLops is released.

    See Also
    --------
    :py:class:`~pycsou.linop.base.BlockDiagonalOperator`, :py:class:`~pycsou.linop.base.LinOpStack`
    """
    pylinops = [[linop.PyLop for linop in linops_line] for linops_line in linops]
    lipschitz_csts = [[linop.lipschitz_cst for linop in linops_line] for linops_line in linops]
    lipschitz_cst = np.max(np.linalg.norm(lipschitz_csts, axis=0))
    block = pylops.Block(ops=pylinops)
    return PyLopLinearOperator(block, lipschitz_cst=lipschitz_cst)


def BlockDiagonalOperator(*linops: LinearOperator, n_jobs: int = 1) -> PyLopLinearOperator:
    r"""
    Construct a block diagonal operator from N linear operators.

    Parameters
    ----------
    linops: LinearOperator
        Linear operators forming the diagonal blocks.
        Alternatively, numpy.ndarray or scipy.sparse.spmatrix can be passed in place of one or more operators.
    n_jobs: int
        Number of processes used to evaluate the N operators in parallel using multiprocessing.
        If ``n_jobs=1`` (default), work in serial mode.


    Returns
    -------
    PyLopLinearOperator
        Block diagonal linear operator.

    Examples
    --------

    .. doctest::

        >>> from pycsou.linop.base import BlockDiagonalOperator
        >>> from pycsou.linop.diff import SecondDerivative
        >>> Nv, Nh = 11, 21
        >>> D2hop = SecondDerivative(size=Nv * Nh, shape=(Nv,Nh), axis=1)
        >>> D2vop = SecondDerivative(size=Nv * Nh, shape=(Nv,Nh), axis=0)
        >>> Dblockdiag = BlockDiagonalOperator(D2vop, 0.5 * D2vop, -1 * D2hop)
        >>> x = np.zeros((Nv, Nh)); x[int(Nv//2), int(Nh//2)] = 1; z = np.tile(x, (3,1)).flatten()
        >>> np.allclose(Dblockdiag(z), np.concatenate((D2vop(x.flatten()), 0.5 * D2vop(x.flatten()), - D2hop(x.flatten()))))
        True

    Notes
    -----
    A block-diagonal operator composed of N linear operators is created such
    as its application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_1}  & \mathbf{0}   &  \cdots &  \mathbf{0}  \\
            \mathbf{0}    & \mathbf{L_2} &  \cdots &  \mathbf{0}  \\
           \vdots           & \vdots          &  \ddots &  \vdots         \\
            \mathbf{0}    & \mathbf{0}   &  \cdots &  \mathbf{L_N}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{x}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_1} \mathbf{x}_{1}  \\
            \mathbf{L_2} \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{L_N} \mathbf{x}_{N}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L_1}^\ast  & \mathbf{0}    &  \cdots &   \mathbf{0}  \\
            \mathbf{0}    &  \mathbf{L_2}^\ast  &  \cdots &   \mathbf{0}  \\
            \vdots           &  \vdots              &  \ddots &   \vdots        \\
            \mathbf{0}    &  \mathbf{0}      &  \cdots &   \mathbf{L_N}^\ast
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L_1}^\ast \mathbf{y}_{1}  \\
            \mathbf{L_2}^\ast \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{L_N}^\ast \mathbf{y}_{N}
        \end{bmatrix}

    The Lipschitz constant of the block-diagonal operator can be bounded by :math:`{\max_{i=1}^N \|\mathbf{L}_{i}\|_2}`.

    Warnings
    --------
    The parameter ``n_jobs`` is currently unused and is there for compatibility with the future API of PyLops.
    The code should be updated when the next version on PyLops is released.

    See Also
    --------
    :py:class:`~pycsou.linop.base.BlockOperator`, :py:class:`~pycsou.linop.base.LinOpStack`
    """
    pylinops = [linop.PyLop for linop in linops]
    lipschitz_cst = np.array([linop.lipschitz_cst for linop in linops]).max()
    block_diag = pylops.BlockDiag(ops=pylinops)
    return PyLopLinearOperator(block_diag, lipschitz_cst=lipschitz_cst)


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
        if self.shape[1] == 1:
            return np.asscalar(self.diag * x)
        else:
            return self.diag * x

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.shape[0] == 1:
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

    def __init__(self, shape: Tuple[int, int], dtype: Optional[type] = np.float64):
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


class PolynomialLinearOperator(LinearOperator):
    r"""
    Polynomial linear operator :math:`P(L)`.

    Base class for polynomial operators. Useful for implementing generalised differential operators.

    Given a polynomial :math:`P(x)=\sum_{k=0}^N a_k x^k` and a square linear operator :math:`\mathbf{L}:\mathbb{R}^N\to \mathbb{R}^N,`
    we define the polynomial linear operator :math:`P(\mathbf{L}):\mathbb{R}^N\to \mathbb{R}^N` as:

    .. math::

       P(\mathbf{L})=\sum_{k=0}^N a_k \mathbf{L}^k,

    where :math:`\mathbf{L}^0` is the identity matrix.
    The *adjoint* of :math:`P(\mathbf{L})` is given by:

    .. math::

       P(\mathbf{L})^\ast=\sum_{k=0}^N a_k (\mathbf{L}^\ast)^k.

    Examples
    --------

    .. testsetup::

       import numpy as np

    .. doctest::

       >>> from pycsou.linop import DenseLinearOperator, PolynomialLinearOperator
       >>> L = DenseLinearOperator(np.arange(64).reshape(8,8))
       >>> PL = PolynomialLinearOperator(LinOp=L, coeffs=[1/2 ,2, 1])
       >>> x = np.arange(8)
       >>> np.allclose(PL(x), x/2 + 2 * L(x) + (L**2)(x))
       True

    """

    def __init__(self, LinOp: LinearOperator, coeffs: Union[np.ndarray, list, tuple]):
        r"""

        Parameters
        ----------
        LinOp: pycsou.core.LinearOperator
            Square linear operator :math:`\mathbf{L}`.
        coeffs: Union[np.ndarray, list, tuple]
            Coefficients :math:`\{a_0,\ldots, a_N\}` of the polynomial :math:`P`.
        """
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


class KroneckerProduct(LinearOperator):
    r"""
    Kronecker product :math:`\otimes` of two operators.

    Examples
    --------

    .. doctest::

        >>> from pycsou.linop.base import KroneckerProduct
        >>> from pycsou.linop.diff import SecondDerivative
        >>> Nv, Nh = 11, 21
        >>> D2hop = SecondDerivative(size=Nh)
        >>> D2vop = SecondDerivative(size=Nv)
        >>> Dkron = KroneckerProduct(D2hop, D2vop)
        >>> x = np.zeros((Nv, Nh)); x[int(Nv//2), int(Nh//2)] = 1
        >>> np.allclose(Dkron(x.flatten()), D2vop.apply_along_axis(D2hop.apply_along_axis(x.transpose(), axis=0).transpose(), axis=0).flatten())
        True

    Notes
    -----
    The *Kronecker product* between two operators :math:`\mathbf{A}\in \mathbb{R}^{k\times l}` and :math:`\mathbf{B}\in \mathbb{R}^{n\times m}`
    is defined as:

    .. math::

        \mathbf{A} \otimes \mathbf{B}=\left[
        \begin{array}{ccc}
        A_{11}\mathbf{B} & \cdots & A_{1l}\mathbf{B} \\
        \vdots & \ddots & \vdots \\
        A_{k1}\mathbf{B} & \cdots & A_{kl}\mathbf{B} \\
        \end{array}
        \right] \in \mathbb{R}^{kn\times lm}

    Let :math:`\mathbf{X}\in \mathbb{R}^{m\times l}` and :math:`\mathbf{Y}\in \mathbb{R}^{n\times k}`. Then we have:

    .. math::

         (\mathbf{A} \otimes \mathbf{B})\mbox{vec}(\mathbf{X})= \mbox{vec}\left(\mathbf{B}\mathbf{X}\mathbf{A}^T\right)

    and

    .. math::

         (\mathbf{A} \otimes \mathbf{B})^\ast\mbox{vec}(\mathbf{Y})= \mbox{vec}\left(\mathbf{B}^\ast\mathbf{Y}\overline{\mathbf{A}}\right)

    where :math:`\mbox{vec}` denotes the vectorisation operator.
    Such operations are leveraged to implement the linear operator in matrix-free form (i.e. the matrix :math:`\mathbf{A} \otimes \mathbf{B}` is not explicitely constructed)
    both in forward and adjoint mode.

    We have also :math:`\|\mathbf{A} \otimes \mathbf{B}\|_2=\|\mathbf{A}\|_2\|\mathbf{B}\|_2` and
    :math:`(\mathbf{A} \otimes \mathbf{B})^\dagger= \mathbf{A}^\dagger \otimes \mathbf{B}^\dagger` which we use to compute efficiently
    ``self.lipschitz_cst`` and ``self.PinvOp``.

    See Also
    --------
    :py:class:`~pycsou.linop.base.KroneckerSum`, :py:class:`~pycsou.linop.base.KhatriRaoProduct`
    """

    def __init__(self, linop1: LinearOperator, linop2: LinearOperator):
        r"""

        Parameters
        ----------
        linop1: LinearOperator
            Linear operator on the left hand-side of the Kronecker product (multiplicand).
        linop2: LinearOperator
            Linear operator on the right hand-side of the Kronecker product (multiplier).
        """
        self.linop1 = linop1
        self.linop2 = linop2
        super(KroneckerProduct, self).__init__(
            shape=(self.linop2.shape[0] * self.linop1.shape[0], self.linop2.shape[1] * self.linop1.shape[1]),
            dtype=self.linop1.dtype,
            lipschitz_cst=self.linop1.lipschitz_cst * self.linop2.lipschitz_cst)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape((self.linop2.shape[1], self.linop1.shape[1]))
        return self.linop2.apply_along_axis(self.linop1.apply_along_axis(X.transpose(), axis=0).transpose(),
                                            axis=0).flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        Y = y.reshape((self.linop2.shape[0], self.linop1.shape[0]))
        return self.linop2.H.apply_along_axis(self.linop1.H.apply_along_axis(Y.transpose(), axis=0).transpose(),
                                              axis=0).flatten()

    @property
    def PinvOp(self) -> 'KroneckerProduct':
        return KroneckerProduct(self.linop1.PinvOp, self.linop2.PinvOp)


class KroneckerSum(LinearOperator):
    r"""
    Kronecker sum :math:`\oplus` of two operators.

    Examples
    --------

    .. testsetup::

        import numpy as np

    .. doctest::

        >>> from pycsou.linop.base import KroneckerProduct, KroneckerSum, DiagonalOperator
        >>> m1=np.linspace(0,3,5); m2=np.linspace(-3,2,7)
        >>> D1=DiagonalOperator(diag=m1); ExpD1=DiagonalOperator(diag=np.exp(m1))
        >>> D2=DiagonalOperator(diag=m2); ExpD2=DiagonalOperator(diag=np.exp(m2))
        >>> Expkronprod=KroneckerProduct(ExpD1,ExpD2)
        >>> Kronsum=KroneckerSum(D1,D2)
        >>> np.allclose(np.diag(Expkronprod.todense().mat), np.exp(np.diag(Kronsum.todense().mat)))
        True

    Notes
    -----
    The *Kronecker sum* between two operators :math:`\mathbf{A}\in \mathbb{R}^{k\times l}` and :math:`\mathbf{B}\in \mathbb{R}^{n\times m}`
    is defined as:

    .. math::

        \mathbf{A} \oplus \mathbf{B}=\mathbf{A} \otimes \mathbf{I}_{n\times m} + \mathbf{I}_{k\times l} \otimes \mathbf{B} \in \mathbb{R}^{kn\times lm}.

    Let :math:`\mathbf{X}\in \mathbb{R}^{m\times l}` and :math:`\mathbf{Y}\in \mathbb{R}^{n\times k}`. Then we have:

    .. math::

         (\mathbf{A} \oplus \mathbf{B})\mbox{vec}(\mathbf{X})= \mbox{vec}\left(\mathbf{X}\mathbf{A}^T + \mathbf{B}\mathbf{X}\right)

    and

    .. math::

         (\mathbf{A} \oplus \mathbf{B})^\ast\mbox{vec}(\mathbf{Y})= \mbox{vec}\left(\mathbf{Y}\overline{\mathbf{A}} + \mathbf{B}^\ast\mathbf{Y}\right)

    where :math:`\mbox{vec}` denotes the vectorisation operator.
    Such operations are leveraged to implement the linear operator in matrix-free form (i.e. the matrix :math:`\mathbf{A} \oplus \mathbf{B}` is not explicitely constructed)
    both in forward and adjoint mode.

    The Lipschitz constant of the Kronecker sum can be bounded by :math:`\|\mathbf{A}\|_2+ \|\mathbf{B}\|_2`.


    See Also
    --------
    :py:class:`~pycsou.linop.base.KroneckerSum`, :py:class:`~pycsou.linop.base.KhatriRaoProduct`
    """

    def __init__(self, linop1: LinearOperator, linop2: LinearOperator):
        r"""

        Parameters
        ----------
        linop1: LinearOperator
            Linear operator on the left hand-side of the Kronecker sum.
        linop2: LinearOperator
            Linear operator on the right hand-side of the Kronecker sum.
        """
        self.linop1 = linop1
        self.linop2 = linop2
        super(KroneckerSum, self).__init__(
            shape=(self.linop2.shape[0] * self.linop1.shape[0], self.linop2.shape[1] * self.linop1.shape[1]),
            dtype=self.linop1.dtype,
            lipschitz_cst=self.linop1.lipschitz_cst + self.linop2.lipschitz_cst)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape((self.linop2.shape[1], self.linop1.shape[1]))
        return self.linop1.apply_along_axis(X.transpose(), axis=0).transpose().flatten() + \
               self.linop2.apply_along_axis(X, axis=0).flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        Y = y.reshape((self.linop2.shape[0], self.linop1.shape[0]))
        return self.linop1.H.apply_along_axis(Y.transpose(), axis=0).transpose().flatten() + \
               self.linop2.H.apply_along_axis(Y, axis=0).flatten()


class KhatriRaoProduct(LinearOperator):
    r"""
    Khatri-Rao product :math:`\circ` of two operators.

    Examples
    --------

    .. doctest::

        >>> from pycsou.linop.base import KhatriRaoProduct
        >>> from pycsou.linop.diff import SecondDerivative
        >>> D1 = SecondDerivative(size=11)
        >>> D2 = SecondDerivative(size=11)
        >>> Dkrao = KhatriRaoProduct(D1, D2)
        >>> x = np.arange(11)
        >>> Dkrao(x).shape
        (121,)
        >>> np.allclose(Dkrao(x), ((D1.todense().mat * x[None, :]) @ D2.todense().mat.transpose()).flatten())
        True

    Notes
    -----
    The *Khatri-Rao product* between two operators :math:`\mathbf{A}\in \mathbb{R}^{k\times l}` and :math:`\mathbf{B}\in \mathbb{R}^{n\times l}`
    is defined as the column-wise Kronecker product:

    .. math::

        \mathbf{A} \circ \mathbf{B}=\left[
        \begin{array}{ccc}
        \mathbf{A}_1\otimes \mathbf{B}_1 & \cdots & \mathbf{A}_l\otimes \mathbf{B}_l
        \end{array}
        \right] \in \mathbb{R}^{kn\times l}

    Let :math:`\mathbf{x}\in \mathbb{R}^{l}` and :math:`\mathbf{Y}\in \mathbb{R}^{n\times k}`. Then we have:

    .. math::

         (\mathbf{A} \circ \mathbf{B})\mathbf{x}= \mbox{vec}\left(\mathbf{B}\mbox{diag}(\mathbf{x})\mathbf{A}^T\right)

    and

    .. math::

         (\mathbf{A} \circ \mathbf{B})^\ast\mbox{vec}(\mathbf{Y})= \mbox{diag}\left(\mathbf{B}^\ast\mathbf{Y}\overline{\mathbf{A}}\right)

    where :math:`\mbox{diag}`,  :math:`\mbox{vec}` denote the diagonal and vectorisation operators respectively.
    Such operations are leveraged to implement the linear operator in matrix-free form (i.e. the matrix :math:`\mathbf{A} \circ \mathbf{B}` is not explicitely constructed)
    both in forward and adjoint mode.

    The Lipschitz constant of the Khatri-Rao product can be bounded by :math:`\|\mathbf{A}\|_2\|\mathbf{B}\|_2`.

    See Also
    --------
    :py:class:`~pycsou.linop.base.KroneckerProduct`, :py:class:`~pycsou.linop.base.KroneckerSum`
    """

    def __init__(self, linop1: LinearOperator, linop2: LinearOperator):
        r"""

        Parameters
        ----------
        linop1: LinearOperator
            Linear operator on the left hand-side of the Khatri-Rao product (multiplicand).
        linop2: LinearOperator
            Linear operator on the right hand-side of the Khatri-Rao product (multiplier).

        Raises
        ------
        ValueError
            If ``linop1.shape[1] != self.linop2.shape[1]``.
        """
        if linop1.shape[1] != linop2.shape[1]:
            raise ValueError('Invalid shapes.')
        self.linop1 = linop1
        self.linop2 = linop2
        super(KhatriRaoProduct, self).__init__(
            shape=(self.linop2.shape[0] * self.linop1.shape[0], self.linop2.shape[1]),
            dtype=self.linop1.dtype, lipschitz_cst=self.linop1.lipschitz_cst * self.linop2.lipschitz_cst)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.linop1.is_dense and self.linop2.is_dense:
            return (self.linop2.mat * x[None, :]) @ self.linop1.mat.transpose()
        elif self.linop1.is_sparse and self.linop2.is_sparse:
            return np.asarray(self.linop2.mat.multiply(x[None, :]).dot(self.linop1.mat.transpose()))
        else:
            return self.linop2.apply_along_axis(self.linop1.apply_along_axis(np.diag(x), axis=0).transpose(),
                                                axis=0).flatten()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        Y = y.reshape((self.linop2.shape[0], self.linop1.shape[0]))
        if self.linop1.is_dense and self.linop2.is_dense:
            return np.sum((self.linop1.mat.transpose().conj() @ Y.transpose()).transpose() * self.linop2.mat.conj(),
                          axis=0)
        elif self.linop1.is_sparse and self.linop2.is_sparse:
            return np.asarray(
                self.linop2.mat.conj().multiply(self.linop1.mat.transpose().conj().dot(Y.transpose()).transpose()).sum(
                    axis=0))
        else:
            return np.diag(
                self.linop2.H.apply_along_axis(self.linop1.H.apply_along_axis(Y.transpose(), axis=0).transpose(),
                                               axis=0)).flatten()


if __name__ == '__main__':
    from pycsou.linop.base import BlockDiagonalOperator
    from pycsou.linop.diff import SecondDerivative

    Nv, Nh = 11, 21
    D2hop = SecondDerivative(size=Nv * Nh, shape=(Nv, Nh), axis=1)
    D2vop = SecondDerivative(size=Nv * Nh, shape=(Nv, Nh), axis=0)
    Dblockdiag = BlockDiagonalOperator(D2vop, 0.5 * D2vop, -1 * D2hop)
