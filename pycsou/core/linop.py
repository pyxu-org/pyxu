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
    def __init__(self, shape=Tuple[int, int], dtype: Optional[type] = None,
                 is_explicit: bool = False, is_dense: bool = False, is_sparse: bool = False, is_dask: bool = False,
                 is_symmetric: bool = False):
        DifferentiableMap.__init__(self, shape=shape, is_linear=True)
        self.dtype = dtype
        self.is_explicit = is_explicit
        self.is_dense = is_dense
        self.is_sparse = is_sparse
        self.is_dask = is_dask
        self.is_symmetric = is_symmetric
        self.is_square = True if shape[0] == shape[1] else False

    def matvec(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.__call__(x)

    @abstractmethod
    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        pass

    def jacobianT(self, arg: Union[Number, np.ndarray] = None) -> 'LinearOperator':
        return self.get_adjointOp()

    def get_adjointOp(self) -> 'AdjointLinearOperator':
        if self.is_symmetric is True:
            return self
        else:
            return AdjointLinearOperator(self)

    @property
    def H(self):
        return self.get_adjointOp()

    def RangeGram(self):
        return SymmetricLinearOperator(self * self.H)

    def DomainGram(self):
        return SymmetricLinearOperator(self.H * self)

    def eigenvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        if self.is_symmetric is True:
            eigen_values = spls.eigsh(A=self.SciOp, k=k, which=which, return_eigenvectors=False, **kwargs)
        elif self.is_square is True:
            eigen_values = spls.eigs(A=self.SciOp, k=k, which=which, return_eigenvectors=False, **kwargs)
        else:
            raise NotImplementedError(
                'The function eigenvals is only for square linear operator. For non square linear operators, use the method singularvals.')
        return eigen_values

    def singularvals(self, k: int, which='LM', **kwargs: dict) -> np.ndarray:
        return spls.svds(A=self.SciOp, k=k, which=which, return_singular_vectors=False, **kwargs)

    def compute_lipschitz_cst(self, **kwargs: dict):
        if self.is_square is True:
            self.lipschitz_cst = self.eigenvals(k=1)
        else:
            self.lipschitz_cst = self.singularvals(k=1)
        self.diff_lipschitz_cst = self.lipschitz_cst

    def todense(self) -> 'DenseLinearOperator':
        return DenseLinearOperator(self.PyLop.todense())

    def tosparse(self) -> 'SparseLinearOperator':
        return SparseLinearOperator(self.PyLop.tosparse())

    def tosciop(self) -> spls.LinearOperator:
        return spls.LinearOperator(dtype=self.dtype, shape=self.shape, matvec=self.matvec, rmatvec=self.adjoint)

    @property
    def SciOp(self):
        return self.tosciop()

    def topylop(self) -> pylops.LinearOperator:
        return pylops.LinearOperator(Op=self.SciOp, explicit=self.is_explicit)

    @property
    def PyLop(self):
        return self.topylop()

    def cond(self, **kwargs):
        return self.PyLop.cond(**kwargs)

    def pinv(self, y: Union[Number, np.ndarray], eps: Number = 0, **kwargs) -> Union[Number, np.ndarray]:
        return NormalEquationsInversion(Op=self.PyLop, Regs=None, data=y, epsI=eps, **kwargs, returninfo=False)

    @property
    def PinvOp(self):
        return LinOpPinv(self)

    @property
    def dagger(self):
        return self.PinvOp

    def RowProjector(self):
        return SymmetricLinearOperator(self * self.dagger)

    def ColProjector(self):
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

    def __mul__(self, other: Union['Map', 'DifferentiableMap', 'LinearOperator', Number]) -> Union[
        'MapSum', 'DiffMapSum', 'LinOpSum']:
        if isinstance(other, Number):
            other = HomothetyMap(constant=other)

        if isinstance(other, LinearOperator):
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
            other = HomothetyMap(constant=other)

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
                                is_symmetric=LinOp1.is_symmetric & LinOp2.is_symmetric)
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
                                is_symmetric=LinOp1.is_symmetric & LinOp2.is_symmetric)
        self.LinOp1, self.LinOp2 = LinOp1, LinOp2

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.LinOp1.adjoint(self.LinOp2.adjoint(y))


class SymmetricLinearOperator(LinearOperator):
    def __init__(self, LinOp: LinearOperator):
        if LinOp.shape[0] != LinOp.shape[1]:
            raise TypeError('The input linear operator is not symmetric.')
        super(SymmetricLinearOperator, self).__init__(shape=LinOp.shape, dtype=LinOp.dtype,
                                                      is_explicit=LinOp.is_explicit,
                                                      is_dask=LinOp.is_dask, is_dense=LinOp.is_dense,
                                                      is_sparse=LinOp.is_sparse, is_symmetric=True)

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

    def RangeGram(self):
        return IdentityOperator(size=self.size, dtype=self.dtype)

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
    def __init__(self, PyLop: pylops.LinearOperator, is_symmetric: bool = False, is_dense: bool = False,
                 is_sparse: bool = False):
        super(PyLopLinearOperator, self).__init__(shape=PyLop.shape, dtype=PyLop.dtype, is_explicit=PyLop.explicit,
                                                  is_dense=is_dense, is_sparse=is_sparse, is_dask=False,
                                                  is_symmetric=is_symmetric)
        self.Op = PyLop

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Op.matvec(x)

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.Op.rmatvec(y)


class ExplicitLinearOperator(LinearOperator):
    def __init__(self, array: Union[np.ndarray, sparse.spmatrix, da.array], is_symmetric: bool = False):
        if isinstance(array, np.ndarray):
            is_dense = True
        elif isinstance(array, sparse.spmatrix):
            is_sparse = True
        elif isinstance(array, da.array):
            is_dask = True
        else:
            raise TypeError('Invalid input type.')
        super(ExplicitLinearOperator, self).__init__(shape=array.shape, dtype=array.dtype, is_explicit=True,
                                                     is_dask=is_dask, is_dense=is_dense, is_sparse=is_sparse,
                                                     is_symmetric=is_symmetric)
        self.mat = array

    def __call__(self, x: Union[Number, np.ndarray, da.array]) -> Union[Number, np.ndarray]:
        if self.is_dask:
            x = da.from_array(x) if not isinstance(x, da.array) else x
            return (self.mat.dot(x)).compute()
        else:
            return self.mat.dot(x)

    def adjoint(self, y: Union[Number, np.ndarray, da.array]) -> Union[Number, np.ndarray]:
        if self.is_dask:
            y = da.from_array(y) if not isinstance(y, da.array) else y
            return (self.mat.conj().transpose().dot(y)).compute()
        else:
            return self.mat.conj().transpose().dot(y)


class DenseLinearOperator(ExplicitLinearOperator):
    def __init__(self, ndarray: np.ndarray, is_symmetric: bool = False):
        super(DenseLinearOperator, self).__init__(array=ndarray, is_symmetric=is_symmetric)


class SparseLinearOperator(ExplicitLinearOperator):
    def __init__(self, spmatrix: sparse.spmatrix, is_symmetric: bool = False):
        super(SparseLinearOperator, self).__init__(array=spmatrix, is_symmetric=is_symmetric)


class DaskLinearOperator(ExplicitLinearOperator):
    def __init__(self, dask_array: da.array, is_symmetric: bool = False):
        super(DaskLinearOperator, self).__init__(array=dask_array, is_symmetric=is_symmetric)


class DiagonalOperator(LinearOperator):
    def __init__(self, diag: np.ndarray):
        self.diag = np.asarray(diag).reshape(-1)
        super(DiagonalOperator, self).__init__(shape=(diag.size, diag.size), dtype=diag.dtype, is_explicit=False,
                                               is_dense=False, is_sparse=False, is_dask=False,
                                               is_symmetric=np.alltrue(np.isreal(diag)))

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.diag * x

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.diag.conj() * y


class IdentityOperator(DiagonalOperator):
    def __init__(self, size: int, dtype: Optional[type] = None):
        diag = np.ones(shape=(size,), dtype=dtype)
        super(IdentityOperator, self).__init__(diag)


class HomothetyMap(DiagonalOperator):
    def __init__(self, constant: Number):
        self.cst = np.asarray(constant)
        super(HomothetyMap, self).__init__(diag=self.cst)


class LinOpStack(LinearOperator, DiffMapStack):
    def __init__(self, *linops: Iterable[LinearOperator], axis: int):
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
                                is_symmetric=bool(np.prod(self.is_symmetric_list).astype(bool)))

    def adjoint(self, y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.axis == 0:
            y_split = np.split(y, self.sections)
            result = 0
            for i, linop in enumerate(self.linops):
                result += linop.adjoint(y_split[i])
            return result
        else:
            out_list = [linop.adjoint(y) for linop in self.linops]
            return np.concatenate(out_list, axis=0)


class LinOpVStack(LinOpStack):
    def __init__(self, *linops: Iterable[LinearOperator]):
        super(LinOpVStack, self).__init__(*linops, axis=0)


class LinOpHStack(LinOpStack):
    def __init__(self, *linops: Iterable[LinearOperator]):
        super(LinOpHStack, self).__init__(*linops, axis=1)
