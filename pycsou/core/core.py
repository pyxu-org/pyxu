import numpy.typing as npt
import numpy as np
import typing as ty
import types
from numbers import Number
import pycsou.util as pycutil

NDArray = npt.ArrayLike


class Property:
    _property_list = frozenset(
        ['apply', '_lipschitz', 'jacobianT', '_diff_lipschitz', 'single_valued', 'prox', 'adjoint'])

    @classmethod
    def properties(cls) -> ty.Set[str]:
        props = set(dir(cls))
        return set(props.intersection(cls._property_list))

    @classmethod
    def has(cls, prop: ty.Tuple[str, ...]) -> ty.List[bool]:
        return [True if p in dir(cls) else False for p in prop]

    def __add__(self: ty.Union['Map', 'DiffMap', 'DiffFunc', 'ProxFunc', 'LinOp'],
                other: ty.Union['Map', 'DiffMap', 'DiffFunc', 'ProxFunc', 'LinOp']) -> ty.Union[
        'Map', 'DiffMap', 'DiffFunc', 'LinOp']:
        if not isinstance(other, Map):
            raise NotImplementedError('Cannot add together the two objects.')
        valid_shapes, out_shape = pycutil.broadcasted_sum(self.shape, other.shape)
        if not valid_shapes:
            raise ValueError('Cannot sum two maps with inconsistent shapes.')
        shared_props = self.properties().intersection(other.properties())
        shared_props.discard('prox')
        for Op in _base_operators:
            if Op.properties() == shared_props:
                break
        if Op == LinOp:
            shared_props.discard('jacobianT')
        shared_props.discard('single_valued')
        out_op = Op(out_shape)
        for prop in shared_props:
            if prop in ['_lispchitz', '_diff_lipschitz']:
                setattr(out_op, prop, getattr(self, prop) + getattr(other, prop))
            else:
                sself = self

                def composite_method(self, arr: NDArray) -> ty.Union[NDArray, 'LinOp']:
                    return getattr(sself, prop)(arr) + getattr(other, prop)(arr)

                setattr(out_op, prop, types.MethodType(composite_method, out_op))
        return out_op.squeeze()

    def __mul__(self, other: 'Map') -> 'Map':
        if not isinstance(other, Map):
            try:
                out_op = self.__class__(self.shape)
                properties = out_op.properties()
                properties.discard('single_valued')
                if isinstance(out_op, LinOp):
                    properties.discard('jacobianT')
                sself = self
                for prop in properties:
                    if prop == 'prox':

                        def postcomp_prox(self, arr, tau):
                            return sself.prox(arr, tau * other)

                        out_op.prox = types.MethodType(postcomp_prox, out_op)
                    elif prop in ['_lispchitz', '_diff_lipschitz']:
                        setattr(out_op, prop, other * getattr(self, prop))
                    else:

                        def composite_method(self, arr: NDArray) -> ty.Union[NDArray, 'LinOp']:
                            return other * getattr(sself, prop)(arr)

                        setattr(out_op, prop, types.MethodType(composite_method, out_op))

            except ValueError:
                raise ValueError('Cannot add together the two objects.')
        else:
            pass
        return out_op.squeeze()

    def __rmul__(self, other: 'Map') -> 'Map':
        pass

    def __pow__(self, power: float) -> 'Map':
        pass

    def __neg__(self) -> 'Map':
        pass

    def argscale(self, scalar: ty.Union[float, 'UnitOp']) -> 'Map':
        pass

    def argshift(self, arr: NDArray) -> 'Map':
        pass


class SingledValued(Property):

    @property
    def single_valued(self):
        return True


class Apply(Property):

    def __init__(self):
        self._lipschitz = np.infty

    def __call__(self, arr: NDArray) -> NDArray:
        return self.apply(arr)

    def apply(self, arr: NDArray) -> NDArray:
        raise NotImplementedError

    def ndapply(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.apply, axis=axis, arr=arr)

    def lipschitz(self) -> float:
        return self._lipschitz


class Differential(Property):

    def __init__(self):
        self._diff_lipschitz = np.infty

    def diff_lipschitz(self) -> float:
        return self._diff_lipschitz

    def jacobianT(self, arr: NDArray) -> 'LinOp':
        raise NotImplementedError


class Gradient(Differential):

    def jacobianT(self, arr: NDArray) -> NDArray:
        raise NotImplementedError

    def gradient(self, arr: NDArray) -> NDArray:
        return self.jacobianT(arr)

    def ndgradient(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.gradient, axis=axis, arr=arr)


class Adjoint(Property):

    def adjoint(self, arr: NDArray) -> NDArray:
        raise NotImplementedError

    def ndadjoint(self, arr: NDArray, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.adjoint, axis=axis, arr=arr)


class Proximal(Property):

    def prox(self, arr: NDArray, tau: Number) -> NDArray:
        raise NotImplementedError

    def ndprox(self, arr: NDArray, tau: Number, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.prox, axis=axis, arr=arr, tau=tau)

    def fenchel_prox(self, arr: NDArray, sigma: Number) -> NDArray:
        return arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)

    def ndfenchel_prox(self, arr: NDArray, sigma: Number, axis: int = 0) -> NDArray:
        return np.apply_along_axis(func1d=self.fenchel_prox, axis=axis, arr=arr, sigma=sigma)


class Map(Apply):

    def __init__(self, shape: ty.Tuple[int, int]):
        if len(shape) > 2:
            raise NotImplementedError(
                'Shapes of map objects must be tuples of length 2 (tensorial maps not supported).')
        self.shape = shape
        super(Map, self).__init__()

    @property
    def dim(self):
        return self.shape[1]

    @property
    def codim(self):
        return self.shape[0]

    def squeeze(self) -> ty.Union['Map', 'Func']:
        return self._squeeze(out=Func)

    def _squeeze(self, out: type) -> ty.Union['Map', 'Func']:
        if self.shape[0] == 1:
            obj = out(self.shape)
            for prop in self.properties():
                setattr(obj, prop, getattr(self, prop))
        else:
            obj = self
        return obj


class DiffMap(Map, Differential):

    def __init__(self, shape: ty.Tuple[int, int]):
        Map.__init__(self, shape)
        Differential.__init__(self)

    def squeeze(self) -> ty.Union['DiffMap', 'DiffFunc']:
        return self._squeeze(out=DiffFunc)


class Func(Map, SingledValued):

    def __init__(self, shape: ty.Union[int, ty.Tuple[int, ...]]):
        shape = tuple(shape)
        if len(shape) == 1:
            shape = (1,) + shape
        else:
            if shape[0] > 1:
                raise ValueError('Functionals'' must be of the form (1,n).')
        Map.__init__(self, shape)


class ProxFunc(Func, Proximal):

    def __init__(self, shape: ty.Union[int, ty.Tuple[int, ...]]):
        Func.__init__(self, shape)

    def __add__(self, other):
        if isinstance(other, LinFunc):
            valid_shapes, out_shape = pycutil.broadcasted_sum(self.shape, other.shape)
            if not valid_shapes:
                raise ValueError('Cannot sum two maps with inconsistent shapes.')
            f = ProxFunc(out_shape)
            f._lipschitz = self._lipschitz + other._lipschitz
            sself = self

            def affine_sum_prox(self, arr, tau):
                return sself.prox(arr - tau * other.asarray(), tau)

            f.prox = types.MethodType(affine_sum_prox, f)
        else:
            f = Property.__add__(self, other)
        return f


class DiffFunc(Func, Gradient):

    def __init__(self, shape: ty.Union[int, ty.Tuple[int, ...]]):
        Func.__init__(self, shape)
        Differential.__init__(self)


class LinOp(DiffMap, Adjoint):

    def __init__(self, shape: ty.Tuple[int, int]):
        Map.__init__(self, shape)

    def squeeze(self) -> ty.Union['DiffMap', 'DiffFunc']:
        return self._squeeze(out=LinFunc)

    def jacobianT(self, arr: NDArray) -> 'LinOp':
        return self.H

    def transpose(self, arr: NDArray) -> NDArray:
        pass

    @property
    def H(self) -> 'LinOp':
        pass

    @property
    def T(self) -> 'LinOp':
        pass

    def eigvals(self, *args, **kwargs):
        pass

    def svdvals(self, *args, **kwargs):
        pass

    def asarray(self, xp: types.ModuleType = np, dtype: type = np.float64) -> NDArray:
        pass

    def __array__(self, dtype: type = np.float64):
        return self.asarray(xp=np, dtype=dtype)

    def Gram(self) -> 'LinOp':
        pass

    def CoGram(self) -> 'LinOp':
        pass

    def pinv(self, arr: NDArray, **kwargs) -> NDArray:
        pass

    def dagger(self, **kwargs) -> 'LinOp':
        pass


class LinFunc(DiffFunc, LinOp):

    def __init__(self, shape: ty.Union[int, ty.Tuple[int, ...]]):
        Func.__init__(self, shape)


_base_operators = frozenset([Map, DiffMap, Func, DiffFunc, ProxFunc, LinOp, LinFunc])
