import numbers as nb
import types
import typing as typ

import numpy as np

import pycsou.abc
import pycsou.util as pycutil

NDArray = pycsou.abc.NDArray


class Property:
    @classmethod
    def _property_list(cls):
        return frozenset(
            ("apply", "_lipschitz", "jacobian", "_diff_lipschitz", "single_valued", "gradient", "prox", "adjoint")
        )

    @classmethod
    def properties(cls) -> typ.Set[str]:
        props = set(dir(cls))
        return set(props.intersection(cls._property_list()))

    @classmethod
    def has(cls, prop: typ.Tuple[str, ...]) -> bool:
        return set(prop) <= cls.properties()

    def __add__(self: "Map", other: "Map") -> "Map":
        if not isinstance(other, Map):
            raise NotImplementedError(f"Cannot add object of type {type(self)} with object of type {type(other)}.")
        valid_shapes, out_shape = pycutil.broadcast_sum_shapes(self.shape, other.shape)
        if not valid_shapes:
            raise ValueError(f"Cannot sum two maps with inconsistent shapes {self.shape} and {other.shape}.")
        shared_props = self.properties() & other.properties()
        shared_props.discard("prox")
        for Op in _base_operators:
            if Op.properties() == shared_props:
                break
        if Op in [LinOp, DiffFunc, LinFunc]:
            shared_props.discard("jacobian")
        shared_props.discard("single_valued")
        out_op = Op(out_shape)
        for prop in shared_props:
            if prop in ["_lispchitz", "_diff_lipschitz"]:
                setattr(out_op, prop, getattr(self, prop) + getattr(other, prop))
            else:

                def composite_method(obj, arr: NDArray) -> typ.Union[NDArray, "LinOp"]:
                    return getattr(self, prop)(arr) + getattr(other, prop)(arr)

                setattr(out_op, prop, types.MethodType(composite_method, out_op))
        return out_op.squeeze()

    def __mul__(self: "Map", other: typ.Union["Map", nb.Number]) -> "Map":
        if isinstance(other, nb.Number):
            hmap = _HomothetyOp(other, dim=self.shape[0])
            return hmap.__mul__(self)
        elif not isinstance(other, Map):
            raise NotImplementedError(f"Cannot multiply object of type {type(self)} with object of type {type(other)}.")
        if self.shape[1] == other.shape[0]:
            out_shape = (self.shape[0], other.shape[1])
        else:
            raise ValueError(f"Cannot compose two maps with inconsistent shapes {self.shape} and {other.shape}.")
        shared_props = self.properties() & other.properties()
        shared_props.discard("prox")
        for Op in _base_operators:
            if Op.properties() == shared_props:
                break
        if Op in [LinOp, DiffFunc, LinFunc]:
            shared_props.discard("jacobian")
        shared_props.discard("single_valued")
        out_op = Op(out_shape)
        for prop in shared_props:
            if prop in ["_lispchitz", "_diff_lipschitz"]:
                setattr(out_op, prop, getattr(self, prop) + getattr(other, prop))
            else:

                def composite_method(obj, arr: NDArray) -> typ.Union[NDArray, "LinOp"]:
                    return getattr(self, prop)(arr) + getattr(other, prop)(arr)

                setattr(out_op, prop, types.MethodType(composite_method, out_op))
        return out_op.squeeze()

    def __rmul__(self, other: nb.Number) -> "Map":
        return self.__mul__(other)

    def __pow__(self, power: float) -> "Map":
        pass

    def __neg__(self) -> "Map":
        pass

    def argscale(self, scalar: typ.Union[float, "UnitOp"]) -> "Map":
        pass

    def argshift(self, arr: NDArray) -> "Map":
        pass


class SingledValued(Property):
    def single_valued(self):
        return True


class Apply(Property):
    def __call__(self, arr: NDArray) -> NDArray:
        return self.apply(arr)

    def apply(self, arr: NDArray) -> NDArray:
        raise NotImplementedError

    def lipschitz(self) -> float:
        raise NotImplementedError


class Differential(Property):
    def diff_lipschitz(self) -> float:
        raise NotImplementedError

    def jacobian(self, arr: NDArray) -> "LinOp":
        raise NotImplementedError


class Gradient(Differential):
    def jacobian(self, arr: NDArray) -> "LinOp":
        from pycsou.linop.base import ExplicitLinFunc

        return ExplicitLinFunc(self.gradient(arr))

    def gradient(self, arr: NDArray) -> NDArray:
        raise NotImplementedError


class Adjoint(Property):
    def adjoint(self, arr: NDArray) -> NDArray:
        raise NotImplementedError


class Proximal(Property):
    def prox(self, arr: NDArray, tau: float) -> NDArray:
        raise NotImplementedError

    def fenchel_prox(self, arr: NDArray, sigma: float) -> NDArray:
        return arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)


class Map(Apply):
    def __init__(self, shape: typ.Tuple[int, int]):
        if len(shape) > 2:
            raise NotImplementedError(
                "Shapes of map objects must be tuples of length 2 (tensorial maps not supported)."
            )
        self._shape = shape
        self._lipschitz = np.infty

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return self.shape[1]

    @property
    def codim(self):
        return self.shape[0]

    def squeeze(self) -> typ.Union["Map", "Func"]:
        return self._squeeze(out=Func)

    def _squeeze(self, out: type) -> typ.Union["Map", "Func"]:
        if self.shape[0] == 1:
            obj = self.specialize(cast_to=out)
        else:
            obj = self
        return obj

    def lipschitz(self) -> float:
        return self._lipschitz

    def specialize(self, cast_to: type) -> "Map":
        if cast_to == self.__class__:
            obj = self
        else:
            if self.properties() > cast_to.properties():
                raise ValueError(
                    f"Cannot specialize an object of type {self.__class__} to an object of type {cast_to}."
                )
            obj = cast_to(self.shape)
            for prop in self.properties():
                if prop == "jacobian" and cast_to.has("single_valued"):
                    obj.gradient = types.MethodType(lambda _, x: self.jacobian(x).asarray().reshape(-1), obj)
                else:
                    setattr(obj, prop, getattr(self, prop))
        return obj


class DiffMap(Map, Differential):
    def __init__(self, shape: typ.Tuple[int, int]):
        super(DiffMap, self).__init__(shape)
        self._diff_lipschitz = np.infty

    def squeeze(self) -> typ.Union["DiffMap", "DiffFunc"]:
        return self._squeeze(out=DiffFunc)

    def diff_lipschitz(self) -> float:
        return self._diff_lipschitz


class Func(Map, SingledValued):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        shape = tuple(shape)
        if len(shape) == 1:
            shape = (1,) + shape
        else:
            if shape[0] > 1:
                raise ValueError("Functionals" " must be of the form (1,n).")
        super(Func, self).__init__(shape)


class ProxFunc(Func, Proximal):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        super(ProxFunc, self).__init__(shape)

    def __add__(self: "ProxFunc", other: "Map") -> "Map":
        if isinstance(other, LinFunc):
            valid_shapes, out_shape = pycutil.broadcast_sum_shapes(self.shape, other.shape)
            if not valid_shapes:
                raise ValueError(f"Cannot sum two maps with inconsistent shapes {self.shape} and {other.shape}.")
            f = ProxFunc(out_shape)
            f.apply = types.MethodType(lambda _, x: self.apply(x) + other.apply(x), f)
            f._lipschitz = self._lipschitz + other._lipschitz
            f.prox = types.MethodType(lambda _, x, tau: self.prox(x - tau * other.asarray(), tau), f)
        else:
            f = Property.__add__(self, other)
        return f


class DiffFunc(Func, Gradient):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        super(DiffFunc, self).__init__(shape)


class LinOp(DiffMap, Adjoint):
    def __init__(self, shape: typ.Tuple[int, int]):
        super(LinOp, self).__init__(shape)
        self._diff_lipschitz = 0

    def squeeze(self) -> typ.Union["LinOp", "LinFunc"]:
        return self._squeeze(out=LinFunc)

    def jacobian(self, arr: NDArray) -> "LinOp":
        return self

    @property
    def T(self) -> "LinOp":
        pass

    def svdvals(self, *args, **kwargs):
        pass

    def asarray(self, xp: types.ModuleType = np, dtype: type = np.float64, **kwargs) -> NDArray:
        pass

    def __array__(self, dtype: type = np.float64) -> np.ndarray:
        return self.asarray(xp=np, dtype=dtype)

    def gram(self) -> "LinOp":
        pass

    def cogram(self) -> "LinOp":
        pass

    def pinv(self, arr: NDArray, **kwargs) -> NDArray:
        pass

    def dagger(self, **kwargs) -> "LinOp":
        pass


class LinFunc(DiffFunc, LinOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        DiffFunc.__init__(self, shape)
        LinOp.__init__(self, shape)

    def __add__(self, other):
        return ProxFunc.__add__(other, self)


class SquareOp(LinOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        shape = tuple(shape)
        if len(shape) > 1 and (shape[0] != shape[1]):
            raise ValueError(f"Inconsistent shape {shape} for operator of type {SquareOp}")
        super(SquareOp, self).__init__(shape=(shape[0], shape[0]))


class NormalOp(SquareOp):
    def eigvals(self, *args, **kwargs):
        pass

    def cogram(self) -> "NormalOp":
        return self.gram().specialize(cast_to=SelfAdjointOp)


class SelfAdjointOp(NormalOp):
    def adjoint(self, arr: NDArray) -> NDArray:
        return self.apply(arr)

    @property
    def T(self) -> "SelfAdjointOp":
        return self


class UnitOp(NormalOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        super(UnitOp, self).__init__(shape)
        self._lipschitz = 1

    def lipschitz(self) -> float:
        return self._lipschitz

    def pinv(self, arr: NDArray, **kwargs) -> NDArray:
        return self.adjoint(arr)

    def dagger(self, **kwargs) -> "UnitOp":
        return self.H


class ProjOp(SquareOp):
    def __pow__(self, power: int) -> typ.Union["ProjOp", "UnitOp"]:
        if power == 0:
            from pycsou.linop.base import IdentityOperator

            return IdentityOperator(self.shape)
        else:
            return self


class OrthProjOp(ProjOp, SelfAdjointOp):
    def __init__(self, shape: typ.Union[int, typ.Tuple[int, ...]]):
        super(OrthProjOp, self).__init__(shape)
        self._lipschitz = 1

    def lipschitz(self) -> float:
        return self._lipschitz

    def pinv(self, arr: NDArray, **kwargs) -> NDArray:
        return self.apply(arr)

    def dagger(self, **kwargs) -> "OrthProjOp":
        return self


class PosDefOp(SelfAdjointOp):
    pass


class _HomothetyOp(SelfAdjointOp):
    def __init__(self, cst: nb.Number, dim: int):
        if not isinstance(cst, nb.Number):
            raise ValueError("Argument [cst] must be a number.")
        super(_HomothetyOp, self).__init__(shape=(dim, dim))
        self._cst = cst
        self._lipschitz = np.abs(cst)
        self._diff_lipschitz = 0

    def apply(self, arr: NDArray) -> NDArray:
        return (self._cst * arr).astype(arr.dtype)

    def adjoint(self, arr: NDArray) -> NDArray:
        return (self._cst * arr).astype(arr.dtype)


_base_operators = frozenset([Map, DiffMap, Func, DiffFunc, ProxFunc, LinOp, LinFunc])
