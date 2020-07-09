import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Any
from numbers import Number
from pycsou.core.linop import LinearOperator, ConstantMap
from pycsou.util.misc import is_range_broadcastable, range_broadcast_shape


class Map(ABC):
    def __init__(self, shape: Tuple[int, int], is_linear: bool = False, is_differentiable: bool = False):
        if len(shape) > 2:
            raise NotImplementedError(
                'Shapes of map objects must be tuples of length 2 (tensorial maps not supported).')
        self.shape = shape
        self.is_linear = is_linear
        self.is_functional = True if self.shape[0] == 1 else False
        self.is_differentiable = is_differentiable
        super(Map, self).__init__()

    @abstractmethod
    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        pass

    def __add__(self, other: 'Map') -> 'MapSum':
        if isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map']) -> 'MapComp':
        if isinstance(other, Number):
            other = ConstantMap(other)

        if isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __matmul__(self, other: 'Map') -> 'MapComp':
        return self.__mul__(other)

    def __neg__(self) -> 'MapComp':
        return self.__mul__(-1)

    def __sub__(self, other: 'Map') -> 'MapSum':
        other = other.__neg__()
        return self.__add__(other)

    def __pow__(self, power: int) -> 'MapComp':
        if type(power) is int:
            exp_map = self
            for i in range(power):
                exp_map = self.__mul__(exp_map)
            return exp_map
        else:
            raise NotImplementedError

    def __truediv__(self, other: Number) -> 'MapComp':
        if isinstance(other, Number):
            return self.__mul__(1 / other)
        else:
            raise NotImplementedError


class MapSum(Map):
    def __init__(self, map1: Map, map2: Map):
        if not is_range_broadcastable(map1.shape, map2.shape):
            raise ValueError('Cannot sum two maps with inconsistent range or domain sizes.')
        else:
            super(MapSum, self).__init__(shape=range_broadcast_shape(map1.shape, map2.shape),
                                         is_linear=map1.is_linear & map2.is_linear,
                                         is_differentiable=map1.is_differentiable & map2.is_differentiable)
            self.map1, self.map2 = map1, map2

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.map1(arg) + self.map2(arg)


class MapComp(Map):
    def __init__(self, map1: Map, map2: Map):
        if map1.shape[0] != map2.shape[1] and map2.shape[1] != 1:
            raise ValueError('Cannot compose two maps with inconsistent range or domain sizes.')
        else:
            super(MapComp, self).__init__(shape=(map1.shape[1], map2.shape[0]),
                                          is_linear=map1.is_linear & map2.is_linear,
                                          is_differentiable=map1.is_differentiable & map2.is_differentiable)
            self.map1 = map1
            self.map2 = map2

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.map1(self.map2(arg))


class DifferentiableMap(Map):
    def __init__(self, shape=Tuple[int, int], is_linear: bool = False, lipschitz_cst: float = np.infty,
                 diff_lipschitz_cst: float = np.infty):
        super(DifferentiableMap, self).__init__(shape=shape, is_linear=is_linear, is_differentiable=True)
        self.lipschitz_cst = lipschitz_cst
        self.diff_lipschitz_cst = diff_lipschitz_cst

    @abstractmethod
    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        pass

    def gradient(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.jacobianT(arg)

    def compute_lipschitz_cst(self):
        pass

    def compute_diff_lipschitz_cst(self):
        pass

    def __add__(self, other: Union['Map', 'DifferentiableMap']) -> Union['MapSum', 'DiffMapSum']:
        if isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map', 'DifferentiableMap']) -> Union['MapComp', 'DiffMapComp']:
        if isinstance(other, DifferentiableMap):
            return DiffMapComp(self, other)
        elif isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError


class DiffMapSum(MapSum, DifferentiableMap):
    def __init__(self, map1: DifferentiableMap, map2: DifferentiableMap):
        MapSum.__init__(self, map1=map1, map2=map2)
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear,
                                   lipschitz_cst=self.map1.lipschitz_cst + self.map2.lipschitz_cst,
                                   diff_lipschitz_cst=self.map1.diff_lipschitz_cst + self.map2.diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map1.jacobianT(arg) + self.map2.jacobianT(arg)


class DiffMapComp(MapComp, DifferentiableMap):
    def __init__(self, map1: DifferentiableMap, map2: DifferentiableMap):
        MapComp.__init__(self, map1=map1, map2=map2)
        lipschitz_cst = self.map2.lipschitz_cst * self.map1.lipschitz_cst
        diff_lipschitz_cst = self.map1.diff_lipschitz_cst * self.map2.diff_lipschitz_cst * self.map2.lipschitz_cst
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear,
                                   lipschitz_cst=lipschitz_cst, diff_lipschitz_cst=diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map2.jacobianT(arg) * self.map1.jacobianT(self.map2(arg))
