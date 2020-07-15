import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Any, Iterable
from numbers import Number
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

    def __add__(self, other: Union['Map', np.ndarray]) -> 'MapSum':
        if isinstance(other, np.ndarray):
            return MapBias(self, bias=other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other: Union['Map', np.ndarray]) -> 'MapSum':
        if isinstance(other, np.ndarray):
            return MapBias(self, bias=other)
        elif isinstance(other, Map):
            return MapSum(other, self)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map']) -> 'MapComp':
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other: Union[Number, 'Map']) -> 'MapComp':
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, Map):
            return MapComp(other, self)
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


class MapBias(Map):
    def __init__(self, map_: Map, bias: Union[Number, np.ndarray]):
        if not is_range_broadcastable(map_.shape, bias.shape):
            raise ValueError('Inconsistent range sizes between map and bias.')
        else:
            super(MapBias, self).__init__(shape=map_.shape, is_linear=False, is_differentiable=map_.is_differentiable)
            self.map, self.bias = map_, bias

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.map(arg) + self.bias


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

    def __add__(self, other: Union['Map', 'DifferentiableMap', np.ndarray]) -> Union['MapSum', 'DiffMapSum']:
        if isinstance(other, np.ndarray):
            return DiffMapBias(self, other)
        elif isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other: Union['Map', 'DifferentiableMap', np.ndarray]) -> Union['MapSum', 'DiffMapSum']:
        if isinstance(other, np.ndarray):
            return DiffMapBias(self, other)
        elif isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map', 'DifferentiableMap']) -> Union['MapComp', 'DiffMapComp']:
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, DifferentiableMap):
            return DiffMapComp(self, other)
        elif isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other: Union[Number, 'Map', 'DifferentiableMap']) -> Union['MapComp', 'DiffMapComp']:
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, DifferentiableMap):
            return DiffMapComp(other, self)
        elif isinstance(other, Map):
            return MapComp(other, self)
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


class DiffMapBias(MapBias, DifferentiableMap):
    def __init__(self, map_: DifferentiableMap, bias: np.ndarray):
        MapBias.__init__(self, map_=map_, bias=bias)
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=False, lipschitz_cst=self.map.lipschitz_cst,
                                   diff_lipschitz_cst=self.map.diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map.jacobianT(arg)


class DiffMapComp(MapComp, DifferentiableMap):
    def __init__(self, map1: DifferentiableMap, map2: DifferentiableMap):
        MapComp.__init__(self, map1=map1, map2=map2)
        lipschitz_cst = self.map2.lipschitz_cst * self.map1.lipschitz_cst
        diff_lipschitz_cst = self.map1.diff_lipschitz_cst * self.map2.diff_lipschitz_cst * self.map2.lipschitz_cst
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear,
                                   lipschitz_cst=lipschitz_cst, diff_lipschitz_cst=diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map2.jacobianT(arg) * self.map1.jacobianT(self.map2(arg))


class MapStack(Map):
    def __init__(self, *maps: Iterable[Map], axis: int):
        self.maps = list(*maps)
        if (np.abs(axis) > 1):
            ValueError('Axis must be one of {0, 1,-1}.')
        self.axis = int(axis)
        self.is_linear_list = [map_.is_linear for map_ in self.maps]
        self.is_differentiable_list = [map_.is_differentiable for map_ in self.maps]
        self.shapes = np.array([map_.shape for map_ in self.maps])
        self.block_sizes = [map_.shape[axis] for map_ in self.maps]
        self.sections = np.cumsum(self.block_sizes)

        if not self.is_valid_stack():
            raise ValueError('Inconsistent map shapes for  stacking.')
        super(MapStack, self).__init__(shape=self.get_shape(),
                                       is_linear=bool(np.prod(self.is_linear_list).astype(bool)),
                                       is_differentiable=bool(np.prod(self.is_differentiable_list).astype(bool)))

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.axis == 0:
            out_list = [map_.__call__(x) for map_ in self.maps]
            return np.concatenate(out_list, axis=0)
        else:
            x_split = np.split(x, self.sections)
            result = 0
            for i, map_ in enumerate(self.maps):
                result += map_.__call__(x_split[i])
            return result

    def is_valid_stack(self) -> bool:
        col_sizes = [map_.shape[1 - self.axis] for map_ in self.maps]
        return np.unique(col_sizes).size == 1

    def get_shape(self) -> Tuple[int, int]:
        sizes = [map_.shape[self.axis] for map_ in self.maps]
        if self.axis == 0:
            return (int(np.sum(sizes).astype(int)), self.maps[0].shape[1 - self.axis])
        else:
            return (self.maps[0].shape[1 - self.axis], int(np.sum(sizes).astype(int)))


class MapVStack(MapStack):
    def __init__(self, *maps: Iterable[Map]):
        super(MapVStack, self).__init__(*maps, axis=0)


class MapHStack(MapStack):
    def __init__(self, *maps: Iterable[Map]):
        super(MapHStack, self).__init__(*maps, axis=1)


class DiffMapStack(MapStack, DifferentiableMap):
    def __init__(self, *diffmaps: Iterable[DifferentiableMap], axis: int):
        MapStack.__init__(self, *diffmaps, axis=axis)
        lipschitz_cst = np.sqrt(np.sum([diffmap.lipschitz_cst ** 2 for diffmap in self.maps]))
        diff_lipschitz_cst = np.sqrt(np.sum([diffmap.diff_lipschitz_cst ** 2 for diffmap in self.maps]))
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear, lipschitz_cst=lipschitz_cst,
                                   diff_lipschitz_cst=diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union['LinOpHStack', 'LinOpHStack']:
        if self.axis == 0:
            from pycsou.core.linop import LinOpHStack
            jacobianT_list = [diffmap.jacobianT(arg) for diffmap in self.maps]
            return LinOpHStack(*jacobianT_list)
        else:
            from pycsou.core.linop import LinOpVStack
            arg_split = np.split(arg, self.sections)
            jacobianT_list = [diffmap.jacobianT(arg_split[i]) for i, diffmap in enumerate(self.maps)]
            return LinOpVStack(*jacobianT_list)


class DiffMapVStack(DiffMapStack):
    def __init__(self, *diffmaps: Iterable[DifferentiableMap]):
        super(DiffMapVStack, self).__init__(*diffmaps, axis=0)


class DiffMapHStack(DiffMapStack):
    def __init__(self, *diffmaps: Iterable[DifferentiableMap]):
        super(DiffMapHStack, self).__init__(*diffmaps, axis=1)
