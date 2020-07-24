# #############################################################################
# map.py
# ======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Abstract classes for multidimensional nonlinear maps.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Any, Iterable
from numbers import Number
from pycsou.util.misc import is_range_broadcastable, range_broadcast_shape


class Map(ABC):
    r"""
    Base class for multidimensional maps.
    Any instance of this class must at least implement the abstract method ``__call__``.
    """
    def __init__(self, shape: Tuple[int, int], is_linear: bool = False, is_differentiable: bool = False):
        r"""
        Parameters
        ----------
        shape: Tuple[int, int]
            Shape of the map.
        is_linear: bool
            Whether the map is linear or not.
        is_differentiable: bool
            Whether the map is differentiable or not.
        """
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
        r"""
        Call self as a function.
        Parameters
        ----------
        arg: Union[Number, np.ndarray]
            Argument of the map.
        Returns
        -------
            Value of ``arg`` through the map.
        """
        pass

    def shifter(self, shift: Union[Number, np.ndarray]) -> 'MapShifted':
        r"""
        Returns a shifted version of the map.

        Parameters
        ----------
        shift: Union[Number, np.ndarray]
            Shift vector.

        Returns
        -------
        py:class:`~pycsou.core.map.MapShifted`
            Shifted map.
        """
        return MapShifted(map=self, shift=shift)

    def __add__(self, other: Union['Map', np.ndarray]) -> Union['MapSum', 'MapBias']:
        r"""
        Add a map with another map instance or array.

        Parameters
        ----------
        other: Union['Map', np.ndarray]
            The other map or array to be added.

        Returns
        -------
        py:class:`~pycsou.core.map.MapSum` if ``isinstance(other, Map)``
        py:class:`~pycsou.core.map.MapBias` otherwise.
            Sum of the map with another map or array.

        Raises
        ------
        NotImplementedError if ``other`` is not a map or an array.
        """
        if isinstance(other, np.ndarray):
            return MapBias(self, bias=other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other: Union['Map', np.ndarray]) -> 'MapSum':
        r"""
        Add a map with another map instance or array when the latter are on the right hand side of the summation.

        Parameters
        ----------
        other: Union['Map', np.ndarray]
            The other map or array to be added.

        Returns
        -------
        py:class:`~pycsou.core.map.MapSum` if ``isinstance(other, Map)``
        py:class:`~pycsou.core.map.MapBias` otherwise.
            Sum of the map with another map or array.

        Raises
        ------
        NotImplementedError if ``other`` is not a map or an array.

        """
        if isinstance(other, np.ndarray):
            return MapBias(self, bias=other)
        elif isinstance(other, Map):
            return MapSum(other, self)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map', np.ndarray]) -> Union['MapComp', np.ndarray]:
        r"""
        Multiply a map with another map, a scalar, or an array.

        The behaviour of this method depends on the type of ``other``:
        * If ``other`` is another map, then it returns the composition of the map with ``other``.
        * If ``other`` is an array with compatible shape, then it calls the map on ``other``.
        * If ``other`` is a scalar, then it multiplies the map with this scalar.

        Parameters
        ----------
        other: Union[Number, 'Map', np.ndarray]
            Scalar, map or array with which to multiply.

        Returns
        -------
        Union[py:class:`~pycsou.core.map.MapComp`,np.ndarray]
            Composition of the map with another map, or product with a scalar or array.

        Raises
        ------
        NotImplementedError if ``other`` is not a scalar, a map or an array.

        See Also
        --------
        :py:func:`~pycsou.core.map.Map.__matmul__`, :py:func:`~pycsou.core.map.Map.__rmul__`

        """
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, np.ndarray):
            return self(other)
        elif isinstance(other, Map):
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


class MapShifted(Map):
    def __init__(self, map: Map, shift: Union[Number, np.ndarray]):
        self.map = map
        self.shift = shift
        if shift.size != map.shape[1]:
            raise TypeError('Invalid shift size.')
        Map.__init__(self, shape=map.shape, is_linear=map.is_linear, is_differentiable=map.is_differentiable)

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.map(arg + self.shift)


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
        if map1.shape[1] != map2.shape[0] and map2.shape[0] != 1 and map1.shape[1] != 1:
            raise ValueError('Cannot compose two maps with inconsistent range or domain sizes.')
        else:
            Map.__init__(self, shape=(map1.shape[0], map2.shape[1]),
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

    def shifter(self, shift: Union[Number, np.ndarray]):
        return DiffMapShifted(map=self, shift=shift)

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

    def __mul__(self, other: Union[Number, 'Map', 'DifferentiableMap', np.ndarray]) \
            -> Union['MapComp', 'DiffMapComp', np.ndarray]:
        if isinstance(other, Number):
            from pycsou.core.linop import HomothetyMap
            other = HomothetyMap(constant=other)

        if isinstance(other, np.ndarray):
            return self(other)
        elif isinstance(other, DifferentiableMap):
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


class DiffMapShifted(MapShifted, DifferentiableMap):
    def __init__(self, map: DifferentiableMap, shift: Union[Number, np.ndarray]):
        MapShifted.__init__(self, map=map, shift=shift)
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear,
                                   lipschitz_cst=self.map.lipschitz_cst,
                                   diff_lipschitz_cst=self.map.diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map.jacobianT(arg + self.shift)


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
        return self.map.jacobianT(arg) * self.map2.jacobianT(arg) * self.map1.jacobianT(self.map2(arg))


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

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union['LinOpHStack', 'LinOpVStack']:
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
