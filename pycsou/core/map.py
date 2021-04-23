# #############################################################################
# map.py
# ======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Abstract classes for multidimensional nonlinear maps.
"""

import numpy as np
import joblib as job

from abc import ABC, abstractmethod
from typing import Union, Tuple
from numbers import Number
from pycsou.util.misc import is_range_broadcastable, range_broadcast_shape


class Map(ABC):
    r"""
    Base class for multidimensional maps.

    Any instance/subclass of this class must at least implement the abstract method ``__call__``.
    This class supports the following arithmetic operators ``+``, ``-``, ``*``, ``@``, ``**`` and ``/``, implemented with the
    class methods ``__add__``/``__radd__``, ``__sub__``/``__neg__``, ``__mul__``/``__rmul__``, ``__matmul__``, ``__pow__``, ``__truediv__``.
    Such arithmetic operators can be used to *add*, *substract*, *scale*, *compose*, *exponentiate* or *evaluate* ``Map`` instances.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L1Norm, SquaredL2Norm
       from pycsou.linop.conv import Convolve1D
       from scipy import signal

       x = np.arange(10)
       filter = signal.hann(5)
       filter[filter.size//2:] = 0
       f1 = L1Norm(dim=x.size)
       f2 = SquaredL2Norm(dim=x.size)
       L1 = Convolve1D(size=x.size, filter=filter)
       L2 = Convolve1D(size=x.size, filter=filter/2)

    Consider four maps: two nonlinear functionals :math:`f_1:\mathbb{R}^{10}\to \mathbb{R}`, :math:`f_2:\mathbb{R}^{10}\to \mathbb{R}`
    and two linear operators :math:`L_1:\mathbb{R}^{10}\to \mathbb{R}^{10}`, :math:`L_2:\mathbb{R}^{10}\to \mathbb{R}^{10}`.

    .. doctest::

       >>> print(f1.shape, f2.shape, L1.shape, L2.shape)
       (1, 10) (1, 10) (10, 10) (10, 10)

    We can combine linearly/compose the maps with consistent domains/ranges:

    .. doctest::

       >>> f3 = f1 / 3 + np.pi * f2
       >>> np.allclose(f3(x), f1(x) / 3 + np.pi * f2(x))
       True
       >>> L3 = L1 * 3 - (L2 ** 2) / 6
       >>> np.allclose(L3(x), L1(x) * 3 - (L2(L2(x))) / 6)
       True
       >>> f4 = f3 * L3
       >>> np.allclose(f4(x), f3(L3(x)))
       True

    Note that multiplying a map with an array is the same as evaluating the map at the array.

    .. doctest::

       >>> np.allclose(f3 * x, f3(x))
       True

    The multiplication operator ``@`` can also be used in place of ``*``, in compliance with Numpy's interface:

    .. doctest::

       >>> np.allclose(f3 * x, f3 @ x)
       True
       >>> np.allclose((f1 * L1)(x), (f1 @ L1)(x))
       True

    Finally, maps can be shifted via the method ``shifter``:

    .. doctest::

       >>> f5=f4.shifter(shift=2 * x)
       >>> np.allclose(f5(x), f4(x + 2 * x))
       True

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
        Union[Number, np.ndarray]
            Value of ``arg`` through the map.
        """
        pass

    def apply_along_axis(self, arr: np.ndarray, axis: int = 0) -> np.ndarray:
        r"""
        Apply the map to 1-D slices along the given axis.

        Parameters
        ----------
        arr: np.ndarray
            Input array.
        axis: int
            Axis along which ``arr`` is sliced.

        Returns
        -------
        np.ndarray
           The output array. The shape of the latter is identical to the shape of ``arr``, except along the specified axis dimension.
           This axis is removed, and replaced with new dimensions equal to ``self.shape[0]``.
           If ``self.shape[0]==1`` the output array will have one fewer dimensions than ``arr``.

        Raises
        ------
        ValueError
            If ``arr.shape[axis] != self.shape[1]``.

        Examples
        --------

        .. doctest::

            >>> from pycsou.linop import DownSampling
            >>> from pycsou.func import SquaredL2Norm
            >>> D=DownSampling(size=20, downsampling_factor=2)
            >>> arr=np.arange(200).reshape(20,2,5)
            >>> out = D.apply_along_axis(arr, axis=0)
            >>> out.shape
            (10, 2, 5)
            >>> ff = SquaredL2Norm(dim=2)
            >>> out = ff.apply_along_axis(arr, axis=1)
            >>> out.shape
            (20, 5)

        """
        if arr.shape[axis] != self.shape[1]:
            raise ValueError(
                f"Array size along specified axis and the map domain's dimension differ: {arr.shape[axis]} != {self.shape[1]}.")
        return np.apply_along_axis(func1d=self.__call__, axis=axis, arr=arr)

    def shifter(self, shift: Union[Number, np.ndarray]) -> 'MapShifted':
        r"""
        Returns a shifted version of the map.

        Parameters
        ----------
        shift: Union[Number, np.ndarray]
            Shift vector.

        Returns
        -------
        :py:class:`~pycsou.core.map.MapShifted`
            Shifted map.

        Notes
        -----
        Let ``A`` be a ``Map`` instance  and ``B=A.shifter(y)`` with ``y`` some vector in the domain of ``A``. Then we have:
        ``B(x)=A(x+y)``.
        """
        return MapShifted(map=self, shift=shift)

    def __add__(self, other: 'Map') -> 'MapSum':
        r"""
        Add a map with another map instance.

        Parameters
        ----------
        other: Union['Map', np.ndarray]
            The other map or array to be added.

        Returns
        -------
        :py:class:`~pycsou.core.map.MapSum`
            Sum of the map with another map.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a map.
        """

        # if isinstance(other, np.ndarray):
        #     return MapBias(self, bias=other)
        if isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other: 'Map') -> 'MapSum':
        r"""
        Add a map with another map instance when the latter is on the right hand side of the summation.

        Parameters
        ----------
        other: Union['Map', np.ndarray]
            The other map or array to be added.

        Returns
        -------
        :py:class:`~pycsou.core.map.MapSum`
            Sum of the map with another map.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a map.

        """

        # if isinstance(other, np.ndarray):
        #     return MapBias(self, bias=other)
        if isinstance(other, Map):
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
        :py:class:`~pycsou.core.map.MapComp`, np.ndarray
            Composition of the map with another map, or product with a scalar or array.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a scalar, a map or an array.

        See Also
        --------
        :py:func:`~pycsou.core.map.Map.__matmul__`, :py:func:`~pycsou.core.map.Map.__rmul__`

        """
        if isinstance(other, Number):
            from pycsou.linop.base import HomothetyMap

            other = HomothetyMap(constant=other, size=self.shape[1])

        if isinstance(other, np.ndarray):
            return self(other)
        elif isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other: Union[Number, 'Map']) -> 'MapComp':
        if isinstance(other, Number):
            from pycsou.linop.base import HomothetyMap

            other = HomothetyMap(constant=other, size=self.shape[0])

        if isinstance(other, Map):
            return MapComp(other, self)
        else:
            raise NotImplementedError

    def __matmul__(self, other: 'Map') -> 'MapComp':
        r"""Alias for :py:func:`~pycsou.core.map.Map.__mul__` offered to comply with Numpy's interface."""
        return self.__mul__(other)

    def __neg__(self) -> 'MapComp':
        r"""Negates a map. Alias for ``self.__mul__(-1)``."""
        return self.__mul__(-1)

    def __sub__(self, other: Union['Map', np.ndarray]) -> Union['MapSum', 'MapBias']:
        r"""Substracts a map, scalar, or array to another map. Alias for ``self.__add__(other.__neg__())``."""
        other = other.__neg__()
        return self.__add__(other)

    def __pow__(self, power: int) -> 'MapComp':
        r"""Raise a map to a certain ``power``. Alias for ``A*A*...*A`` with ``power`` multiplications."""
        if type(power) is int:
            exp_map = self
            for i in range(1, power):
                exp_map = self.__mul__(exp_map)
            return exp_map
        else:
            raise NotImplementedError

    def __truediv__(self, scalar: Number) -> 'MapComp':
        r"""Divides a map by a ``scalar``. Alias for ``self.__mul__(1 / scalar)``."""
        if isinstance(scalar, Number):
            return self.__mul__(1 / scalar)
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
            Map.__init__(self, shape=range_broadcast_shape(map1.shape, map2.shape),
                         is_linear=map1.is_linear & map2.is_linear,
                         is_differentiable=map1.is_differentiable & map2.is_differentiable)
            self.map1, self.map2 = map1, map2

    def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.map1(arg) + self.map2(arg)


# class MapBias(Map):
#     def __init__(self, map_: Map, bias: np.ndarray):
#         if not is_range_broadcastable(map_.shape, bias.shape):
#             raise ValueError('Inconsistent range sizes between map and bias.')
#         else:
#             super(MapBias, self).__init__(shape=map_.shape, is_linear=False, is_differentiable=map_.is_differentiable)
#             self.map, self.bias = map_, bias
#
#     def __call__(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
#         return self.map(arg) + self.bias


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
    r"""
    Base class for multidimensional differentiable maps.

    Any instance/subclass of this class must at least implement the abstract methods ``__call__`` and ``jacobianT``.
    The attributes ``lipschitz_cst``, ``diff_lipschitz_cst`` and the method ``jacobianT`` are automatically updated
    using standard differentiation rules when scaling, composing , summing or shifting differentiable maps.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.linop.diff import FirstDerivative

       x = np.arange(10)
       f = SquaredL2Norm(dim=x.size)
       L1 = FirstDerivative(size=x.size)
       L1.compute_lipschitz_cst()
       L2 = FirstDerivative(size=x.size, kind='centered')
       L2.compute_lipschitz_cst()

    Consider three differentiable maps: a nonlinear differentiable functional :math:`f:\mathbb{R}^{10}\to \mathbb{R}`,
    and two linear operators :math:`L_1:\mathbb{R}^{10}\to \mathbb{R}^{10}`, :math:`L_2:\mathbb{R}^{10}\to \mathbb{R}^{10}`.

    .. doctest::

       >>> print(f.lipschitz_cst, f.diff_lipschitz_cst)
       inf 2
       >>> print(np.round(L1.lipschitz_cst,1), np.round(L1.diff_lipschitz_cst,1))
       2.0 2.0
       >>> print(np.round(L2.lipschitz_cst,1), np.round(L2.diff_lipschitz_cst,1))
       1.5 1.5
       >>> L3 = 2 * L1 + (L2 ** 2) / 3
       >>> np.allclose(L3.lipschitz_cst, 2 * L1.lipschitz_cst + (L2.lipschitz_cst ** 2)/3)
       True
       >>> map_ = f * L1
       >>> np.allclose(map_.lipschitz_cst, f.lipschitz_cst * L1.lipschitz_cst)
       True
       >>> np.allclose(map_.jacobianT(x), L1.jacobianT(x) * f.jacobianT(L1(x)))
       True

    Notes
    -----
    This class supports the following arithmetic operators ``+``, ``-``, ``*``, ``@``, ``**`` and ``/``, implemented with the
    class methods ``__add__``/``__radd__``, ``__sub__``/``__neg__``, ``__mul__``/``__rmul__``, ``__matmul__``, ``__pow__``, ``__truediv__``.
    Such arithmetic operators can be used to *add*, *substract*, *scale*, *compose*, *exponentiate* or *evaluate* ``DifferentiableMap`` instances.

    """

    def __init__(self, shape: Tuple[int, int], is_linear: bool = False, lipschitz_cst: float = np.infty,
                 diff_lipschitz_cst: float = np.infty):
        r"""
        Parameters
        ----------
        shape: Tuple[int, int]
            Shape of the differentiable map.
        is_linear: bool
            Whether the differentiable map is linear or not.
        lipschitz_cst: float
            Lispchitz constant of the differentiable map if it exists/is known. Default to :math:`+\infty`.
        diff_lipschitz_cst: float
            Lispchitz constant of the derivative of the differentiable map if it exists/is known. Default to :math:`+\infty`.
        """
        super(DifferentiableMap, self).__init__(shape=shape, is_linear=is_linear, is_differentiable=True)
        self.lipschitz_cst = lipschitz_cst
        self.diff_lipschitz_cst = diff_lipschitz_cst

    @abstractmethod
    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        r"""
        Transpose of the Jacobian matrix of the differentiable map evaluated at ``arg``.
        
        Parameters
        ----------
        arg: Union[Number, np.ndarray]
            Point at which the transposed Jacobian matrix is evaluated.

        Returns
        -------
        ::py:class:`~pycsou.core.linop.LinearOperator`
            Linear operator associated to the transposed Jacobian matrix.
        """
        pass

    def gradient(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        r"""
        Alias for ``self.jacobianT``.
        """
        return self.jacobianT(arg)

    def compute_lipschitz_cst(self):
        r"""
        User-implemented method to compute the Lipschitz constant of the map.
        """
        pass

    def compute_diff_lipschitz_cst(self):
        r"""
        User-implemented method to compute the Lipschitz constant of the derivative of the map.
        """
        pass

    def shifter(self, shift: Union[Number, np.ndarray]) -> 'DiffMapShifted':
        r"""
        Returns a shifted version of the map.

        Parameters
        ----------
        shift: Union[Number, np.ndarray]
            Shift vector.

        Returns
        -------
        :py:class:`~pycsou.core.map.DiffMapShifted`
            Shifted map.

        See Also
        --------
        :py:meth:`~pycsou.core.map.Map.shifter`
        """
        return DiffMapShifted(map=self, shift=shift)

    def __add__(self, other: Union['Map', 'DifferentiableMap']) -> Union['MapSum', 'DiffMapSum']:
        # if isinstance(other, np.ndarray):
        #     return DiffMapBias(self, other)
        if isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other: Union['Map', 'DifferentiableMap']) -> Union['MapSum', 'DiffMapSum']:
        # if isinstance(other, np.ndarray):
        #     return DiffMapBias(self, other)
        if isinstance(other, DifferentiableMap):
            return DiffMapSum(self, other)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, 'Map', 'DifferentiableMap', np.ndarray]) \
            -> Union['MapComp', 'DiffMapComp', np.ndarray]:
        if isinstance(other, Number):
            from pycsou.linop.base import HomothetyMap

            other = HomothetyMap(constant=other, size=self.shape[1])

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
            from pycsou.linop.base import HomothetyMap

            other = HomothetyMap(constant=other, size=self.shape[0])

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


# class DiffMapBias(MapBias, DifferentiableMap):
#     def __init__(self, map_: DifferentiableMap, bias: np.ndarray):
#         MapBias.__init__(self, map_=map_, bias=bias)
#         DifferentiableMap.__init__(self, shape=self.shape, is_linear=False, lipschitz_cst=self.map.lipschitz_cst,
#                                    diff_lipschitz_cst=self.map.diff_lipschitz_cst)
#
#     def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
#         return self.map.jacobianT(arg) * self.map2.jacobianT(arg) * self.map1.jacobianT(self.map2(arg))


class DiffMapComp(MapComp, DifferentiableMap):
    def __init__(self, map1: DifferentiableMap, map2: DifferentiableMap):
        from pycsou.linop.base import HomothetyMap

        MapComp.__init__(self, map1=map1, map2=map2)
        lipschitz_cst = self.map2.lipschitz_cst * self.map1.lipschitz_cst
        if isinstance(map1, HomothetyMap):
            diff_lipschitz_cst = self.map1.diff_lipschitz_cst * self.map2.diff_lipschitz_cst
        else:
            diff_lipschitz_cst = self.map1.diff_lipschitz_cst * self.map2.diff_lipschitz_cst * self.map2.lipschitz_cst
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear,
                                   lipschitz_cst=lipschitz_cst, diff_lipschitz_cst=diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> 'LinearOperator':
        return self.map2.jacobianT(arg) * self.map1.jacobianT(self.map2(arg))


class MapStack(Map):
    r"""
    Stack maps together.

    This class constructs a map by stacking multiple maps together, either **vertically** (``axis=0``) or **horizontally** (``axis=1``):

    - **Vertical stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N}\to \mathbb{R}^{M_i}, i=1,\ldots, k\}`
      of maps. Their vertical stacking is defined as the operator

      .. math::

         V:\begin{cases}\mathbb{R}^{N}\to \mathbb{R}^{M_1}\times \cdots \times\mathbb{R}^{M_k}\\
         \mathbf{x}\mapsto (L_1\mathbf{x},\ldots, L_k\mathbf{x}).
         \end{cases}

    - **Horizontal stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N_i}\to \mathbb{R}^{M}, i=1,\ldots, k\}`
      of maps. Their horizontal stacking is defined as the operator

      .. math::

         H:\begin{cases}\mathbb{R}^{N_1}\times \cdots \times\mathbb{R}^{N_k}\to \mathbb{R}^{M}\\
         (\mathbf{x}_1,\ldots, \mathbf{x}_k)\mapsto \sum_{i=1}^k L_i \mathbf{x}_i.
         \end{cases}

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.linop.diff import FirstDerivative
       from pycsou.core.map import MapStack

       x = np.arange(10)
       y = np.arange(20)
       f = SquaredL2Norm(dim=x.size)
       g = SquaredL2Norm(dim=y.size)
       A1 = FirstDerivative(size=x.size)
       A2 = FirstDerivative(size=y.size, kind='centered')
       A3 = A1 / 2
       K1 = f * A1
       K2 = g * A2
       K3 = A3

    Consider three maps :math:`K_1:\mathbb{R}^{10}\to \mathbb{R}`, :math:`K_2:\mathbb{R}^{20}\to \mathbb{R}` and
    :math:`K_3:\mathbb{R}^{10}\to \mathbb{R}^{10}`

    .. doctest::

       >>> V = MapStack(K1, K3, axis=0)
       >>> V.shape
       (11, 10)
       >>> np.allclose(V(x), np.concatenate((K1(x).flatten(), K3(x).flatten())))
       True
       >>> parV = MapStack(K1, K3, axis=0, n_jobs=-1)
       >>> np.allclose(V(x), parV(x))
       True
       >>> H = MapStack(K1,K2, axis=1)
       >>> H.shape
       (1, 30)
       >>> np.allclose(H(np.concatenate((x,y))), K1(x) + K2(y))
       True
       >>> parH = MapStack(K1,K2, axis=1, n_jobs=-1)
       >>> np.allclose(H(np.concatenate((x,y))), parH(np.concatenate((x,y))))
       True

    See Also
    --------
    :py:class:`~pycsou.core.map.MapVStack`, :py:class:`~pycsou.core.map.MapHStack`, :py:class:`~pycsou.core.map.DiffMapStack`

    """

    def __init__(self, *maps: Map, axis: int, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        maps: Map
            List of maps to stack.
        axis:
            Stacking direction: 0 for vertical and 1 for horizontal stacking.
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack.
            If ``n_jobs==1``, the map stack is evaluated sequentially, otherwise it is
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        self.maps = list(maps)
        if (np.abs(axis) > 1):
            ValueError('Axis must be one of {0, 1,-1}.')
        self.axis = int(axis)
        self.is_linear_list = [map_.is_linear for map_ in self.maps]
        self.is_differentiable_list = [map_.is_differentiable for map_ in self.maps]
        self.shapes = np.array([map_.shape for map_ in self.maps])
        self.block_sizes = [map_.shape[axis] for map_ in self.maps]
        self.sections = np.cumsum(self.block_sizes)
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend

        if not self.is_valid_stack():
            raise ValueError('Inconsistent map shapes for  stacking.')
        Map.__init__(self, shape=self.get_shape(),
                     is_linear=bool(np.prod(self.is_linear_list).astype(bool)),
                     is_differentiable=bool(np.prod(self.is_differentiable_list).astype(bool)))

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.axis == 0:
            if self.n_jobs == 1:
                out_list = [map_.__call__(x).flatten() for map_ in self.maps]
            else:
                with job.Parallel(backend=self.joblib_backend, n_jobs=self.n_jobs, verbose=False) as parallel:
                    out_list = parallel(job.delayed(map_.__call__)(x) for map_ in self.maps)
                out_list = [y.flatten() for y in out_list]
            return np.concatenate(out_list, axis=0)
        else:
            x_split = np.split(x, self.sections)
            if self.n_jobs == 1:
                result = 0
                for i, map_ in enumerate(self.maps):
                    result += map_.__call__(x_split[i])
            else:
                with job.Parallel(backend=self.joblib_backend, n_jobs=self.n_jobs, verbose=False) as parallel:
                    out_list = parallel(job.delayed(map_.__call__)(x_split[i])
                                        for i, map_ in enumerate(self.maps))
                    result = np.sum(np.stack(out_list, axis=0), axis=0)
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
    r"""
    Alias for vertical stacking, equivalent to ``MapStack(*maps, axis=0)``.

    Examples
    --------

    .. testsetup::

       from pycsou.core.map import MapVStack

    .. doctest::

       >>> V1 = MapStack(K1, K3, axis=0)
       >>> V2 = MapVStack(K1, K3)
       >>> np.allclose(V1(x), V2(x))
       True

    """

    def __init__(self, *maps: Map, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        maps: Map
            List of maps to stack.
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack.
            If ``n_jobs==1``, the map stack is evaluated sequentially, otherwise it is
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        super(MapVStack, self).__init__(*maps, axis=0, n_jobs=n_jobs, joblib_backend=joblib_backend)


class MapHStack(MapStack):
    r"""
    Alias for horizontal stacking, equivalent to ``MapStack(*maps, axis=1)``.

    Examples
    --------

    .. testsetup::

      from pycsou.core.map import MapHStack

    .. doctest::

      >>> V1 = MapStack(K1, K2, axis=1)
      >>> V2 = MapHStack(K1, K2)
      >>> np.allclose(V1(np.arange(V1.shape[1])), V2(np.arange(V1.shape[1])))
      True

    """

    def __init__(self, *maps: Map, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        maps: Map
            List of maps to stack.
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack.
            If ``n_jobs==1``, the map stack is evaluated sequentially, otherwise it is
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        super(MapHStack, self).__init__(*maps, axis=1, n_jobs=n_jobs, joblib_backend=joblib_backend)


class DiffMapStack(MapStack, DifferentiableMap):
    r"""
    Stack differentiable maps together.

    This class constructs a differentiable map by stacking multiple maps together, either **vertically** (``axis=0``) or **horizontally** (``axis=1``).
    The attributes ``lipschitz_cst``, ``diff_lipschitz_cst`` and the method ``jacobianT`` are inferred from those of
    the stacked maps.

    - **Vertical stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N}\to \mathbb{R}^{M_i}, i=1,\ldots, k\}`
      of maps, with Lipschitz constants :math:`\{\beta_i\in\mathbb{R}_+\cup\{+\infty\}, i=1,\ldots, k\}`, Jacobian matrices
      :math:`\{\mathbf{J}_i(\mathbf{x})\in\mathbb{R}^{M_i\times N},  i=1,\ldots, k\}` for some :math:`\mathbf{x}\in\mathbb{R}^N`.
      Then the vertically stacked operator

      .. math::

         V:\begin{cases}\mathbb{R}^{N}\to \mathbb{R}^{M_1}\times \cdots \times\mathbb{R}^{M_k}\\
         \mathbf{x}\mapsto (L_1\mathbf{x},\ldots, L_k\mathbf{x}).
         \end{cases}

      has a Lipschitz constant bounded by :math:`\sqrt{\sum_{i=1}^k \beta_i^2}`. Moreover the Jacobian matrix of :math:`V` is obtained by stacking the individual Jacobian matrices vertically:

      .. math::

         \mathbf{J}(\mathbf{x})=\left[\begin{array}{c}\mathbf{J}_1(\mathbf{x})\\\vdots\\ \mathbf{J}_k(\mathbf{x}) \end{array}\right]\in\mathbb{R}^{(\sum_{i=1}^k M_i)\times N}.

    - **Horizontal stacking**: Consider a collection :math:`\{L_i:\mathbb{R}^{N_i}\to \mathbb{R}^{M}, i=1,\ldots, k\}`
      of maps, with Lipschitz constants :math:`\{\beta_i\in\mathbb{R}_+\cup\{+\infty\}, i=1,\ldots, k\}`, Jacobian matrices
      :math:`\{\mathbf{J}_i(\mathbf{x}_i)\in\mathbb{R}^{M\times N_i},  i=1,\ldots, k\}` for some :math:`\mathbf{x}_i\in\mathbb{R}^{N_i}`.
      Then the horizontally stacked operator

      .. math::

         H:\begin{cases}\mathbb{R}^{N_1}\times \cdots \times\mathbb{R}^{N_k}\to \mathbb{R}^{M}\\
         (\mathbf{x}_1,\ldots, \mathbf{x}_k)\mapsto \sum_{i=1}^k L_i \mathbf{x}_i.
         \end{cases}

      has a Lipschitz constant bounded by :math:`\max_{i=1}^k \beta_i`. Moreover the Jacobian matrix of :math:`H` is obtained by stacking the individual Jacobian matrices horizontally:

      .. math::

         \mathbf{J}(\mathbf{x}_1,\ldots,\mathbf{x}_k)=\left[\begin{array}{c}\mathbf{J}_1(\mathbf{x}_1)&\cdots& \mathbf{J}_k(\mathbf{x}_k) \end{array}\right]\in\mathbb{R}^{M\times (\sum_{i=1}^k N_i)}.



    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.linop.diff import FirstDerivative
       from pycsou.core.map import MapStack, DiffMapStack

       x = np.arange(10)
       y = np.arange(20)
       f = SquaredL2Norm(dim=x.size)
       g = SquaredL2Norm(dim=y.size)
       A1 = FirstDerivative(size=x.size)
       A1.compute_lipschitz_cst()
       A2 = FirstDerivative(size=y.size, kind='centered')
       A2.compute_lipschitz_cst()
       A3 = A1 / 2
       K1 = f * A1
       K2 = g * A2
       K3 = A3

    .. doctest::

       >>> V = DiffMapStack(K1, K3, axis=0)
       >>> np.allclose(V.lipschitz_cst, np.sqrt(K1.lipschitz_cst ** 2 + K3.lipschitz_cst ** 2))
       True
       >>> parV = DiffMapStack(K1, K3, axis=0, n_jobs=-1)
       >>> np.allclose(V.jacobianT(x) * np.ones(shape=(V.jacobianT(x).shape[1])), parV.jacobianT(x) * np.ones(shape=(V.jacobianT(x).shape[1])))
       True

    See Also
    --------
    :py:class:`~pycsou.core.map.DiffMapVStack`, :py:class:`~pycsou.core.map.DiffMapHStack`, :py:class:`~pycsou.core.map.MapStack`

    """

    def __init__(self, *diffmaps: DifferentiableMap, axis: int, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        diffmaps: DifferentiableMap
            List of differentiable maps to be stacked
        axis: int
            Stacking direction: 0 for vertical and 1 for horizontal stacking.
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack and its Jacobian matrix.
            If ``n_jobs==1``, the map stack and its Jacobian matrix are evaluated sequentially, otherwise they are
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        MapStack.__init__(self, *diffmaps, axis=axis, n_jobs=n_jobs, joblib_backend=joblib_backend)

        if axis == 0:
            lipschitz_cst = np.sqrt(np.sum([diffmap.lipschitz_cst ** 2 for diffmap in self.maps]))
            diff_lipschitz_cst = np.sqrt(np.sum([diffmap.diff_lipschitz_cst ** 2 for diffmap in self.maps]))
        else:
            lipschitz_cst = np.max([diffmap.lipschitz_cst for diffmap in self.maps])
            diff_lipschitz_cst = np.max([diffmap.diff_lipschitz_cst for diffmap in self.maps])

        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear, lipschitz_cst=lipschitz_cst,
                                   diff_lipschitz_cst=diff_lipschitz_cst)

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union['LinOpHStack', 'LinOpVStack']:
        from pycsou.func.base import ExplicitLinearFunctional
        jacobianT_list = []
        if self.axis == 0:
            from pycsou.linop.base import LinOpVStack
            for diffmap in self.maps:
                jacobian = diffmap.jacobianT(arg)
                if isinstance(jacobian, np.ndarray):
                    jacobian = ExplicitLinearFunctional(jacobian, dtype=jacobian.dtype)
                jacobianT_list.append(jacobian)
            return LinOpVStack(*jacobianT_list, n_jobs=self.n_jobs, joblib_backend=self.joblib_backend)
        else:
            from pycsou.linop.base import LinOpHStack
            arg_split = np.split(arg, self.sections)
            for i, diffmap in enumerate(self.maps):
                jacobian = diffmap.jacobianT(arg_split[i])
                if isinstance(jacobian, np.ndarray):
                    jacobian = ExplicitLinearFunctional(jacobian, dtype=jacobian.dtype)
                jacobianT_list.append(jacobian)
            return LinOpHStack(*jacobianT_list, n_jobs=self.n_jobs, joblib_backend=self.joblib_backend)


class DiffMapVStack(DiffMapStack):
    r"""
    Alias for vertical stacking of differentiable maps, equivalent to ``DiffMapStack(*maps, axis=0)``.
    """

    def __init__(self, *diffmaps: DifferentiableMap, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        diffmaps: DifferentiableMap
            List of differentiable maps to be stacked
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack and its Jacobian matrix.
            If ``n_jobs==1``, the map stack and its Jacobian matrix are evaluated sequentially, otherwise they are
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        super(DiffMapVStack, self).__init__(*diffmaps, axis=0, n_jobs=n_jobs, joblib_backend=joblib_backend)


class DiffMapHStack(DiffMapStack):
    r"""
    Alias for horizontal stacking of differentiable maps, equivalent to ``DiffMapStack(*maps, axis=1)``.
    """

    def __init__(self, *diffmaps: DifferentiableMap, n_jobs: int = 1, joblib_backend: str = 'loky'):
        r"""
        Parameters
        ----------
        diffmaps: DifferentiableMap
            List of differentiable maps to be stacked.
        n_jobs: int
            Number of cores to be used for parallel evaluation of the map stack and its Jacobian matrix.
            If ``n_jobs==1``, the map stack and its Jacobian matrix are evaluated sequentially, otherwise they are
            evaluated in parallel. Setting ``n_jobs=-1`` uses all available cores.
        joblib_backend: str
            Joblib backend (`more details here <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_).

        """
        super(DiffMapHStack, self).__init__(*diffmaps, axis=1, n_jobs=n_jobs, joblib_backend=joblib_backend)


if __name__ == '__main__':
    pass
    # import numpy as np
    # from pycsou.linop.base import DenseLinearOperator, LinOpStack, LinOpVStack, LinOpHStack, HomothetyMap
    # from pycsou.linop.conv import Convolve1D
    # from scipy import signal
    #
    # x1 = np.arange(10)
    # x2 = np.arange(20)
    # filter = signal.hann(5)
    # filter[filter.size // 2:] = 0
    # L1 = Convolve1D(size=x1.size, filter=filter)
    # L2 = DenseLinearOperator(np.arange(x2.size * L1.shape[0]).reshape(L1.shape[0], x2.size))
    # L3 = LinOpStack(L1, L2, axis=1)
    # L3(np.concatenate((x1, x2)))
