from pycsou.core.map import Map, DifferentiableMap, MapSum, MapComp, MapHStack
from abc import abstractmethod
from pycsou.core.linop import LinearOperator, UnitaryOperator
from typing import Union, Tuple, Optional, Any, Iterable, Callable
from numbers import Number
import numpy as np
import warnings


class Functional(Map):
    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_differentiable: bool = False,
                 is_linear: bool = False):
        super(Functional, self).__init__(shape=(1, dim), is_differentiable=is_differentiable, is_linear=is_linear)
        self.data = data
        self.dim = dim


class DifferentiableFunctional(Functional, DifferentiableMap):
    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_linear: bool = False,
                 lipschitz_cst: float = np.infty, diff_lipschitz_cst: float = np.infty):
        Functional.__init__(self, dim=dim, data=data, is_differentiable=True, is_linear=is_linear)
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear, lipschitz_cst=lipschitz_cst,
                                   diff_lipschitz_cst=diff_lipschitz_cst)


class LinearFunctional(Functional, LinearOperator):
    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, dtype: Optional[type] = None,
                 is_explicit: bool = False, is_dense: bool = False,
                 is_sparse: bool = False, is_dask: bool = False, is_symmetric: bool = False):
        Functional.__init__(self, dim=dim, data=data, is_differentiable=True, is_linear=True)
        LinearOperator.__init__(self, shape=self.shape, dtype=dtype, is_explicit=is_explicit, is_dense=is_dense,
                                is_sparse=is_sparse, is_dask=is_dask, is_symmetric=is_symmetric)


class ProximableFunctional(Functional):
    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_differentiable: bool = False,
                 is_linear: bool = False):
        if is_differentiable == True or is_linear == True:
            warnings.warn(
                'For differentiable and/or linear maps, consider the dedicated classes DifferentiableMap and LinearOperator.')
        super(ProximableFunctional, self).__init__(dim=dim, data=data, is_differentiable=is_differentiable,
                                                   is_linear=is_linear)

    @abstractmethod
    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        pass

    def fenchel_prox(self, z: Union[Number, np.ndarray], sigma: Number) -> Union[Number, np.ndarray]:
        return z - sigma * self.prox(x=z / sigma, tau=1 / sigma)

    def __add__(self, other: Union[Map, LinearFunctional]) -> Union[MapSum, 'ProxFuncAffineSum']:
        if isinstance(other, LinearFunctional):
            return ProxFuncAffineSum(self, linear_part=other, intercept=0)
        elif isinstance(other, Map):
            return MapSum(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other: Union[Number, Map, UnitaryOperator, np.ndarray]) \
            -> Union[MapComp, 'ProxFuncPreComp', 'ProxFuncPreCompUnitOp']:
        if isinstance(other, Number) or isinstance(other, np.ndarray):
            return ProxFuncPreComp(self, scale=other, shift=0)
        elif isinstance(other, UnitaryOperator):
            return ProxFuncPreCompUnitOp(self, other)
        elif isinstance(other, Map):
            return MapComp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other: Union[Number, Map]) -> Union[MapComp, 'ProxFuncPostComp']:
        if isinstance(other, Number):
            return ProxFuncPostComp(self, scale=other, shift=0)
        elif isinstance(other, Map):
            return MapComp(other, self)
        else:
            raise NotImplementedError


class ProxFuncPostComp(ProximableFunctional):
    def __init__(self, prox_func: ProximableFunctional, scale: Number, shift: Number):
        super(ProxFuncPostComp, self).__init__(dim=prox_func.dim, data=prox_func.data,
                                               is_differentiable=prox_func.is_differentiable)
        self.prox_func = prox_func
        self.scale = scale
        self.shift = shift

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return self.scale * self.prox_func.__call__(x) + self.shift

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return self.prox_func.prox(x, tau * self.scale)


class ProxFuncAffineSum(ProximableFunctional):
    def __init__(self, prox_func: ProximableFunctional, linear_part: LinearFunctional, intercept: Number):
        if not isinstance(linear_part, LinearFunctional) or linear_part.dim != prox_func.dim:
            raise TypeError('Invalid affine sum.')
        super(ProxFuncAffineSum, self).__init__(dim=prox_func.dim, data=prox_func.data,
                                                is_differentiable=prox_func.is_differentiable)
        self.prox_func = prox_func
        self.linear_part = linear_part
        self.intercept = intercept

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return self.prox_func.__call__(x) + self.linear_part.__call__(x) + self.intercept

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        a = self.linear_part.todense().mat.reshape(-1)
        return self.prox_func.prox(x - tau * a, tau)


class ProxFuncPreComp(ProximableFunctional):
    def __init__(self, prox_func: ProximableFunctional, scale: Union[Number, np.ndarray],
                 shift: Union[Number, np.ndarray]):
        super(ProxFuncPreComp, self).__init__(dim=prox_func.dim, data=prox_func.data,
                                              is_differentiable=prox_func.is_differentiable)
        self.prox_func = prox_func
        self.scale = scale
        self.shift = shift

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return self.prox_func.__call__(self.scale * x + self.shift)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return (self.prox_func.prox(self.scale * x + self.shift, tau * (self.scale ** 2)) - self.shift) / self.scale


class ProxFuncPreCompUnitOp(ProximableFunctional):
    def __init__(self, prox_func: ProximableFunctional, unitary_op: UnitaryOperator):
        super(ProxFuncPreCompUnitOp, self).__init__(dim=prox_func.dim, data=prox_func.data,
                                                    is_differentiable=prox_func.is_differentiable)
        self.prox_func = prox_func
        self.unitary_op = unitary_op

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return self.prox_func.__call__(self.unitary_op.matvec(x))

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return self.unitary_op.adjoint(self.prox_func.prox(self.unitary_op.matvec(x), tau=tau))


class ProxFuncHStack(ProximableFunctional, MapHStack):
    def __init__(self, *proxfuncs):
        MapHStack.__init__(self, *proxfuncs)
        self.proxfuncs = self.maps
        ProximableFunctional.__init__(self, dim=self.shape[1], data=None, is_differentiable=self.is_differentiable,
                                      is_linear=self.is_linear)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        x_split = np.split(x, self.sections)
        result = [func.prox(x_split[i]) for i, func in enumerate(self.proxfuncs)]
        return np.concatenate(result, axis=0)


class LpNorm(ProximableFunctional):
    def __init__(self, dim: int, proj_lq_ball: Callable):
        super(LpNorm, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.proj_lq_ball = proj_lq_ball

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return x - tau * self.proj_lq_ball(x / tau, radius=1)


class IndicatorFunctional(ProximableFunctional):
    def __init__(self, dim: int, condition_func: Callable, projection_func: Callable):
        super(IndicatorFunctional, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.condition_func = condition_func
        self.projection_func = projection_func

    def __call__(self, x: Union[Number, np.ndarray], **kwargs) -> Number:
        return 0 if self.condition_func(x, **kwargs) else np.infty

    def prox(self, x: Union[Number, np.ndarray], tau: Number, **kwargs) -> Union[Number, np.ndarray]:
        return self.projection_func(x, **kwargs)
