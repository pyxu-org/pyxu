# #############################################################################
# functional.py
# =============
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Abstract classes for functionals.
"""

from pycsou.core.map import Map, DifferentiableMap, MapSum, MapComp
from abc import abstractmethod
from pycsou.core.linop import LinearOperator, UnitaryOperator
from typing import Union, Optional
from numbers import Number
import numpy as np
import warnings


class Functional(Map):
    r"""
    Base class for functionals.

    Notes
    -----
    Functionals are (real) single-valued nonlinear maps.
    """

    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_differentiable: bool = False,
                 is_linear: bool = False):
        r"""
        Parameters
        ----------
        dim: int,
            Dimension of the functional's domain.
        data: Union[None, Number, np.ndarray]
            Optional data vector.
        is_differentiable: bool
            Whether the functional is differentiable or not.
        is_linear: bool
            Whether the functional is linear or not.
        """
        Map.__init__(self, shape=(1, dim), is_differentiable=is_differentiable, is_linear=is_linear)
        self.data = data
        self.dim = dim


class DifferentiableFunctional(Functional, DifferentiableMap):
    r"""
    Base class for differentiable functionals.
    """

    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_linear: bool = False,
                 lipschitz_cst: float = np.infty, diff_lipschitz_cst: float = np.infty):
        r"""
        Parameters
        ----------
        dim: int,
            Dimension of the functional's domain.
        data: Union[None, Number, np.ndarray]
            Optional data vector.
        is_linear: bool
            Whether the functional is linear or not.
        lipschitz_cst: float
            Lispchitz constant of the differentiable map if it exists/is known. Default to :math:`+\infty`.
        diff_lipschitz_cst: float
            Lispchitz constant of the derivative of the differentiable map if it exists/is known. Default to :math:`+\infty`.
        """
        Functional.__init__(self, dim=dim, data=data, is_differentiable=True, is_linear=is_linear)
        DifferentiableMap.__init__(self, shape=self.shape, is_linear=self.is_linear, lipschitz_cst=lipschitz_cst,
                                   diff_lipschitz_cst=diff_lipschitz_cst)

    @abstractmethod
    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        pass


class LinearFunctional(Functional, LinearOperator):
    r"""
    Base class for linear functionals.
    """

    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, dtype: type = np.float64,
                 is_explicit: bool = False, is_dense: bool = False, is_sparse: bool = False, is_dask: bool = False):
        Functional.__init__(self, dim=dim, data=data, is_differentiable=True, is_linear=True)
        LinearOperator.__init__(self, shape=self.shape, dtype=dtype, is_explicit=is_explicit, is_dense=is_dense,
                                is_sparse=is_sparse, is_dask=is_dask, is_symmetric=False)


class ProximableFunctional(Functional):
    r"""
    Base class for proximable functionals.

    Any instance/subclass of this class must at least implement the abstract methods ``__call__`` and ``prox``.

    Notes
    -----
    A functional :math:`f:\mathbb{R}^N\to \mathbb{R}` is said *proximable* is its **proximity operator** ([ProxAlg]_ Section 1.1)

    .. math::

       \mathbf{\text{prox}}_{\tau f}(\mathbf{z}):=\arg\min_{\mathbf{x}\in\mathbb{R}^N} f(x)+\frac{1}{2\tau} \|\mathbf{x}-\mathbf{z}\|_2^2, \quad \forall \mathbf{z}\in\mathbb{R}^N,

    admits a *simple closed-form expression* **or** can be evaluated *efficiently* and with *high accuracy*.

    This class supports the following arithmetic operators ``+``, ``-``, ``*``, ``@``, ``**`` and ``/``, implemented with the
    class methods ``__add__``/``__radd__``, ``__sub__``/``__neg__``, ``__mul__``/``__rmul__``, ``__matmul__``, ``__pow__``, ``__truediv__``.
    Such arithmetic operators can be used to *add*, *substract*, *scale*, *compose*, *exponentiate* or *evaluate* ``LinearOperator`` instances.
    For the following basic operations moreover, the proximal operator is automatically updated according to the formula of  [ProxAlg]_ Section 2.2.:

    - **Postcomposition:** :math:`g(\mathbf{x})=\alpha f(\mathbf{x})`,
    - **Precomposition:** :math:`g(\mathbf{x})= f(\alpha\mathbf{x}+b)` or :math:`g(\mathbf{x})= f(U\mathbf{x})` with :math:`U` a *unitary operator*,
    - **Affine Sum:** :math:`g(\mathbf{x})= f(\mathbf{x})+\mathbf{a}^T\mathbf{x}.`

    Examples
    --------

    .. testsetup::

       import numpy as np

    .. doctest::

       >>> from pycsou.func.penalty import L1Norm
       >>> func = L1Norm(dim=10)
       >>> x = np.arange(10); tau=0.1
       >>> np.allclose((2 * func).prox(x, tau), func.prox(x, 2 * tau))
       True
       >>> np.allclose((func * 2).prox(x, tau), func.prox(x * 2, 4 * tau)/2)
       True
       >>> np.allclose(func.shifter(x/2).prox(x, tau), func.prox(x+x/2, tau)-x/2)
       True


    """

    def __init__(self, dim: int, data: Union[None, Number, np.ndarray] = None, is_differentiable: bool = False,
                 is_linear: bool = False):
        r"""
        Parameters
        ----------
        dim: int,
            Dimension of the functional's domain.
        data: Union[None, Number, np.ndarray]
            Optional data vector.
        is_linear: bool
            Whether the functional is linear or not.
        is_differentiable: bool
            Whether the functional is differentiable or not.
        """
        if is_differentiable == True or is_linear == True:
            warnings.warn(
                'For differentiable and/or linear maps, consider the dedicated classes DifferentiableMap and LinearOperator.')
        super(ProximableFunctional, self).__init__(dim=dim, data=data, is_differentiable=is_differentiable,
                                                   is_linear=is_linear)

    @abstractmethod
    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""
        Evaluate the proximity operator of the ``tau``-scaled functional at the point ``x``.

        Parameters
        ----------
        x: Union[Number, np.ndarray]
            Point at which to perform the evaluation.
        tau: Number
            Scale.

        Returns
        -------
        Union[Number, np.ndarray]
            Evaluation of the proximity operator of the ``tau``-scaled functional at the point ``x``.
        """
        pass

    def fenchel_prox(self, z: Union[Number, np.ndarray], sigma: Number) -> Union[Number, np.ndarray]:
        r"""
        Evaluate the proximity operator of the ``sigma``-scaled Fenchel conjugate of the functional at a point ``z``.

        Parameters
        ----------
        z: Union[Number, np.ndarray]
            Point at which to perform the evaluation.
        sigma: Number
            Scale.

        Returns
        -------
        Union[Number, np.ndarray]
            Result of the evaluation.

        Notes
        -----
        The *Fenchel conjugate* is defined as [FuncSphere]_ Chapter 7, Section 1:

        .. math::

           f^\ast(\mathbf{z}):=\max_{\mathbf{x}\in\mathbb{R}^N} \langle \mathbf{x},\mathbf{z} \rangle - f(\mathbf{x}).

        From **Moreau's identity**, its proximal operator is given by:

        .. math::

           \mathbf{\text{prox}}_{\sigma f^\ast}(\mathbf{z})= \mathbf{z}- \sigma \mathbf{\text{prox}}_{f/\sigma}(\mathbf{z}/\sigma).

        """
        return z - sigma * self.prox(x=z / sigma, tau=1 / sigma)

    def shifter(self, shift: Union[Number, np.ndarray]) -> 'ProxFuncPreComp':
        r"""
        Returns a shifted version of the functional.

        Parameters
        ----------
        shift: Union[Number, np.ndarray]
            Shift vector.

        Returns
        -------
        :py:class:`~pycsou.core.functional.ProxFuncPreComp`
            Shifted map.
        """
        return ProxFuncPreComp(prox_func=self, scale=1, shift=shift)

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
        if isinstance(other, Number) and other > 0:
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
