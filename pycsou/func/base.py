# #############################################################################
# base.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Interface classes for constructing functionals.
"""

from numbers import Number
from typing import Union, Callable

import numpy as np

from pycsou.core.functional import ProximableFunctional, DifferentiableFunctional
from pycsou.core.map import MapHStack


class ProxFuncHStack(ProximableFunctional, MapHStack):
    r"""
    Stack functionals horizontally.

    Consider a collection :math:`\{f_i:\mathbb{R}^{N_i}\to \mathbb{R}, i=1,\ldots, k\}`
    of functionals. Their horizontal stacking is defined as the operator

    .. math::

       h:\begin{cases}\mathbb{R}^{N_1}\times \cdots \times\mathbb{R}^{N_k}\to \mathbb{R}\\
       (\mathbf{x}_1,\ldots, \mathbf{x}_k)\mapsto \sum_{i=1}^k f_i(\mathbf{x}_i).
       \end{cases}

    The proximity operator of :math:`h` is moreover given by ([ProxAlg]_ Section 2.1):

    .. math::
       \mathbf{\text{prox}}_{\tau h}(\mathbf{x}_1,\ldots, \mathbf{x}_k)=\left(\mathbf{\text{prox}}_{\tau f_1}(\mathbf{x}_1),\ldots, \mathbf{\text{prox}}_{\tau f_k}(\mathbf{x}_k)\right).

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.base import ProxFuncHStack

    .. doctest::

       >>> from pycsou.func.penalty import L1Norm, SquaredL1Norm
       >>> func1 = L1Norm(dim=10)
       >>> func2 = SquaredL1Norm(dim=10)
       >>> func = ProxFuncHStack(func1, func2)
       >>> x = np.arange(10); y= x/2; tau=0.1
       >>> np.allclose(func.prox(np.concatenate((x,y)), tau), np.concatenate((func1.prox(x, tau), func2.prox(y, tau))))
       True

    """

    def __init__(self, *proxfuncs):
        r"""
        Parameters
        ----------
        proxfuncs: ProximableFunctional
            List of proximable functionals to stack.
        """
        MapHStack.__init__(self, *proxfuncs)
        self.proxfuncs = self.maps
        ProximableFunctional.__init__(self, dim=self.shape[1], data=None, is_differentiable=self.is_differentiable,
                                      is_linear=self.is_linear)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        x_split = np.split(x, self.sections)
        result = [func.prox(x_split[i], tau) for i, func in enumerate(self.proxfuncs)]
        return np.concatenate(result, axis=0)


class IndicatorFunctional(ProximableFunctional):
    r"""
    Base class for indicator functionals.
    """

    def __init__(self, dim: int, condition_func: Callable, projection_func: Callable):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the functional's domain.
        condition_func: Callable
            Condition delimiting the domain of the indicator functional.
        projection_func: Callable
            Projecton onto the domain of the indicator functional.

        See Also
        --------
        :py:func:`~pycsou.math.prox.proj_nonnegative_orthant`, :py:func:`~pycsou.math.prox.proj_segment`
        """
        super(IndicatorFunctional, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.condition_func = condition_func
        self.projection_func = projection_func

    def __call__(self, x: Union[Number, np.ndarray], **kwargs) -> Number:
        return 0 if self.condition_func(x, **kwargs) else np.infty

    def prox(self, x: Union[Number, np.ndarray], tau: Number, **kwargs) -> Union[Number, np.ndarray]:
        return self.projection_func(x, **kwargs)


class NullDifferentiableFunctional(DifferentiableFunctional):

    def __init__(self, dim: int):
        super(NullDifferentiableFunctional, self).__init__(dim=dim, is_linear=True, lipschitz_cst=0,
                                                           diff_lipschitz_cst=0)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return 0

    def jacobianT(self, arg: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return np.zeros(shape=self.dim)


class NullProximableFunctional(ProximableFunctional):

    def __init__(self, dim: int):
        super(NullProximableFunctional, self).__init__(dim=dim, is_linear=True)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return 0

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return x


class LpNorm(ProximableFunctional):
    r"""
    Base class for :math:`\ell_p`-norms.

    Proximity operators of :math:`\ell_p`-norms are computed via Moreau's identity and the knowledge of the projection
    onto the conjugate  :math:`\ell_p`-ball with :math:`1/p+1/q=1.`
    """

    def __init__(self, dim: int, proj_lq_ball: Callable):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the functional's domain.
        proj_lq_ball: Callable[x: np.ndarray, radius: float]
            Projection onto the :math:`\ell_q`-ball where :math:`1/p+1/q=1.`

        See Also
        --------
        :py:func:`~pycsou.math.prox.proj_l2_ball`, :py:func:`~pycsou.math.prox.proj_l1_ball`, :py:func:`~pycsou.math.prox.proj_linfty_ball`
        """
        super(LpNorm, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.proj_lq_ball = proj_lq_ball

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        return x - tau * self.proj_lq_ball(x / tau, radius=1)
