# #############################################################################
# penalty.py
# ==========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# Contributors: Pol del Aguila Pla [polsocjo@gmail.com] - L21Norm class
# #############################################################################

r"""
Repository of common penalty functionals.
"""

from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional
from pycsou.func.base import IndicatorFunctional, LpNorm
from pycsou.linop.base import DenseLinearOperator
from pycsou.core import LinearOperator
from pycsou.math.prox import soft, proj_l1_ball, proj_l2_ball, proj_linfty_ball, proj_nonnegative_orthant, proj_segment
from typing import Union, Optional
from numbers import Number
import numpy as np
import scipy.optimize as sciop


class L2Norm(LpNorm):
    r"""
    :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N |x_i|^2}`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L2Norm

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = L2Norm(dim=x.size)
       >>> norm(x)
       16.881943016134134
       >>> tau = 1.2; np.allclose(norm.prox(x, tau=tau),np.clip(1 - tau / norm(x), a_min=0, a_max=None) * x)
       True
       >>> lambda_ = 3; scaled_norm = lambda_ * norm; scaled_norm(x)
       50.645829048402405
       >>> np.allclose(scaled_norm.prox(x, tau=tau),np.clip(1 - tau * lambda_ / norm(x), a_min=0, a_max=None) * x)
       True

    Notes
    -----
    The :math:`\ell_2`-norm is a strictly-convex but non differentiable penalty functional.
    Solutions to :math:`\ell_2`-penalised convex optimisation problems are usually non unique and very smooth.
    The proximal operator of the :math:`\ell_2`-norm can be found in [ProxAlg]_ section 6.5.1.

    See Also
    --------
    :py:func:`~pycsou.func.loss.L2Loss`, :py:class:`~pycsou.func.penalty.SquaredL2Norm`, :py:func:`~pycsou.func.penalty.L2Ball`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(L2Norm, self).__init__(dim=dim, proj_lq_ball=proj_l2_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.linalg.norm(x)


class SquaredL2Norm(DifferentiableFunctional):
    r"""
    :math:`\ell^2_2`-norm, :math:`\Vert\mathbf{x}\Vert^2_2:=\sum_{i=1}^N |x_i|^2`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.linop.base import DenseLinearOperator

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = SquaredL2Norm(dim=x.size)
       >>> norm(x)
       285.00000000000006
       >>> np.allclose(norm.gradient(x), 2 * x)
       True
       >>> lambda_=3; scaled_norm = lambda_ * norm
       >>> scaled_norm(x)
       855.0000000000002
       >>> np.allclose(scaled_norm.gradient(x), 2 * lambda_ *  x)
       True
       >>> Gmat = np.arange(100).reshape(10, 10)
       >>> G = DenseLinearOperator(Gmat, is_symmetric=False)
       >>> weighted_norm = norm * G
       >>> np.allclose(weighted_norm.gradient(x), 2 * Gmat.transpose() @ (Gmat @ x))
       True

    Notes
    -----
    The :math:`\ell^2_2` penalty or *Tikhonov* penalty is strictly-convex and differentiable. It is used in ridge regression.
    Solutions to :math:`\ell^2_2`-penalised convex optimisation problems are unique and usually very smooth.

    See Also
    --------
    :py:func:`~pycsou.func.loss.SquaredL2Loss`, :py:class:`~pycsou.func.penalty.L2Norm`, :py:func:`~pycsou.func.penalty.L2Ball`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(SquaredL2Norm, self).__init__(dim=dim, data=None, is_linear=False, lipschitz_cst=np.infty,
                                            diff_lipschitz_cst=2)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.linalg.norm(x) ** 2

    def jacobianT(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        r"""Gradient of the squared L2 norm at x."""
        return 2 * x


def L2Ball(dim: int, radius: Number) -> IndicatorFunctional:
    r"""
    Indicator function of the :math:`\ell_2`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_2\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_2\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim : int
        Dimension of the domain.
    radius: Number
        Radius of the :math:`\ell_2`-ball.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the :math:`\ell_2`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L2Ball
       from pycsou.math.prox import proj_l2_ball

    .. doctest::

       >>> x1 = np.arange(10); x2 = x1 / np.linalg.norm(x1)
       >>> radius=10; ball = L2Ball(dim=x1.size, radius=radius)
       >>> ball(x1), ball(x2)
       (inf, 0)
       >>> np.allclose(ball.prox(x1,tau=1), proj_l2_ball(x1, radius=radius)), np.linalg.norm(ball.prox(x1,tau=1))
       (True, 10.0)
       >>> np.allclose(ball.prox(x2,tau=1), x2)
       True

    Notes
    -----
    The :math:`\ell_2`-ball penalty is convex and proximable. It is a constrained variant of the :math:`\ell_2`-norm penalty.
    The proximal operator of the :math:`\ell_2`-ball indicator is the projection onto the :math:`\ell_2`-ball
    (see [ProxAlg]_ Section 1.2).

    See Also
    --------
    :py:func:`~pycsou.func.loss.L2BallLoss`, :py:class:`~pycsou.func.penalty.L2Norm`, py:class:`pycsou.class.penalty.SquaredL2Norm`.
    """
    condition_func = lambda x: np.linalg.norm(x) <= radius
    projection_func = lambda x: proj_l2_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class L1Norm(LpNorm):
    r"""
    :math:`\ell_1`-norm, :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L1Norm
       from pycsou.math.prox import soft

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = L1Norm(dim=x.size)
       >>> norm(x)
       45
       >>> tau=1.2; np.allclose(norm.prox(x, tau=tau),soft(x,tau=tau))
       True
       >>> lambda_=3; scaled_norm = lambda_ * norm; scaled_norm(x)
       135
       >>> np.allclose(scaled_norm.prox(x, tau=tau),soft(x,tau=tau * lambda_))
       True

    Notes
    -----
    The :math:`\ell_1`-norm penalty is convex and proximable. This penalty tends to produce non unique and sparse solutions.
    The proximal operator of the :math:`\ell_1`-norm is provided in [ProxAlg]_ Section 6.5.2.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:func:`~pycsou.func.loss.L1Loss`, :py:class:`~pycsou.func.penalty.SquaredL1Norm`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim : int
            Dimension of the domain.
        """
        super(L1Norm, self).__init__(dim=dim, proj_lq_ball=proj_linfty_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x))

    def soft(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""Soft thresholding operator (see :py:func:`~pycsou.math.prox.soft` for a definition)."""
        return soft(x=x, tau=tau)


class SquaredL1Norm(ProximableFunctional):
    r"""
    :math:`\ell^2_1`-norm, :math:`\Vert\mathbf{x}\Vert^2_1:=\left(\sum_{i=1}^N |x_i|\right)^2`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import SquaredL1Norm

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = SquaredL1Norm(dim=x.size, prox_computation='sort')
       >>> norm(x)
       2025
       >>> norm2 = SquaredL1Norm(dim=x.size, prox_computation='root')
       >>> np.allclose(norm.prox(x, tau=1),norm2.prox(x, tau=1))
       True

    Notes
    -----
    The :math:`\ell^2_1`-norm penalty is strictly-convex and proximable. This penalty tends to produce a unique and sparse solution.
    Two alternative ways of computing the proximal operator of the :math:`\ell^2_1`-norm are provided in [FirstOrd]_ Lemma 6.70
    and [OnKerLearn]_ Algorithm 2 respectively.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:class:`~pycsou.func.penalty.L1Norm`, :py:func:`~pycsou.func.penalty.SquaredL1Loss`.
    """

    def __init__(self, dim: int, prox_computation: str = 'sort'):
        r"""
        Parameters
        ----------
        dim : int
            Dimension of the domain.
        prox_computation: str, optional
            Algorithm for computing the proximal operator: 'root' uses [FirstOrd]_ Lemma 6.70, while 'sort' uses [OnKerLearn]_ Algorithm 2 (faster).
        """
        self.prox_computation = prox_computation
        super(SquaredL1Norm, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.abs(x)) ** 2

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""
        Proximal operator, see :py:class:`pycsou.core.functional.ProximableFunctional` for a detailed description.
        """
        if self.prox_computation == 'root':
            if np.linalg.norm(x) > 0:
                mu_max = np.max(np.abs(x) ** 2) / (4 * tau)
                mu_min = 1e-12
                func = lambda mu: np.sum(np.clip(np.abs(x) * np.sqrt(tau / mu) - 2 * tau, a_min=0, a_max=None)) - 1
                mu_star = sciop.brentq(func, a=mu_min, b=mu_max)
                lambda_ = np.clip(np.abs(x) * np.sqrt(tau / mu_star) - 2 * tau, a_min=0, a_max=None)
                return lambda_ * x / (lambda_ + 2 * tau)
            else:
                return x
        elif self.prox_computation == 'sort':
            y = np.sort(np.abs(x))[::-1]
            cumsum_y = np.cumsum(y)
            test_array = y - (2 * tau / (1 + (np.arange(y.size) + 1) * 2 * tau)) * cumsum_y
            max_nzi = np.max(np.nonzero(test_array > 0))
            threshold = (2 * tau / (1 + (max_nzi + 1) * 2 * tau)) * cumsum_y[max_nzi]
            return soft(x, threshold)


def L1Ball(dim: int, radius: Number) -> IndicatorFunctional:
    r"""
    Indicator function of the :math:`\ell_1`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_1\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_1\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim : int
        Dimension of the domain.
    radius: Number
        Radius of the :math:`\ell_1`-ball.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the :math:`\ell_1`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L1Ball
       from pycsou.math.prox import proj_l1_ball

    .. doctest::

       >>> x1 = np.arange(10); x2 = x1 / np.linalg.norm(x1, ord=1)
       >>> radius=10; ball = L1Ball(dim=x1.size, radius=radius)
       >>> ball(x1), ball(x2)
       (inf, 0)
       >>> np.allclose(ball.prox(x1,tau=1), proj_l1_ball(x1, radius=radius)), np.linalg.norm(ball.prox(x1,tau=1), ord=1)
       (True, 10.0)
       >>> np.allclose(ball.prox(x2,tau=1), x2)
       True

    Notes
    -----
    The :math:`\ell_1`-ball penalty is convex and proximable. It is a constrained variant of the :math:`\ell_1`-norm penalty.
    The proximal operator of the :math:`\ell_1`-ball indicator is the projection onto the :math:`\ell_1`-ball
    (see [ProxAlg]_ Section 6.5.2).

    See Also
    --------
    :py:func:`~pycsou.func.loss.L1BallLoss`, :py:class:`~pycsou.func.penalty.L1Norm`, py:class:`pycsou.func.penalty.SquaredL1Norm`.
    """

    condition_func = lambda x: np.sum(np.abs(x)) <= radius
    projection_func = lambda x: proj_l1_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class LInftyNorm(LpNorm):
    r"""
    :math:`\ell_\infty`-norm, :math:`\Vert\mathbf{x}\Vert_\infty:=\max_{i=1,\ldots,N} |x_i|`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import LInftyNorm

    .. doctest::

       >>> x = np.arange(10)
       >>> norm = LInftyNorm(dim=x.size)
       >>> norm(x)
       9
       >>> lambda_ = 3; scaled_norm = lambda_ * norm; scaled_norm(x)
       27

    Notes
    -----
    The :math:`\ell_\infty`-norm is a convex but non differentiable penalty functional.
    Solutions to :math:`\ell_\infty`-penalised convex optimisation problems are non unique and binary (i.e. :math:`x_i\in\{1,-1\}`).
    The proximal operator of the :math:`\ell_\infty`-norm does not admit a closed-form but can be computed effficiently
    as discussed in [ProxAlg]_ section 6.5.2.

    See Also
    --------
    :py:func:`~pycsou.func.loss.LInftyLoss`, :py:func:`~pycsou.func.penalty.LInftyBall`.
    """

    def __init__(self, dim: int):
        super(LInftyNorm, self).__init__(dim=dim, proj_lq_ball=proj_l1_ball)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.max(np.abs(x))


def LInftyBall(dim: int, radius: Number) -> IndicatorFunctional:
    r"""
    Indicator function of the :math:`\ell_\infty`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_\infty\leq \text{radius}\}`

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}\|_\infty\leq \text{radius},\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim : int
        Dimension of the domain.
    radius: Number
        Radius of the :math:`\ell_\infty`-ball.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the :math:`\ell_\infty`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import LInftyBall
       from pycsou.math.prox import proj_linfty_ball

    .. doctest::

       >>> x1 = np.arange(10); x2 = x1 / np.linalg.norm(x1, ord=np.inf)
       >>> radius=8; ball = LInftyBall(dim=x1.size, radius=radius)
       >>> ball(x1), ball(x2)
       (inf, 0)
       >>> np.allclose(ball.prox(x1,tau=1), proj_linfty_ball(x1, radius=radius)), np.linalg.norm(ball.prox(x1,tau=1), ord=np.inf)
       (True, 8.0)
       >>> np.allclose(ball.prox(x2,tau=1), x2)
       True

    Notes
    -----
    The :math:`\ell_\infty`-ball penalty is convex and proximable. It is a constrained variant of the :math:`\ell_\infty`-norm penalty.
    The proximal operator of the :math:`\ell_\infty`-ball indicator is the projection onto the :math:`\ell_\infty`-ball
    (see [ProxAlg]_ Section 6.5.2).

    See Also
    --------
    :py:func:`~pycsou.func.loss.LInftyBallLoss`, :py:class:`~pycsou.func.penalty.LInftyNorm`.
    """
    condition_func = lambda x: np.max(np.abs(x)) <= radius
    projection_func = lambda x: proj_linfty_ball(x, radius=radius)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class L21Norm(ProximableFunctional):
    r"""
    :math:`\ell_{2,1}`-norm, :math:`\Vert\mathbf{x}\Vert_{2,1}:=\sum_{g=1}^G \sqrt{ \sum_{i\in\mathcal{G}_g} |x_i|^2}\,.`

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import L21Norm, L2Norm, L1Norm

    .. doctest::

       >>> x = np.arange(10,dtype=np.float64)
       >>> groups = np.concatenate((np.ones(5),2*np.ones(5)))
       >>> group_norm = L21Norm(dim=x.size,groups=groups)
       >>> type(group_norm)
       <class 'pycsou.func.penalty.L21Norm'>
       >>> group_norm(x)
       21.44594499772297
       >>> l2_norm = L21Norm(dim=x.size,groups=np.ones(x.size))
       >>> type(l2_norm)
       <class 'pycsou.func.penalty.L2Norm'>
       >>> l1_norm = L21Norm(dim=x.size,groups=np.arange(x.size)) # Also if groups = None
       >>> type(l1_norm)
       <class 'pycsou.func.penalty.L1Norm'>
       >>> single_group_l2 = L2Norm(dim=x.size/2)
       >>> tau = 0.5; np.allclose(group_norm.prox(x,tau=tau),np.concatenate((single_group_l2.prox(x[0:5], tau=tau),single_group_l2.prox(x[5:10],tau=tau))))
       True

    Notes
    -----
    The :math:`\ell_{2,1}`-norm penalty is convex and proximable. This penalty tends to produce group sparse solutions,
    where all elements :math:`x_i` for :math:`i\in\mathcal{G}_g` for some :math:`g\in\lbrace 1,2,\dots,G` tend to
    be zero or non-zero jointly. A critical assumtion is that the groups are not overlapping, i.e.,
    :math:`\mathcal{G}_j \cap \mathcal{G}_i = \emptyset` for :math:`j,i \in \lbrace 1,2,\dots,G\rbrace` such that :math:`j\neq i.`
    The proximal operator of the :math:`\ell_{2,1}`-norm is obtained easily from that of the :math:`\ell_2`-norm and the
    separable sum property.

    See Also
    --------
    :py:func:`~pycsou.func.loss.L2Norm`, :py:class:`~pycsou.func.penalty.SquaredL2Norm`, :py:func:`~pycsou.func.penalty.L1Norm`.
    """

    def __new__(cls, dim: int, groups: Union[np.ndarray, None] = None):
        if np.all(groups == None) or np.unique(groups).size == dim:
            return L1Norm(dim=dim)
        if np.unique(groups).size == 1:
            return L2Norm(dim=dim)
        return super(L21Norm, cls).__new__(cls)

    def __init__(self, dim: int, groups: np.ndarray):
        r"""
        Parameters
        ----------
        dim : int
            Dimension of the domain.
        groups : np.ndarray, optional, defaults to None
            Numerical variable of the same size as :math:`x`, where different
            groups are distinguished by different values. If each element of `x`
            belongs to a different group, an L1Norm is returned. If all elements
            of `x` belong to the same group, an L2Norm is returned.
        """
        self.groups = groups
        self.groups_idxs = np.unique(self.groups)
        super(L21Norm, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        return np.sum(np.array([self.__L2_norm_in_group(x, group_id) for group_id in self.groups_idxs]))

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        y = np.empty_like(x)
        group_norms = np.array([self.__L2_norm_in_group(x, group_id) for group_id in self.groups_idxs])
        normalizations = np.clip(1 - tau / group_norms, a_min=0, a_max=None)
        for idx, group_idx in enumerate(self.groups_idxs):
            y[self.groups == group_idx] = normalizations[idx] * x[self.groups == group_idx]
        return y

    def __L2_norm_in_group(self, x: np.ndarray, group_idx: Number) -> Number:
        return np.linalg.norm(x[self.groups == group_idx])


def NonNegativeOrthant(dim: int) -> IndicatorFunctional:
    r"""
    Indicator function of the non negative orthant (positivity constraint).

    It is used to enforce positive real solutions. It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}\in \mathbb{R}^N_+,\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the non negative orthant.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import NonNegativeOrthant
       from pycsou.math.prox import proj_nonnegative_orthant

    .. doctest::

       >>> x1 = np.arange(10) - 5
       >>> func = NonNegativeOrthant(dim=x1.size)
       >>> func(x1), func(np.abs(x1))
       (inf, 0)
       >>> np.allclose(func.prox(x1,tau=1), proj_nonnegative_orthant(x1))
       True
       >>> np.alltrue(func.prox(x1,tau=1) >= 0)
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.LogBarrier`.
    """
    condition_func = lambda x: np.alltrue(x >= 0)
    projection_func = lambda x: proj_nonnegative_orthant(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def Segment(dim: int, a: Number = 0, b: Number = 1):
    r"""
    Indicator function of the segment :math:`[a,b]\subset\mathbb{R}`.

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}\in [a,b]^N,\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    a: Number
        Left endpoint of the segement.
    b: Number
        Right endpoint of the segment.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the segment :math:`[a,b]\subset\mathbb{R}`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import Segment
       from pycsou.math.prox import proj_segment

    .. doctest::

       >>> x1 = np.arange(10) - 3
       >>> func = Segment(dim=x1.size, a=1, b=4)
       >>> func(x1), func(np.clip(x1, a_min=1, a_max=4))
       (inf, 0)
       >>> np.allclose(func.prox(x1,tau=1), proj_segment(x1, a=1, b=4))
       True
       >>> func.prox(x1,tau=1)
       array([1, 1, 1, 1, 1, 2, 3, 4, 4, 4])

    See Also
    --------
    :py:func:`~pycsou.func.penalty.RealLine`, :py:func:`~pycsou.func.penalty.ImagLine`.
    """
    condition_func = lambda x: np.alltrue((x >= a) & (x <= b))
    projection_func = lambda x: proj_segment(x, a=a, b=b)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def RealLine(dim: int):
    r"""
    Indicator function of the real line :math:`\mathbb{R}`.

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}\in \mathbb{R}^N,\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the real line :math:`\mathbb{R}`.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import RealLine

    .. doctest::

       >>> x1 = np.arange(10) + 1j
       >>> func = RealLine(dim=x1.size)
       >>> func(x1), func(np.real(x1))
       (inf, 0)
       >>> np.allclose(func.prox(x1,tau=1), np.real(x1))
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.NonNegativeOrthant`, :py:func:`~pycsou.func.penalty.ImagLine`.
    """
    condition_func = lambda x: np.alltrue(np.isreal(x))
    projection_func = lambda x: np.real(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


def ImagLine(dim: int):
    r"""
    Indicator function of the imaginary line :math:`j\mathbb{R}`.

    It is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}\in (j\mathbb{R})^N,\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.

    Returns
    -------
    py:class:`pycsou.core.functional.IndicatorFunctional`
        Indicator function of the imaginary line :math:`j\mathbb{R}`.

    Examples
    --------


    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import ImagLine

    .. doctest::

       >>> x1 = np.arange(10) + 1j * np.arange(10)
       >>> func = ImagLine(dim=x1.size)
       >>> func(x1), func(1j * np.imag(x1))
       (inf, 0)
       >>> np.allclose(func.prox(x1,tau=1), np.imag(x1))
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.NonNegativeOrthant`, :py:func:`~pycsou.func.penalty.RealLine`.
    """
    condition_func = lambda x: np.alltrue(np.real(x) == 0)
    projection_func = lambda x: np.imag(x)
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class LogBarrier(ProximableFunctional):
    r"""
    Log barrier, :math:`f(\mathbf{x}):= -\sum_{i=1}^N \log(x_i).`

    The log barrier is defined as:

    .. math::

       f(x):=\begin{cases}
       -\log(x) & \text{if} \, x>0,\\
       +\infty & \text{otherwise.}
       \end{cases}

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import LogBarrier

    .. doctest::

       >>> x1 = np.arange(10)
       >>> func = LogBarrier(dim=x1.size)
       >>> func(x1), func(x1+2)
       (inf, -17.502307845873887)
       >>> np.round(func.prox(x1,tau=1))
       array([1., 2., 2., 3., 4., 5., 6., 7., 8., 9.])

    Notes
    -----
    The log barrier can be used to enforce positivity in the solution.
    Its proximal operator is given in [ProxAlg]_ Section 6.7.5.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.NonNegativeOrthant`, :py:func:`~pycsou.func.penalty.ShannonEntropy`.
    """

    def __init__(self, dim: int):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the domain.
        """
        super(LogBarrier, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        y = 0 * x - np.infty
        y[x > 0] = np.log(x[x > 0])
        return - y.sum()

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""
        Proximal operator of the log barrier.

        Parameters
        ----------
        x: Union[Number, np.ndarray]
            Input.
        tau: Number
            Scaling constant.

        Returns
        -------
        Union[Number, np.ndarray]
            Proximal point of x.
        """
        return (x + np.sqrt(x ** 2 + 4 * tau)) / 2


class ShannonEntropy(ProximableFunctional):
    r"""
    Negative Shannon entropy, :math:`f(\mathbf{x}):= \sum_{i=1}^N x_i\log(x_i).`

    The (negative) Shannon entropy is defined as:

    .. math::

       f(x):=\begin{cases}
       x\log(x) & \text{if} \, x>0,\\
       0& \text{if} \, x=0,\\
       +\infty & \text{otherwise.}
       \end{cases}

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.penalty import ShannonEntropy

    .. doctest::

       >>> x1 = np.arange(10); x2=np.zeros(10); x2[0]=10
       >>> func = ShannonEntropy(dim=x1.size)
       >>> func(x1), func(x2)
       (79.05697962199447, 23.02585092994046)
       >>> np.round(func.prox(x1,tau=2))
       array([0., 0., 1., 1., 1., 2., 2., 3., 3., 4.])

    Notes
    -----
    This regularization functional is based the information-theoretic notion of entropy, a mathematical generalisation of
    entropy as introduced by Boltzmann in thermodynamics. It favours solutions with maximal entropy. The latter are typically
    positive and featureless: smooth functions indeed carry much less spatial information than functions with sharp, localised features,
    and hence have higher entropy. This penalty is restricted to positive solutions and often results in overly-smooth estimates.
    Its proximal operator is given in [ProxSplit]_ Table 2 or [ProxEnt]_ Section 3.1.

    .. [ProxEnt] Afef, Cherni, Chouzenoux Émilie, and Delsuc Marc-André. "Proximity operators for a class of hybrid sparsity+ entropy priors application to dosy NMR signal reconstruction." 2016 International Symposium on Signal, Image, Video and Communications (ISIVC). IEEE, 2016.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.LogBarrier`, :py:class:`~pycsou.func.loss.KLDivergence`.
    """

    def __init__(self, dim: int):
        r"""

        Parameters
        ----------
        dim: int
            Dimension of the domain.
        """
        super(ShannonEntropy, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        y = 0 * x + np.infty
        y[x == 0] = 0
        y[x > 0] = x[x > 0] * np.log(x[x > 0])
        return y.sum()

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""
        Proximal operator of the Shannon entropy functional.

        Parameters
        ----------
        x: Union[Number, np.ndarray]
            Input.
        tau: Number
            Scaling constant.

        Returns
        -------
        Union[Number, np.ndarray]
            Proximal point of x.
        """
        from scipy.special import lambertw
        return np.real(tau * lambertw(np.exp(-1 + (x / tau)) / tau, k=0))


class QuadraticForm(DifferentiableFunctional):
    r"""
    Quadratic form :math:`\mathbf{x}^\ast \mathbf{L} \mathbf{x}`.

    Examples
    --------
    .. testsetup::

        import numpy as np
        from pycsou.linop import DenseLinearOperator
        from pycsou.func.penalty import QuadraticForm

    .. doctest::

        >>> rng = np.random.default_rng(0)
        >>> L =  rng.standard_normal(100).reshape(10,10)
        >>> L = L.transpose() @ L #make definite positive
        >>> Lop = DenseLinearOperator(L)
        >>> F = QuadraticForm(dim=10,linop=Lop)
        >>> x = np.arange(10)
        >>> np.allclose(F(x), np.dot(x, Lop @ x))
        True
        >>> np.allclose(F.gradient(x), 2 * Lop @ x)
        True

    Notes
    -----
    The quadratic form is defined as the functional :math:`F:\mathbb{R}^n\to \mathbb{R}, \; \mathbf{x}\mapsto \mathbf{x}^\ast \mathbf{L}\mathbf{x}`
    for some *positive semi-definite* linear operator :math:`\mathbf{L}:\mathbb{R}^n\to \mathbb{R}^n`.
    Its gradient is given by :math:`\nabla F(\mathbf{x})=2 \mathbf{L}\mathbf{x}.` The latter is :math:`\beta`-Lipschitz continuous with
    :math:`\beta=2\|\mathbf{L}\|_2.`

    See Also
    --------
    :py:func:`~pycsou.func.penalty.SquaredL2Norm`
    """

    def __init__(self, dim: int, linop: Optional[LinearOperator] = None):
        r"""

        Parameters
        ----------
        dim: int
            Dimension of the domain.
        linop: LinearOperator
            Positive semi-definite operator defining the quadratic form. If ``None`` the identity operator is assumed.
        """
        self.linop = linop
        if self.linop is None:
            diff_lipschitz_cst = 2
        else:
            diff_lipschitz_cst = 2 * linop.diff_lipschitz_cst
        super(QuadraticForm, self).__init__(dim=dim, data=None, is_linear=False,
                                            diff_lipschitz_cst=diff_lipschitz_cst)

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        if self.linop is None:
            return np.dot(x.conj(), x)
        else:
            return np.dot(x.conj(), self.linop * x)

    def jacobianT(self, x: Union[Number, np.ndarray]) -> np.ndarray:
        if self.linop is None:
            return 2 * x
        else:
            return 2 * self.linop * x
