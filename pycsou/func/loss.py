# #############################################################################
# loss.py
# ==========
# Authors : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Repository of common loss functionals.
"""

from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, ProxFuncPreComp
from pycsou.func.base import IndicatorFunctional
from pycsou.linop.base import DenseLinearOperator
from pycsou.func.penalty import L2Norm, L1Norm, LInftyNorm, L2Ball, L1Ball, LInftyBall, SquaredL1Norm, SquaredL2Norm
from typing import Union
from numbers import Number
import numpy as np


def ProximableLoss(func: ProximableFunctional, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    r"""
    Constructor of proximable loss functions.

    Constructs a proximable loss from a proximable functional and a data vector.
    Let :math:`\varphi:\mathbb{R}^N\rightarrow \mathbb{R}` be some proximable functional and :math:`\mathbf{y}\in\mathbb{R}^N`.
    This routine defines the loss functional :math:`F(\mathbf{x}; \mathbf{y}):= \varphi(\mathbf{x}-\mathbf{y}), \,\forall \mathbf{x}\in\mathbb{R}^N.`

    Parameters
    ----------
    func: ProximableFunctional
        Some proximable functional :math:`\varphi:\mathbb{R}^N\rightarrow \mathbb{R}`.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}\in\mathbb{R}^N`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        Proximable loss functional constructed as :math:`F(\mathbf{x}; \mathbf{y}):= \varphi(\mathbf{x}-\mathbf{y}), \,\forall \mathbf{x}\in\mathbb{R}^N.`

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import ProximableLoss
       from pycsou.func.penalty import L1Norm

    .. doctest::

       >>> y = np.arange(10)
       >>> func = L1Norm(dim=y.size)
       >>> loss = ProximableLoss(func=func, data=y)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss(x), func(x-y))
       True
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The proximity operator of the loss functional is automatically computed from the one of the input functional :math:`\varphi` using
    properties described in [ProxAlg]_ Section 2.1.

    See Also
    --------
    :py:func:`~pycsou.func.loss.DifferentiableLoss`.
    """
    return ProxFuncPreComp(func, scale=1, shift=-data)


def DifferentiableLoss(func: DifferentiableFunctional, data: Union[Number, np.ndarray]) -> DifferentiableFunctional:
    r"""
    Constructor of proximable loss functions.

    Constructs a differentiable loss from a differentiable functional and a data vector.
    Let :math:`\varphi:\mathbb{R}^N\rightarrow \mathbb{R}` be some differentiable functional and :math:`\mathbf{y}\in\mathbb{R}^N`.
    This routine defines the loss functional :math:`F(\mathbf{x}; \mathbf{y}):= \varphi(\mathbf{x}-\mathbf{y}), \,\forall \mathbf{x}\in\mathbb{R}^N.`

    Parameters
    ----------
    func: DifferentiableFunctional
        Some differentiable functional :math:`\varphi:\mathbb{R}^N\rightarrow \mathbb{R}`.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}\in\mathbb{R}^N`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.DifferentiableFunctional`
        Differentiable loss functional constructed as :math:`F(\mathbf{x}; \mathbf{y}):= \varphi(\mathbf{x}-\mathbf{y}), \,\forall \mathbf{x}\in\mathbb{R}^N.`

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import DifferentiableLoss
       from pycsou.func.penalty import SquaredL2Norm

    .. doctest::

       >>> y = np.arange(10)
       >>> func = SquaredL2Norm(dim=y.size)
       >>> loss = DifferentiableLoss(func=func, data=y)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss(x), func(x-y))
       True
       >>> np.allclose(loss.gradient(x), 2*(x-y))
       True

    Notes
    -----
    The derivative and Lipschitz constant of the loss functional are automatically computed from those of the input functional :math:`\varphi`.

    See Also
    --------
    :py:func:`~pycsou.func.loss.ProximableLoss`.
    """
    return func.shifter(shift=-data)


def L2Loss(dim: int, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    r"""
    :math:`\ell_2` loss functional, :math:`F(\mathbf{y},\mathbf{x}):=\|\mathbf{y}-\mathbf{x}\|_2`.

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_2` loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import L2Loss
       from pycsou.func.penalty import L2Norm

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = L2Loss(dim=y.size, data=y)
       >>> func = L2Norm(dim=y.size)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L2Norm`, :py:func:`~pycsou.func.loss.L1Loss`, :py:func:`~pycsou.func.loss.LInftyLoss`
    """
    L2_norm = L2Norm(dim=dim)
    return ProximableLoss(L2_norm, data=data)


def SquaredL2Loss(dim: int, data: Union[Number, np.ndarray]) -> DifferentiableFunctional:
    r"""
    :math:`\ell^2_2` loss functional, :math:`F(\mathbf{y},\mathbf{x}):=\|\mathbf{y}-\mathbf{x}\|^2_2`.

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.DifferentiableFunctional`
        The :math:`\ell^2_2` loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import SquaredL2Loss
       from pycsou.func.penalty import SquaredL2Norm
       from pycsou.linop.base import DenseLinearOperator

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = SquaredL2Loss(dim=y.size, data=y)
       >>> Gmat = np.arange(100).reshape(10, 10).astype(float)
       >>> G = DenseLinearOperator(Gmat, is_symmetric=False)
       >>> G.compute_lipschitz_cst()
       >>> fwd_loss = loss * G
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss(x), np.linalg.norm(y - x) ** 2)
       True
       >>> np.allclose(fwd_loss(x), loss(G(x)))
       True
       >>> np.allclose(fwd_loss.diff_lipschitz_cst, 2 * (G.lipschitz_cst ** 2))
       True
       >>> np.allclose(fwd_loss.gradient(x), 2 * G.adjoint(G(x) - y))
       True

    Notes
    -----
    The :math:`\ell_2^2` functional is the likelihood of the data :math:`\mathbf{y}` under the assumtpion of
    Gaussian white noise.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.SquaredL2Norm`, :py:func:`~pycsou.func.loss.L2Loss`.
    """
    squared_L2_norm = SquaredL2Norm(dim=dim)
    return DifferentiableLoss(squared_L2_norm, data=data)


def L2BallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    r"""
    :math:`\ell_2`-ball loss functional, :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{y}-\mathbf{x}\|_2\leq \text{radius}\}`.

    The :math:`\ell_2`-ball loss functional is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}-\mathbf{y}\|_2\leq \text{radius},\\
         \, 0\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.
    radius: Number
        Radius of the ball.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_2`-ball loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import L2BallLoss
       from pycsou.func.penalty import L2Ball

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = L2BallLoss(dim=y.size, data=y, radius=2)
       >>> func = L2Ball(dim=y.size, radius=2)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The :math:`\ell_2`-ball loss functional is particularly useful in the context of Gaussian white noise with
    known standard deviation. In which case, the :math:`\ell_2`-ball defines a confidence region for the data :math:`\mathbf{y}` ([FuncSphere]_ Section 5 of Chapter 7).

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L2Ball`, :py:func:`~pycsou.func.loss.L2Loss`.
    """
    L2_ball = L2Ball(dim=dim, radius=radius)
    return ProximableLoss(L2_ball, data=data)


def L1Loss(dim: int, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    r"""
    :math:`\ell_1` loss functional, :math:`F(\mathbf{y},\mathbf{x}):=\|\mathbf{y}-\mathbf{x}\|_1`.

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_1` loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import L1Loss
       from pycsou.func.penalty import L1Norm

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = L1Loss(dim=y.size, data=y)
       >>> func = L1Norm(dim=y.size)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The :math:`\ell_1` loss functional leads to sparse residuals, with most of the predicted samples matching exactly the
    observed samples, and a few –potentially large– misfits ([FuncSphere]_ Section 5 of Chapter 7).
    Such a functional is particularly useful in the context of salt-and-pepper noise with strong outliers, or more generally
    for noise distributions with heavy tails, templated by the Laplace distribution.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Norm`, :py:func:`~pycsou.func.loss.SquaredL1Loss`.
    """
    L1_norm = L1Norm(dim=dim)
    return ProximableLoss(L1_norm, data=data)


def SquaredL1Loss(dim: int, data: Union[Number, np.ndarray], prox_computation='sort') -> ProximableFunctional:
    r"""
    :math:`\ell^2_1` loss functional, :math:`F(\mathbf{y},\mathbf{x}):=\|\mathbf{y}-\mathbf{x}\|^2_1`.

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell^2_1` loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import SquaredL1Loss
       from pycsou.func.penalty import SquaredL1Norm

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = SquaredL1Loss(dim=y.size, data=y)
       >>> func = SquaredL1Norm(dim=y.size)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    See Also
    --------
    :py:func:`~pycsou.func.penalty.SquaredL1Norm`, :py:func:`~pycsou.func.loss.L1Loss`.
    """
    squared_L1_norm = SquaredL1Norm(dim=dim, prox_computation=prox_computation)
    return ProximableLoss(squared_L1_norm, data=data)


def L1BallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    r"""
    :math:`\ell_1`-ball loss functional, :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{y}-\mathbf{x}\|_1\leq \text{radius}\}`.

    The :math:`\ell_1`-ball loss functional is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}-\mathbf{y}\|_1\leq \text{radius},\\
         \, 0\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.
    radius: Number
        Radius of the ball.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_1`-ball loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import L1BallLoss
       from pycsou.func.penalty import L1Ball

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = L1BallLoss(dim=y.size, data=y, radius=2)
       >>> func = L1Ball(dim=y.size, radius=2)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The :math:`\ell_1`-ball loss functional is particularly useful in the context of salt-and-pepper noise with
    known standard deviation. In which case, the :math:`\ell_1`-ball defines a confidence region for the data :math:`\mathbf{y}`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:func:`~pycsou.func.loss.L1Loss`, :py:func:`~pycsou.func.loss.SquaredL1Loss`.
    """
    L1_ball = L1Ball(dim=dim, radius=radius)
    return ProximableLoss(L1_ball, data=data)


def LInftyLoss(dim: int, data: Union[Number, np.ndarray]) -> ProximableFunctional:
    r"""
    :math:`\ell_\infty` loss functional, :math:`F(\mathbf{y},\mathbf{x}):=\|\mathbf{y}-\mathbf{x}\|_\infty`.

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_\infty` loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import LInftyLoss
       from pycsou.func.penalty import LInftyNorm

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = LInftyLoss(dim=y.size, data=y)
       >>> func = LInftyNorm(dim=y.size)
       >>> x = 2 * np.arange(10)
       >>> loss(x)
       9
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The :math:`\ell_\infty` loss functional is particularly useful in the context of quantisation noise, or more generally
    for noise distributions with compact support.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.LInftyNorm`, :py:func:`~pycsou.func.loss.LInftyBallLoss`.
    """
    LInfty_norm = LInftyNorm(dim=dim)
    return ProximableLoss(LInfty_norm, data=data)


def LInftyBallLoss(dim: int, data: Union[Number, np.ndarray], radius: Number = 1) -> ProximableFunctional:
    r"""
    :math:`\ell_\infty`-ball loss functional, :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{y}-\mathbf{x}\|_\infty\leq \text{radius}\}`.

    The :math:`\ell_1`-ball loss functional is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\|\mathbf{x}-\mathbf{y}\|_\infty\leq \text{radius},\\
         \, 0\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}`.
    radius: Number
        Radius of the ball.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The :math:`\ell_\infty`-ball loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import LInftyBallLoss
       from pycsou.func.penalty import LInftyBall

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = LInftyBallLoss(dim=y.size, data=y, radius=2)
       >>> func = LInftyBall(dim=y.size, radius=2)
       >>> x = 2 * np.arange(10)
       >>> np.allclose(loss.prox(x, tau=1), func.prox(x-y, tau=1) + y)
       True

    Notes
    -----
    The :math:`\ell_\infty`-ball loss functional is particularly useful in the context of quantisation noise with
    compact support. In which case, the :math:`\ell_\infty`-ball defines a confidence region for the data :math:`\mathbf{y}`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.LInftyBall`, :py:func:`~pycsou.func.loss.LInftyLoss`, :py:func:`~pycsou.func.penalty.LInftyNorm`.
    """
    LInfty_ball = LInftyBall(dim=dim, radius=radius)
    return ProximableLoss(LInfty_ball, data=data)


def ConsistencyLoss(dim: int, data: Union[Number, np.ndarray]):
    r"""
    Consistency loss functional :math:`\mathbf{y}=\mathbf{x}`.

    The consistency loss functional is defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}=\mathbf{y},\\
         \, 0\,\text{ortherwise}.
         \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}` to match.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The consistency loss functional.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import ConsistencyLoss

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = ConsistencyLoss(dim=y.size, data=y)
       >>> x = 2 * np.arange(10)
       >>> loss(x), loss(y)
       (inf, 0)
       >>> np.allclose(loss.prox(x, tau=1), y)
       True

    Notes
    -----
    This functional enforces an exact match between the predicted and observed samples, as required in interpolation problems.
    Such a functional is mainly useful in the context of noiseless data as it can lead to serious overfitting issues in the presence of noise.

    """
    condition_func = lambda x: np.allclose(x, data)
    projection_func = lambda x: data
    return IndicatorFunctional(dim=dim, condition_func=condition_func, projection_func=projection_func)


class KLDivergence(ProximableFunctional):
    r"""
    Generalised Kullback-Leibler divergence :math:`D_{KL}(\mathbf{y}||\mathbf{x}):=\sum_{i=1}^N y_i\log(y_i/x_i) -y_i +z_i`.

    The generalised Kullback-Leibler divergence is defined as:

    .. math::

       D_{KL}(\mathbf{y}||\mathbf{x}):=\sum_{i=1}^N H(y_i,x_i) -y_i +z_i, \quad \forall \mathbf{y}, \mathbf{x} \in \mathbb{R}^N,

    where

    .. math::

       H(y,x):=\begin{cases}
       y\log(y/x) &\, \text{if} \,x>0, y>0,\\
       0&\, \text{if} \,x=0, y\geq 0,\\
       +\infty &\,\text{otherwise.}
       \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}` to match.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The KL-divergence.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import KLDivergence

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = KLDivergence(dim=y.size, data=y)
       >>> x = 2 * np.arange(10)
       >>> loss(x)
       13.80837687480246
       >>> np.round(loss.prox(x, tau=1))
       array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])

    Notes
    -----
    In information theory, and in the case where :math:`\mathbf{y}` and :math:`\mathbf{x}`  sum to one  --and hence can be interpreted as discrete probability distributions,
    the KL-divergence can be interpreted as the relative entropy of :math:`\mathbf{y}` w.r.t. :math:`\mathbf{x}`,
    i.e. the amount of information lost when using :math:`\mathbf{x}` to approximate :math:`\mathbf{y}`.
    It is particularly useful in the context of count data with Poisson distribution. Indeed, the KL-divergence corresponds
    –up to an additive constant– to the likelihood of the data :math:`\mathbf{y}` where each component is independent
    with Poisson distribution and respective intensities given by the entries of :math:`\mathbf{x}`.
    See [FuncSphere]_ Section 5 of Chapter 7 for the computation of its proximal operator.

    See Also
    --------
    :py:class:`~pycsou.func.penalty.ShannonEntropy`, :py:class:`~pycsou.func.penalty.LogBarrier`
    """

    def __init__(self, dim: int, data: Union[Number, np.ndarray]):
        super(KLDivergence, self).__init__(dim=dim, data=None, is_differentiable=False, is_linear=False)
        self.data = data

    def __call__(self, x: Union[Number, np.ndarray]) -> Number:
        z = 0 * x + np.infty
        z[(x > 0) * (self.data > 0)] = self.data[(x > 0) * (self.data > 0)] * np.log(
            self.data[(x > 0) * (self.data > 0)] / x[(x > 0) * (self.data > 0)])
        z[(x == 0) * (self.data >= 0)] = 0
        return np.sum(z - self.data + x)

    def prox(self, x: Union[Number, np.ndarray], tau: Number) -> Union[Number, np.ndarray]:
        r"""
        Proximal operator of the KL-divergence functional (see [FuncSphere]_ Section 5 of Chapter 7).

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
        return (x - tau + np.sqrt((x - tau) ** 2 + 4 * tau * self.data)) / 2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
