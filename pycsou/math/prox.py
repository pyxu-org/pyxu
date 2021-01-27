# #############################################################################
# prox.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Common proximal/projection operators.
"""

import numpy as np
from typing import Union
from numbers import Number
import scipy.optimize as sciop


def sign(x: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
    r"""
    Sign function.

    The sign function is defined as:

    .. math::
       sign(x)=\begin{cases}
       \frac{\bar{x}}{|x|} & x\in\mathbb{C}\backslash\{0\},\\
       0 & \text{if} \,x=0.
       \end{cases}

    We have in particular: :math:`sign(x)x=|x|.`

    Parameters
    ----------
    x: Union[np.ndarray, Number]
        Input array.

    Returns
    -------
    Union[np.ndarray, Number]
        An array whose entries are given by the signs of the entries of ``x``.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.math.prox import sign

    .. doctest::

       >>> x = np.linspace(-1, 1, 5)
       >>> sign(x)
       array([-1., -1.,  0.,  1.,  1.])
       >>> np.allclose(sign(x) * x, np.abs(x))
       True
       >>> x = x + 1j
       >>> np.allclose(sign(x) * x, np.abs(x))
       True

    """
    x = np.asarray(x)
    y = np.asarray(0 * x)
    y[np.abs(x) != 0] = np.conj(x[np.abs(x) != 0]) / np.abs(x[np.abs(x) != 0])
    return y


def soft(x: Union[np.ndarray, Number], tau: Number) -> Union[np.ndarray, Number]:
    r"""
    Soft thresholding operator.

    The soft thresholding operator is defined as:

    .. math::

       \text{soft}_\tau(x)(x)=\max\{|x|-\tau, 0\} \text{sign}(x), \quad x\in\mathbb{C},

    where :math:`\tau\geq 0` and :math:`sign:\mathbb{C}\rightarrow \{-1,1,0\}` is the *sign* function (see
    :py:func:`~pycsou.math.prox.sign`).

    Parameters
    ----------
    x: Union[np.ndarray, Number]
        Input array.
    tau: Number
        Threshold value.

    Returns
    -------
    Union[np.ndarray, Number]
        Array ``x`` with element-wise soft thresholded entries.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.math.prox import soft

    .. doctest::

       >>> x = np.linspace(-1, 1, 5)
       >>> soft(x, tau=0.5)
       array([-0.5, -0. ,  0. ,  0. ,  0.5])
       >>> x = 3 + 1j
       >>> soft(x, tau=0.1)
       (2.905131670194949-0.9683772233983162j)

    Notes
    -----
    The soft thresholding operator is the proximal operator of the :math:`\ell_1` norm.
    See :py:class:`~pycsou.func.penalty.L1Norm`.
    """
    return np.clip(np.abs(x) - tau, a_min=0, a_max=None) * sign(x)


def proj_l1_ball(x: np.ndarray, radius: Number) -> np.ndarray:
    r"""
    Orthogonal projection onto the :math:`\ell_1`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_1\leq \text{radius}\}`.

    Parameters
    ----------
    x: np.ndarray
        Vector to be projected.
    radius: Number
        Radius of the :math:`\ell_1`-ball.

    Returns
    -------
    np.ndarray
        Projection of ``x`` onto the :math:`\ell_1`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.math.prox import proj_l1_ball

    .. doctest::

       >>> x = np.linspace(-1, 1, 5)
       >>> proj_l1_ball(x, radius=2)
       array([-0.75, -0.25,  0.  ,  0.25,  0.75])
       >>> np.linalg.norm(proj_l1_ball(x, radius=2), ord=1)
       2.0

    Notes
    -----
    The projection onto the :math:`\ell_1`-ball is described in [ProxAlg]_ Section 6.5.2.
    Note that this is also the proximal operator of the :math:`\ell_1`-ball functional :py:func:`~pycsou.func.penalty.L1Ball`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L1Ball`, :py:func:`~pycsou.math.prox.proj_l2_ball`, :py:func:`~pycsou.math.prox.proj_linfty_ball`.
    """
    if np.sum(np.abs(x)) <= radius:
        return x
    else:
        mu_max = np.max(np.abs(x))
        func = lambda mu: np.sum(np.clip(np.abs(x) - mu, a_min=0, a_max=None)) - radius
        mu_star = sciop.brentq(func, a=0, b=mu_max)
        return soft(x, mu_star)


def proj_l2_ball(x: np.ndarray, radius: Number) -> np.ndarray:
    r"""
    Orthogonal projection onto the :math:`\ell_2`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_2\leq \text{radius}\}`.

    Parameters
    ----------
    x: np.ndarray
        Vector to be projected.
    radius: Number
        Radius of the :math:`\ell_2`-ball.

    Returns
    -------
    np.ndarray
        Projection of ``x`` onto the :math:`\ell_2`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.math.prox import proj_l2_ball

    .. doctest::

       >>> x = np.linspace(-2, 2, 5)
       >>> np.allclose(proj_l2_ball(x, radius=1), x/np.linalg.norm(x))
       True
       >>> np.linalg.norm(proj_l2_ball(x, radius=1), ord=2)
       1.0

    Notes
    -----
    Note that this is also the proximal operator of the :math:`\ell_2`-ball functional :py:func:`~pycsou.func.penalty.L2Ball`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.L2Ball`, :py:func:`~pycsou.math.prox.proj_l1_ball`, :py:func:`~pycsou.math.prox.proj_linfty_ball`.
    """
    if np.linalg.norm(x) <= radius:
        return x
    else:
        return radius * x / np.linalg.norm(x)


def proj_linfty_ball(x: np.ndarray, radius: Number) -> np.ndarray:
    r"""
    Orthogonal projection onto the :math:`\ell_\infty`-ball :math:`\{\mathbf{x}\in\mathbb{R}^N: \|\mathbf{x}\|_\infty\leq \text{radius}\}`.

    Parameters
    ----------
    x: np.ndarray
        Vector to be projected.
    radius: Number
        Radius of the :math:`\ell_\infty`-ball.

    Returns
    -------
    np.ndarray
        Projection of ``x`` onto the :math:`\ell_\infty`-ball.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.math.prox import proj_linfty_ball

    .. doctest::

       >>> x = np.linspace(-2, 2, 5)
       >>> proj_linfty_ball(x, radius=2)
       array([-2., -1.,  0.,  1.,  2.])
       >>> np.linalg.norm(proj_linfty_ball(x, radius=2), ord=np.inf)
       2.0

    Notes
    -----
    Note that this is also the proximal operator of the :math:`\ell_\infty`-ball functional :py:func:`~pycsou.func.penalty.LInftyBall`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.LInftyBall`, :py:func:`~pycsou.math.prox.proj_l1_ball`, :py:func:`~pycsou.math.prox.proj_l2_ball`.
    """
    y = x
    y[y > radius] = radius
    y[y < -radius] = -radius
    return y


def proj_nonnegative_orthant(x: np.ndarray) -> np.ndarray:
    r"""
    Orthogonal projection on the non negative orthant.

    Parameters
    ----------
    x: np.ndarray
        Vector to be projected.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.math.prox import proj_nonnegative_orthant

    .. doctest::

       >>> x = np.linspace(-1, 1, 2)
       >>> proj_nonnegative_orthant(x)
       array([0., 1.])

    Returns
    -------
    np.ndarray
        Projection onto non negative orthant: negative entries of ``x`` are set to zero.

    Notes
    -----
    This is also the proximal operator of the indicator functional :py:func:`~pycsou.func.penalty.NonNegativeOrthant`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.NonNegativeOrthant`, :py:func:`~pycsou.math.prox.proj_segment`.

    """
    y = np.real(x)
    y[y < 0] = 0
    return y


def proj_segment(x: np.ndarray, a: Number = 0, b: Number = 1) -> np.ndarray:
    r"""
    Orthogonal projection into a real segment.

    Parameters
    ----------
    x: np.ndarray
        Vector to be projected.
    a: Number
        Left endpoint of the segement.
    b: Number
        Right endpoint of the segment.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.math.prox import proj_segment

    .. doctest::

       >>> x = np.linspace(-3, 3, 5)
       >>> proj_segment(x, a=-2,b=1)
       array([-2. , -1.5,  0. ,  1. ,  1. ])

    Returns
    -------
    np.ndarray
        Projection onto non negative orthant: negative entries of ``x`` are set to zero.

    Notes
    -----
    This is also the proximal operator of the indicator functional :py:func:`~pycsou.func.penalty.Segment`.

    See Also
    --------
    :py:func:`~pycsou.func.penalty.Segment`, :py:func:`~pycsou.math.prox.proj_nonnegative_orthant`.

    """
    y = np.real(x)
    y[y < a] = a
    y[y > b] = b
    return y


if __name__ == "__main__":
    import doctest

    doctest.testmod()
