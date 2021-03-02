# #############################################################################
# green.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Green functions of common pseudo-differential operators.
"""

from typing import Literal, Union
from numbers import Number
import numpy as np


class Matern:
    r"""
    Matern function for :math:`k\in\{0,1,2,3\}`.

    Examples
    --------

    Notes
    -----

    The Matern function is defined in full generality as ([GaussProcesses]_, eq (4.14)):

    .. math::

       S_\nu^\epsilon(r) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu} r}{\epsilon}\right)^\nu K_{\nu}\left(\frac{\sqrt{2\nu} r}{\epsilon}\right), \qquad \forall r\in\mathbb{R}_+,

    with :math:`\nu, \epsilon>0`,  :math:`\Gamma` and :math:`K_\nu`  are the Gamma and modified Bessel function of the second kind, respectively.
    The parameter :math:`nu` determines the smoothness of the Matern function (the higher, the smoother).
    The parameter :math:`epsilon` determines the localisation of the Matern function (the higher, the more localised).
    For :math:`\nu\in\mathbb{N}+1/2` the above equation simplifies to:

    .. math::

       S_{k+1/2}^\epsilon(r)=\exp\left(-\frac{\sqrt{2\nu} r}{\epsilon}\right) \frac{k!}{(2k)!} \sum_{i=0}^{k} \frac{(k+i)!}{i!(k-i)!}\left(\frac{\sqrt{8\nu}r}{\epsilon}\right)^{k-i}, , \qquad \forall r\in\mathbb{R}_+,

    with :math:`k\in \mathbb{N}`. This class provides the Matern function for :math:`k\in\{0,1,2,3\}` (Matern functions with :math:`k>3` are nearly indistinguishable from a Gaussian function
    with standard deviation :math:`\epsilon`). The Matern radial basis function :math:`S_{\nu}^\epsilon(\|\cdot\|)` in :math:`\mathbb{R}^d` is proportional to the Green function of the pseudo-differential operator
    :math:`\left(\mbox{Id} - \frac{\epsilon^2}{2\nu}\Delta_{\mathbb{R}^d}\right)^{2\nu+d}`, i.e. :math:`\left(\mbox{Id} - \frac{\epsilon^2}{2\nu}\Delta_{\mathbb{R}^d}\right)^{2\nu+d}S_{\nu}^\epsilon(\|\cdot\|)\propto \delta`.
    """
    def __init__(self, k: Literal[0, 1, 2, 3], epsilon: float = 1.):
        self.k = k
        self.epsilon = epsilon

    def __call__(self, r: Union[Number, np.ndarray]) -> np.ndarray:
        if self.k == 0:
            y = np.exp(-r / self.epsilon)
        elif self.k == 1:
            y = (1 + np.sqrt(3) * r / self.epsilon) * np.exp(-np.sqrt(3) * r / self.epsilon)
        elif self.k == 2:
            y = (1 + np.sqrt(5) * r / self.epsilon + (5 * r ** 2) / (3 * self.epsilon ** 2)) \
                * np.exp(-np.sqrt(5) * r / self.epsilon)
        elif self.k == 3:
            y = (1 + np.sqrt(7) * r / self.epsilon + (42 * r ** 2) / (15 * self.epsilon ** 2)
                 + (7 * np.sqrt(7) * r ** 3) / (15 * self.epsilon ** 3)) \
                * np.exp(-np.sqrt(7) * r / self.epsilon)
        return y

    def halfsupport(self, sigmas: int = 3):
        return sigmas * self.epsilon


class Wendland:
    pass


class GreenIteratedDerivative:
    pass


class GreenExponential:
    pass


class SubGaussian:
    pass
