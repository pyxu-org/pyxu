# #############################################################################
# green.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Green functions of common pseudo-differential operators.
"""

from typing import Union
from numbers import Number
import numpy as np


class Matern:
    r"""
    Matern function for :math:`k\in\{0,1,2,3\}`.

    Examples
    --------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pycsou.math.green import Matern

       x=np.linspace(-3,3,1024)
       for k in range(4):
          matern=Matern(k)
          plt.plot(x, matern(np.abs(x)))
       plt.legend(['k=0', 'k=1', 'k=2', 'k=3'])

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

       S_{k+1/2}^\epsilon(r)=\exp\left(-\frac{\sqrt{2\nu} r}{\epsilon}\right) \frac{k!}{(2k)!} \sum_{i=0}^{k} \frac{(k+i)!}{i!(k-i)!}\left(\frac{\sqrt{8\nu}r}{\epsilon}\right)^{k-i}, \qquad \forall r\in\mathbb{R}_+,

    with :math:`k\in \mathbb{N}`. This class provides the Matern function for :math:`k\in\{0,1,2,3\}` (Matern functions with :math:`k>3` are nearly indistinguishable from a Gaussian function
    with standard deviation :math:`\epsilon`). The Matern radial basis function :math:`S_{\nu}^\epsilon(\|\cdot\|)` in :math:`\mathbb{R}^d` is proportional to the Green function of the pseudo-differential operator
    :math:`\left(\mbox{Id} - \frac{\epsilon^2}{2\nu}\Delta_{\mathbb{R}^d}\right)^{2\nu+d}`, i.e. :math:`\left(\mbox{Id} - \frac{\epsilon^2}{2\nu}\Delta_{\mathbb{R}^d}\right)^{2\nu+d}S_{\nu}^\epsilon(\|\cdot\|)\propto \delta`.

    See Also
    --------
    :py:class:`~pycsou.math.green.Wendland`
    """

    def __init__(self, k: int, epsilon: float = 1.):
        r"""

        Parameters
        ----------
        k: int, Literal[0, 1 ,2 ,3]
            Order of the Matern function.
        epsilon: float
            Spread of the Matern function (support is approximately `3*epsilon`).
        """
        if k not in [0, 1, 2, 3]:
            raise TypeError('Parameter k must be one of [0,1,2,3].')
        self.k = k
        self.epsilon = epsilon

    def __call__(self, r: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
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

    def support(self, sigmas: int = 3) -> float:
        r"""
        Effective support of the Matern function using a Gaussian approximation.

        The approximation is poor for :math:`k=0`.

        Parameters
        ----------
        sigmas: int
            Number of sigmas defining the support of the approximating Gaussian function.

        Returns
        -------
        float
            Approximate effective support of the Matern function.
        """
        return sigmas * self.epsilon


class Wendland:
    r"""
    Wendland functions for :math:`k\in\{0, 1, 2, 3\}`.

    Examples
    --------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pycsou.math.green import Wendland

       x=np.linspace(-1,1,1024)
       for k in range(4):
          wendland=Wendland(k)
          plt.plot(x, wendland(np.abs(x)))
       plt.legend(['k=0', 'k=1', 'k=2', 'k=3'])

    Notes
    -----
    Wendland functions are constructed by repeatedly applying an integral operator to *Askey's truncated power functions*.
    They can be shown to take the form:

    .. math::
       \phi_{d,k}(r)=\left(1-\frac{r}{\epsilon}\right)_+^{l + k} p_{k,l}(r),\qquad r\in\mathbb{R}_+,

    where :math:`l=\lfloor d/2\rfloor+k+1`, :math:`a_+=\max(a,0)` and :math:`p_{k,l}` is a polynomial of degree :math:`k` whose coefficients depend
    on :math:`l`. These functions are *compactly supported piecewise polynomials* with support :math:`[0,\epsilon)` which
    yield **positive definite** radial kernels in :math:`\mathbb{R}^d` with *minimal degree* and *prescribed smoothness*.

    This class provides an implementation of Wendland functions using the closed-form forumlae in [FuncSphere]_ Figure 8.2  for :math:`k\in\{0, 1 ,2 ,3\}`.

    See Also
    --------
    :py:class:`~pycsou.math.green.Matern`
    """

    def __init__(self, k: int, epsilon: float = 1.):
        r"""

        Parameters
        ----------
        k: int
            Order of the Wendland function.
        epsilon: float
            Support of the Wendland function
        """
        if k not in [0, 1, 2, 3]:
            raise TypeError('Parameter k must be one of [0,1,2,3].')
        self.k = k
        self.epsilon = epsilon
        self.support = epsilon

    def __call__(self, r: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
        truncated_power_function = np.clip(1 - r / self.epsilon, a_min=0, a_max=None)
        if self.k == 0:
            y = truncated_power_function ** 2
        elif self.k == 1:
            y = (truncated_power_function ** 4) * (1 + 4 * r / self.epsilon)
        elif self.k == 2:
            y = (truncated_power_function ** 6) * (1 + 6 * r / self.epsilon + 35 * r ** 2 / (3 * self.epsilon ** 2))
        elif self.k == 3:
            y = (truncated_power_function ** 8) * (1 + 8 * r / self.epsilon + 25 * r ** 2 / (self.epsilon ** 2) +
                                                   32 * r ** 3 / (self.epsilon ** 3))
        return y

    def support(self):
        return self.support


class CausalGreenIteratedDerivative:
    r"""
    Causal Green function of the iterated derivative :math:`D^k`.

    Examples
    --------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pycsou.math.green import CausalGreenIteratedDerivative

       x=np.linspace(-1,1,1024)
       for k in range(1,5):
          green=CausalGreenIteratedDerivative(k)
          plt.plot(x, green(x))
       plt.legend(['k=1', 'k=2', 'k=3', 'k=4'])

    Notes
    -----
    The Green function :math:`\rho_k` of the iterated derivative operator :math:`D^k` on  :math:`\mathbb{R}` is defined as
    :math:`D^k\rho_k=\delta`. It is given by: :math:`\rho_k(t)\propto t^{k-1}_+`.

    See Also
    --------
    :py:class:`~pycsou.math.green.CausalGreenExponential`
    """

    def __init__(self, k: int):
        r"""

        Parameters
        ----------
        k: int
            Exponent :math:`k` of the iterated derivative operator :math:`D^k` for which the Green function is desired.
        """
        self.k = k

    def __call__(self, x: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
        return x ** (self.k - 1) * (x >= 0)


class CausalGreenExponential:
    r"""
    Causal Green function of :math:`(D+\alpha\mbox{Id})^k`.

    Examples
    --------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pycsou.math.green import CausalGreenExponential

       x=np.linspace(-1,5,1024)
       for k in range(1,5):
          green=CausalGreenExponential(k)
          plt.plot(x, green(x))
       plt.legend(['k=1', 'k=2', 'k=3', 'k=4'])

    Notes
    -----
    The Green function :math:`\rho_k` of the 1D pseudo-differential operator :math:`(D+\alpha\mbox{Id})^k` on  :math:`\mathbb{R}` is defined as
    :math:`(D+\alpha\mbox{Id})^k\rho_k=\delta`. It is given by: :math:`\rho_k(t)\propto t_+^{k-1}e^{-\alpha t}`.

    See Also
    --------
    :py:class:`~pycsou.math.green.CausalGreenIteratedDerivative`
    """

    def __init__(self, k: int, alpha: float = 1):
        R"""

        Parameters
        ----------
        k: int
            Exponent :math:`k` of the exponential pseudo-differential operator :math:`(D+\alpha\mbox{Id})^k` for which the Green function is desired.
        alpha: float
            Strictly positive parameter :math:`\alpha` of the exponential pseudo-differential operator :math:`(D+\alpha\mbox{Id})^k` for which the Green function is desired.
        """
        if alpha <= 0:
            raise TypeError('Parameter alpha must be strictly positive.')
        self.k = k
        self.alpha = alpha

    def __call__(self, x: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
        return x ** (self.k - 1) * np.exp(-self.alpha * x) * (x >= 0)


class SubGaussian:
    r"""
    Sub-Gaussian function.

    Examples
    --------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       from pycsou.math.green import SubGaussian

       x=np.linspace(-3,3,1024)
       lg=[]
       for alpha in np.linspace(0,2,6)[1:-1]:
          subgaussian=SubGaussian(alpha)
          plt.plot(x, subgaussian(np.abs(x)))
          lg.append(f'$\\alpha={np.round(alpha,1)}$')
       plt.legend(lg)

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.sampling import MappedDistanceMatrix
       from pycsou.math.green import SubGaussian

       t = np.linspace(0, 2, 256)
       rng = np.random.default_rng(seed=2)
       x,y = np.meshgrid(t,t)
       samples1 = np.stack((x.flatten(), y.flatten()), axis=-1)
       samples2 = np.stack((2 * rng.random(size=4), 2 * rng.random(size=4)), axis=-1)
       alpha = np.ones(samples2.shape[0])
       ord=.8
       epsilon = 1 / 12
       func = SubGaussian(alpha=ord, epsilon=epsilon)
       MDMOp = MappedDistanceMatrix(samples1=samples1, samples2=samples2, function=func, ord=ord, operator_type='dask')
       plt.contourf(x,y,(MDMOp * alpha).reshape(t.size, t.size), 50)
       plt.title('Sum of 4 (radial) sub-Gaussians')
       plt.colorbar()
       plt.xlabel('$x$')
       plt.ylabel('$y$')

    Notes
    -----
    The sub-Gaussian function is defined, for every :math:`\alpha\in (0,2)` and :math:`\epsilon>0`, as:

    .. math::

       f_\epsilon(r)=\exp\left(-\frac{r^\alpha}{\epsilon}\right), \qquad r\in \mathbb{R}_+.

    It can be used to build sub-Gaussian radial kernels [SubGauss]_, which are known to be *positive-definite* and their inverse Fourier transforms (the so-called α-stable distributions)
    are heavy-tailed and infinitely smooth with algebraically decaying derivatives of any order.
    """

    def __init__(self, alpha: float, epsilon: float = 1.):
        r"""

        Parameters
        ----------
        alpha: float
            Exponent :math:`\alpha\in (0,2)` of the sub-Gaussian function.
        epsilon: float
            Spread :math:`\epsilon>0` of the sub-Gaussian function.
        """
        if (alpha <= 0) or (alpha >= 2):
            raise TypeError('Parameter alpha must be in (0,2).')
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, r: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
        return np.exp(-r ** self.alpha / self.epsilon)
