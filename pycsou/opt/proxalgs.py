# #############################################################################
# solver.py
# =========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Proximal algorithms.

This module provides various proximal algorithms for convex optimisation.
"""

from numbers import Number
from typing import Optional, Tuple, Any

import numpy as np
from pandas import DataFrame

from pycsou.core.functional import ProximableFunctional, DifferentiableFunctional
from pycsou.core.linop import LinearOperator
from pycsou.core.map import DifferentiableMap
from pycsou.core.solver import GenericIterativeAlgorithm
from pycsou.func.base import NullDifferentiableFunctional, NullProximableFunctional
from pycsou.linop.base import IdentityOperator, NullOperator


class PrimalDualSplitting(GenericIterativeAlgorithm):
    r"""
    Primal dual splitting algorithm.

    This class is also accessible via the alias ``PDS()``.

    Notes
    -----
    The *Primal Dual Splitting (PDS)* method is described in [PDS]_ (this particular implementation is based on the pseudo-code Algorithm 7.1 provided in [FuncSphere]_ Chapter 7, Section1).
    It can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *linear operator*, with **operator norm**:

      .. math::
         \Vert{\mathbf{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathbf{K}\mathbf{x}\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    Assume that the following holds:

    * :math:`\beta>0` and:

      - :math:`\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\geq \frac{\beta}{2}`,
      - :math:`\rho \in ]0,\delta[`, where :math:`\delta:=2-\frac{\beta}{2}\left(\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\right)^{-1}\in[1,2[.`

    * or :math:`\beta=0` and:

      - :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2\leq 1`
      - :math:`\rho \in [\epsilon,2-\epsilon]`, for some  :math:`\epsilon>0.`

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M`} solution s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively, i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    **Default values of the hyperparameters provided here always ensure convergence of the algorithm.**

    Examples
    --------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.`
    This problem can be solved via PDS with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`, :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,`
    :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1` and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pycsou.linop.diff import FirstDerivative
        from pycsou.func.loss import SquaredL2Loss
        from pycsou.func.penalty import L1Norm, NonNegativeOrthant
        from pycsou.linop.sampling import DownSampling
        from pycsou.opt.proxalgs import PrimalDualSplitting

        x = np.repeat([0, 2, 1, 3, 0, 2, 0], 10)
        D = FirstDerivative(size=x.size, kind='forward')
        D.compute_lipschitz_cst(tol=1e-3)
        rng = np.random.default_rng(0)
        G = DownSampling(size=x.size, downsampling_factor=3)
        G.compute_lipschitz_cst()
        y = G(x)
        l22_loss = (1 / 2) * SquaredL2Loss(dim=G.shape[0], data=y)
        F = l22_loss * G
        lambda_ = 0.1
        H = lambda_ * L1Norm(dim=D.shape[0])
        G = 0.01 * L1Norm(dim=G.shape[1])
        pds = PrimalDualSplitting(dim=G.shape[1], F=F, G=G, H=H, K=D, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        plt.figure()
        plt.stem(x, linefmt='C0-', markerfmt='C0o')
        plt.stem(estimate['primal_variable'], linefmt='C1--', markerfmt='C1s')
        plt.legend(['Ground truth', 'PDS Estimate'])
        plt.show()

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.PDS`, :py:class:`~pycsou.opt.proxalgs.ChambollePockSplitting`, :py:class:`~pycsou.opt.proxalgs.DouglasRachford`
    """

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 H: Optional[ProximableFunctional] = None, K: Optional[LinearOperator] = None,
                 tau: Optional[float] = None, sigma: Optional[float] = None, rho: Optional[float] = None,
                 beta: Optional[float] = None, x0: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
                 max_iter: int = 500, min_iter: int = 10, accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        F: Optional[DifferentiableMap]
            Differentiable map :math:`\mathcal{F}`.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        H: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{H}`.
        K: Optional[LinearOperator]
            Linear operator :math:`\mathbf{K}`.
        tau: Optional[float]
            Primal step size.
        sigma: Optional[float]
            Dual step size.
        rho: Optional[float]
            Momentum parameter.
        beta: Optional[float]
            Lipschitz constant :math:`\beta` of the derivative of :math:`\mathcal{F}`.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        z0: Optional[np.ndarray]
            Initial guess for the dual variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        self.dim = dim
        self._H = True
        if isinstance(F, DifferentiableMap):
            if F.shape[1] != dim:
                raise ValueError(f'F does not have the proper dimension: {F.shape[1]}!={dim}.')
            else:
                self.F = F
            if F.diff_lipschitz_cst < np.infty:
                self.beta = self.F.diff_lipschitz_cst if beta is None else beta
            elif (beta is not None) and isinstance(beta, Number):
                self.beta = beta
            else:
                raise ValueError('F must be a differentiable functional with Lipschitz-continuous gradient.')
        elif F is None:
            self.F = NullDifferentiableFunctional(dim=dim)
            self.beta = 0
        else:
            raise TypeError(f'F must be of type {DifferentiableMap}.')

        if isinstance(G, ProximableFunctional):
            if G.dim != dim:
                raise ValueError(f'G does not have the proper dimension: {G.dim}!={dim}.')
            else:
                self.G = G
        elif G is None:
            self.G = NullProximableFunctional(dim=dim)
        else:
            raise TypeError(f'G must be of type {ProximableFunctional}.')

        if isinstance(K, LinearOperator) and isinstance(H, ProximableFunctional):
            if (K.shape[1] != dim) or (K.shape[0] != H.dim):
                raise ValueError(
                    f'Operator K with shape {K.shape} is inconsistent with functional H with dimension {H.dim}.')

        if isinstance(H, ProximableFunctional):
            self.H = H
            if isinstance(K, LinearOperator):
                self.K = K
            elif K is None:
                self.K = IdentityOperator(size=H.dim)
                self.K.lipschitz_cst = self.K.diff_lipschitz_cst = 1
            else:
                raise TypeError(f'K must be of type {LinearOperator}.')
        elif H is None:
            self.H = NullProximableFunctional(dim=dim)
            self._H = False
            self.K = NullOperator(shape=(dim, dim))
            self.K.lipschitz_cst = self.K.diff_lipschitz_cst = 0
        else:
            raise TypeError(f'H must be of type {ProximableFunctional}.')

        if (tau is not None) and (sigma is not None):
            self.tau, self.sigma = tau, sigma
        elif (tau is not None) and (sigma is None):
            self.tau = self.sigma = tau
        elif (tau is None) and (sigma is not None):
            self.tau = self.sigma = sigma
        else:
            self.tau, self.sigma = self.set_step_sizes()

        if rho is not None:
            self.rho = rho
        else:
            self.rho = self.set_momentum_term()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_primal_variable()

        if z0 is not None:
            self.z0 = np.asarray(z0)
        else:
            self.z0 = self.initialize_dual_variable()

        objective_functional = (self.F + self.G) + (self.H * self.K)
        init_iterand = {'primal_variable': self.x0, 'dual_variable': self.z0}
        super(PrimalDualSplitting, self).__init__(objective_functional=objective_functional, init_iterand=init_iterand,
                                                  max_iter=max_iter, min_iter=min_iter,
                                                  accuracy_threshold=accuracy_threshold, verbose=verbose)

    def set_step_sizes(self) -> Tuple[float, float]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[float, float]
            Sensible primal/dual step sizes.

        Notes
        -----
        In practice, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and :math:`\tau` as large as possible and relatively well-balanced --so that both the primal and dual variables converge at the same pace. In practice, it is hence recommended to choose perfectly balanced parameters :math:`\sigma=\tau` saturating the convergence inequalities.

        For :math:`\beta>0` this yields:

        .. math::

           \frac{1}{\tau}-\tau\Vert\mathbf{K}\Vert_{2}^2= \frac{\beta}{2} \quad\Longleftrightarrow\quad -2\tau^2\Vert\mathbf{K}\Vert_{2}^2-\beta\tau+2=0,


        which admits one positive root

        .. math::

           \tau=\sigma=\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}\left(-\frac{\beta}{4}+\sqrt{\frac{\beta^2}{16}+\Vert\mathbf{K}\Vert_{2}^2}\right).


        For :math:`\beta=0`, this yields

        .. math::

           \tau=\sigma=\Vert\mathbf{K\Vert_{2}^{-1}.}

        """
        if self.beta > 0:
            if self._H is False:
                tau = 2 / self.beta
                sigma = 0
            else:
                if self.K.lipschitz_cst < np.infty:
                    tau = sigma = (1 / (self.K.lipschitz_cst) ** 2) * (
                            (-self.beta / 4) + np.sqrt((self.beta ** 2 / 16) + self.K.lipschitz_cst ** 2))
                else:
                    raise ValueError(
                        'Please compute the Lipschitz constant of the linear operator K by calling its method "compute_lipschitz_cst()".')
        else:
            if self._H is False:
                tau = 1
                sigma = 0
            else:
                if self.K.lipschitz_cst < np.infty:
                    tau = sigma = 1 / self.K.lipschitz_cst
                else:
                    raise ValueError(
                        'Please compute the Lipschitz constant of the linear operator K by calling its method "compute_lipschitz_cst()".')
        return tau, sigma

    def set_momentum_term(self) -> float:
        r"""
        Sets the momentum term.

        Returns
        -------
        float
            Momentum term.
        """
        if self.beta > 0:
            rho = 0.9
        else:
            rho = 1
        return rho

    def initialize_primal_variable(self) -> np.ndarray:
        """
        Initialize the primal variable to zero.

        Returns
        -------
        np.ndarray
            Zero-initialized primal variable.
        """
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def initialize_dual_variable(self) -> Optional[np.ndarray]:
        """
        Initialize the dual variable to zero.

        Returns
        -------
        np.ndarray
            Zero-initialized dual variable.
        """
        if self._H is False:
            return None
        else:
            return np.zeros(shape=(self.H.dim,), dtype=np.float)

    def update_iterand(self) -> dict:
        if self.iter == 0:
            x, z = self.init_iterand.values()
        else:
            x, z = self.iterand.values()
        x_temp = self.G.prox(x - self.tau * self.F.gradient(x) - self.tau * self.K.adjoint(z), tau=self.tau)
        if self._H is True:
            u = 2 * x_temp - x
            z_temp = self.H.fenchel_prox(z + self.sigma * self.K(u), sigma=self.sigma)
            z = self.rho * z_temp + (1 - self.rho) * z
        x = self.rho * x_temp + (1 - self.rho) * x
        iterand = {'primal_variable': x, 'dual_variable': z}
        return iterand

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        else:
            return self.diagnostics.loc[self.iter - 1, 'Relative Improvement (primal variable)']

    def update_diagnostics(self):
        if self._H is True:
            if self.iter == 0:
                self.diagnostics = DataFrame(
                    columns=['Iter', 'Relative Improvement (primal variable)', 'Relative Improvement (dual variable)'])
            self.diagnostics.loc[self.iter, 'Iter'] = self.iter
            if np.linalg.norm(self.old_iterand['primal_variable']) == 0:
                self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.infty
            else:
                self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.linalg.norm(
                    self.old_iterand['primal_variable'] - self.iterand['primal_variable']) / np.linalg.norm(
                    self.old_iterand['primal_variable'])
            if np.linalg.norm(self.old_iterand['dual_variable']) == 0:
                self.diagnostics.loc[self.iter, 'Relative Improvement (dual variable)'] = np.infty
            else:
                self.diagnostics.loc[self.iter, 'Relative Improvement (dual variable)'] = np.linalg.norm(
                    self.old_iterand['dual_variable'] - self.iterand['dual_variable']) / np.linalg.norm(
                    self.old_iterand['dual_variable'])
        else:
            if self.iter == 0:
                self.diagnostics = DataFrame(
                    columns=['Iter', 'Relative Improvement (primal variable)'])
            self.diagnostics.loc[self.iter, 'Iter'] = self.iter
            if np.linalg.norm(self.old_iterand['primal_variable']) == 0:
                self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.infty
            else:
                self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.linalg.norm(
                    self.old_iterand['primal_variable'] - self.iterand['primal_variable']) / np.linalg.norm(
                    self.old_iterand['primal_variable'])


PDS = PrimalDualSplitting


class AcceleratedProximalGradientDescent(GenericIterativeAlgorithm):
    r"""
    Accelerated proximal gradient descent.

    This class is also accessible via the alias ``APGD()``.

    Notes
    -----
    The *Accelerated Proximal Gradient Descent (APGD)* method can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and *convex function* with a *simple proximal operator*.
    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:** the algorithm is still valid if one or more of the terms :math:`\mathcal{F}` or :math:`\mathcal{G}` is zero.

    **Remark 2:**  The convergence is guaranteed for step sizes :math:`\tau\leq 1/\beta`. Without acceleration, APGD can be seen
    as a PDS method with :math:`\rho=1`. The various acceleration schemes are described in [APGD]_.
    For :math:`0<\tau\leq 1/\beta` and Chambolle and Dossal's acceleration scheme (``acceleration='CD'``), APGD achieves the following (optimal) *convergence rates*:

    .. math::

       \lim\limits_{n\rightarrow \infty} n^2\left\vert \mathcal{J}(\mathbf{x}^\star)- \mathcal{J}(\mathbf{x}_n)\right\vert=0\qquad \&\qquad \lim\limits_{n\rightarrow \infty} n^2\Vert \mathbf{x}_n-\mathbf{x}_{n-1}\Vert^2_\mathcal{X}=0,


    for *some minimiser* :math:`{\mathbf{x}^\star}\in\arg\min_{\mathbf{x}\in\mathbb{R}^N} \;\left\{\mathcal{J}(\mathbf{x}):=\mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})\right\}`.
    In other words, both the objective functional and the APGD iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` converge at a rate :math:`o(1/n^2)`. In comparison
    Beck and Teboule's acceleration scheme (``acceleration='BT'``) only achieves a convergence rate of :math:`O(1/n^2)`.
    Significant practical *speedup* can moreover be achieved for values of :math:`d` in the range  :math:`[50,100]`  [APGD]_.

    Examples
    --------
    Consider the *LASSO problem*:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda \|\mathbf{x}\|_1,

    with :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda>0.` This problem can be solved via APGD with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2` and :math:`\mathcal{G}(\mathbf{x})=\lambda \|\mathbf{x}\|_1`. We have:

    .. math::

       \mathbf{\nabla}\mathcal{F}(\mathbf{x})=\mathbf{G}^T(\mathbf{G}\mathbf{x}-\mathbf{y}), \qquad  \text{prox}_{\lambda\|\cdot\|_1}(\mathbf{x})=\text{soft}_\lambda(\mathbf{x}).

    This yields the so-called *Fast Iterative Soft Thresholding Algorithm (FISTA)*, whose convergence is guaranteed for :math:`d>2` and :math:`0<\tau\leq \beta^{-1}=\|\mathbf{G}\|_2^{-2}`.

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.func.loss import SquaredL2Loss
       from pycsou.func.penalty import L1Norm
       from pycsou.linop.base import DenseLinearOperator
       from pycsou.opt.proxalgs import APGD

       rng = np.random.default_rng(0)
       G = DenseLinearOperator(rng.standard_normal(15).reshape(3,5))
       G.compute_lipschitz_cst()
       x = np.zeros(G.shape[1])
       x[1] = 1
       x[-2] = -1
       y = G(x)
       l22_loss = (1/2) * SquaredL2Loss(dim=G.shape[0], data=y)
       F = l22_loss * G
       lambda_ = 0.9 * np.max(np.abs(F.gradient(0 * x)))
       G = lambda_ * L1Norm(dim=G.shape[1])
       apgd = APGD(dim=G.shape[1], F=F, G=G, acceleration='CD', verbose=None)
       estimate, converged, diagnostics = apgd.iterate()
       plt.figure()
       plt.stem(x, linefmt='C0-', markerfmt='C0o')
       plt.stem(estimate['iterand'], linefmt='C1--', markerfmt='C1s')
       plt.legend(['Ground truth', 'LASSO Estimate'])
       plt.show()

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.APGD`
    """

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 tau: Optional[float] = None, acceleration: Optional[str] = 'CD', beta: Optional[float] = None,
                 x0: Optional[np.ndarray] = None, max_iter: int = 500, min_iter: int = 10,
                 accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1, d: float = 75.):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        F: Optional[DifferentiableMap]
            Differentiable map :math:`\mathcal{F}`.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        tau: Optional[float]
            Primal step size.
        acceleration: Optional[str] [None, 'BT', 'CD']
            Which acceleration scheme should be used (`None` for no acceleration).
        beta: Optional[float]
            Lipschitz constant :math:`\beta` of the derivative of :math:`\mathcal{F}`.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        d: float
            Parameter :math:`d` for Chambolle and Dossal's acceleration scheme (``acceleration='CD'``).
        """
        self.dim = dim
        self.acceleration = acceleration
        self.d = d
        if isinstance(F, DifferentiableMap):
            if F.shape[1] != dim:
                raise ValueError(f'F does not have the proper dimension: {F.shape[1]}!={dim}.')
            else:
                self.F = F
            if F.diff_lipschitz_cst < np.infty:
                self.beta = self.F.diff_lipschitz_cst if beta is None else beta
            elif (beta is not None) and isinstance(beta, Number):
                self.beta = beta
            else:
                raise ValueError('F must be a differentiable functional with Lipschitz-continuous gradient.')
        elif F is None:
            self.F = NullDifferentiableFunctional(dim=dim)
            self.beta = 0
        else:
            raise TypeError(f'F must be of type {DifferentiableMap}.')

        if isinstance(G, ProximableFunctional):
            if G.dim != dim:
                raise ValueError(f'G does not have the proper dimension: {G.dim}!={dim}.')
            else:
                self.G = G
        elif G is None:
            self.G = NullProximableFunctional(dim=dim)
        else:
            raise TypeError(f'G must be of type {ProximableFunctional}.')

        if tau is not None:
            self.tau = tau
        else:
            self.tau = self.set_step_size()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_iterate()
        objective_functional = self.F + self.G
        init_iterand = {'iterand': self.x0, 'past_aux': 0 * self.x0, 'past_t': 1}
        super(AcceleratedProximalGradientDescent, self).__init__(objective_functional=objective_functional,
                                                                 init_iterand=init_iterand,
                                                                 max_iter=max_iter, min_iter=min_iter,
                                                                 accuracy_threshold=accuracy_threshold,
                                                                 verbose=verbose)

    def set_step_size(self) -> float:
        r"""
        Set the step size to its largest admissible value :math:`1/\beta`.

        Returns
        -------
        Tuple[float, float]
            Largest admissible step size.
        """
        return 1 / self.beta

    def initialize_iterate(self) -> np.ndarray:
        """
        Initialize the iterand to zero.

        Returns
        -------
        np.ndarray
            Zero-initialized iterand.
        """
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def update_iterand(self) -> Any:
        if self.iter == 0:
            x, x_old, t_old = self.init_iterand.values()
        else:
            x, x_old, t_old = self.iterand.values()
        x_temp = self.G.prox(x - self.tau * self.F.gradient(x), tau=self.tau)
        if self.acceleration == 'BT':
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        elif self.acceleration == 'CD':
            t = (self.iter + self.d) / self.d
        else:
            t = t_old = 1
        a = (t_old - 1) / t
        x = x_temp + a * (x_temp - x_old)
        iterand = {'iterand': x, 'past_aux': x_temp, 'past_t': t}
        return iterand

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        else:
            return self.diagnostics.loc[self.iter - 1, 'Relative Improvement']

    def update_diagnostics(self):
        if self.iter == 0:
            self.diagnostics = DataFrame(
                columns=['Iter', 'Relative Improvement'])
        self.diagnostics.loc[self.iter, 'Iter'] = self.iter
        if np.linalg.norm(self.old_iterand['iterand']) == 0:
            self.diagnostics.loc[self.iter, 'Relative Improvement'] = np.infty
        else:
            self.diagnostics.loc[self.iter, 'Relative Improvement'] = np.linalg.norm(
                self.old_iterand['iterand'] - self.iterand['iterand']) / np.linalg.norm(
                self.old_iterand['iterand'])


APGD = AcceleratedProximalGradientDescent


class ChambollePockSplitting(PrimalDualSplitting):
    r"""
    Chambolle and Pock primal-dual splitting method.

    This class is also accessible via the alias ``CPS()``.

    Notes
    -----
    The *Chambolle and Pock primal-dual splitting (CPS)* method can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M``` is a *linear operator*, with **operator norm**:

      .. math::
         \Vert{\mathbf{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathbf{K}\mathbf{x}\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    Assume that the following holds:

    - :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2\leq 1`
    - :math:`\rho \in [\epsilon,2-\epsilon]`, for some  :math:`\epsilon>0.`

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M`} solution s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively, i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    **Default values of the hyperparameters provided here always ensure convergence of the algorithm.**

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.CPS`, :py:class:`~pycsou.opt.proxalgs.PrimalDualSplitting`, :py:class:`~pycsou.opt.proxalgs.DouglasRachfordSplitting`
    """

    def __init__(self, dim: int, G: Optional[ProximableFunctional] = None,
                 H: Optional[ProximableFunctional] = None, K: Optional[LinearOperator] = None,
                 tau: Optional[float] = None, sigma: Optional[float] = None, rho: Optional[float] = 1,
                 x0: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
                 max_iter: int = 500, min_iter: int = 10, accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        H: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{H}`.
        K: Optional[LinearOperator]
            Linear operator :math:`\mathbf{K}`.
        tau: Optional[float]
            Primal step size.
        sigma: Optional[float]
            Dual step size.
        rho: Optional[float]
            Momentum parameter.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        z0: Optional[np.ndarray]
            Initial guess for the dual variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        super(ChambollePockSplitting, self).__init__(dim=dim, F=None, G=G, H=H, K=K, tau=tau, sigma=sigma, rho=rho,
                                                     x0=x0,
                                                     z0=z0, max_iter=max_iter, min_iter=min_iter,
                                                     accuracy_threshold=accuracy_threshold, verbose=verbose)


CPS = ChambollePockSplitting


class DouglasRachfordSplitting(PrimalDualSplitting):
    r"""
    Douglas Rachford splitting algorithm.

    This class is also accessible via the alias ``DRS()``.

    Notes
    -----
    The *Douglas Rachford Splitting (DRS)* can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{x}).}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Default values of the hyperparameters provided here always ensure convergence of the algorithm.**

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.DRS`, :py:class:`~pycsou.opt.proxalgs.PrimalDualSplitting`, :py:class:`~pycsou.opt.proxalgs.ChambollePockSplitting`
    """

    def __init__(self, dim: int, G: Optional[ProximableFunctional] = None,
                 H: Optional[ProximableFunctional] = None,
                 tau: float = 1., x0: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
                 max_iter: int = 500, min_iter: int = 10, accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        H: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{H}`.
        tau: Optional[float]
            Primal step size.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        z0: Optional[np.ndarray]
            Initial guess for the dual variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        super(DouglasRachfordSplitting, self).__init__(dim=dim, F=None, G=G, H=H, K=None, tau=tau, sigma=1 / tau, rho=1,
                                                       x0=x0, z0=z0, max_iter=max_iter, min_iter=min_iter,
                                                       accuracy_threshold=accuracy_threshold, verbose=verbose)


DRS = DouglasRachfordSplitting


class ForwardBackwardSplitting(PrimalDualSplitting):
    r"""
    Forward-backward splitting algorithm.

    This class is also accessible via the alias ``FBS()``.

    Notes
    -----
    The *Forward-backward splitting (FBS)* method can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is *proper*, *lower semicontinuous* and *convex function* with *simple proximal operator*.
    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{F}` or :math:`\mathcal{G}` is zero.

    **Remark 2:**

    Assume that the following holds:

      - :math:`\frac{1}{\tau}\geq \frac{\beta}{2}`,
      - :math:`\rho \in ]0,\delta[`, where :math:`\delta:=2-\frac{\beta}{2}\tau\in[1,2[.`

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M`} solution s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively, i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    **Default values of the hyperparameters provided here always ensure convergence of the algorithm.**

    See Also
    --------
    :py:class:`~pycsou.opt.proxalgs.FBS`, :py:class:`~pycsou.opt.proxalgs.AcceleratedProximalGradientDescent`
    """

    def __init__(self, dim: int, F: Optional[DifferentiableFunctional] = None, G: Optional[ProximableFunctional] = None,
                 tau: Optional[float] = None, rho: Optional[float] = 1, x0: Optional[np.ndarray] = None,
                 max_iter: int = 500, min_iter: int = 10, accuracy_threshold: float = 1e-3, verbose: Optional[int] = 1):
        r"""
        Parameters
        ----------
        dim: int
            Dimension of the objective functional's domain.
        F: Optional[DifferentiableMap]
            Differentiable map :math:`\mathcal{F}`.
        G: Optional[ProximableFunctional]
            Proximable functional :math:`\mathcal{G}`.
        tau: Optional[float]
            Primal step size.
        rho: Optional[float]
            Momentum parameter.
        beta: Optional[float]
            Lipschitz constant :math:`\beta` of the derivative of :math:`\mathcal{F}`.
        x0: Optional[np.ndarray]
            Initial guess for the primal variable.
        max_iter: int
            Maximal number of iterations.
        min_iter: int
            Minimal number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        super(ForwardBackwardSplitting, self).__init__(dim=dim, F=F, G=G, H=None, K=None, tau=tau, rho=rho,
                                                       x0=x0, max_iter=max_iter, min_iter=min_iter,
                                                       accuracy_threshold=accuracy_threshold, verbose=verbose)


FBS = ForwardBackwardSplitting

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pycsou.linop.diff import FirstDerivative
    from pycsou.func.loss import SquaredL2Loss
    from pycsou.func.penalty import L1Norm
    from pycsou.linop.sampling import DownSampling
    from pycsou.opt.proxalgs import PrimalDualSplitting

    x = np.repeat([0, 2, 1, 3, 0, 2, 0], 10)
    D = FirstDerivative(size=x.size, kind='forward')
    D.compute_lipschitz_cst(tol=1e-3)
    rng = np.random.default_rng(0)
    G = DownSampling(size=x.size, downsampling_factor=3)
    G.compute_lipschitz_cst()
    y = G(x)
    l22_loss = (1 / 2) * SquaredL2Loss(dim=G.shape[0], data=y)
    F = l22_loss * G
    lambda_ = 0.1
    H = lambda_ * L1Norm(dim=D.shape[0])
    G = 0.01 * L1Norm(dim=G.shape[1])
    pds = PrimalDualSplitting(dim=G.shape[1], F=F, G=G, H=H, K=D, verbose=None)
    estimate, converged, diagnostics = pds.iterate()
    plt.figure()
    plt.stem(x, linefmt='C0-', markerfmt='C0o')
    plt.stem(estimate['primal_variable'], linefmt='C1--', markerfmt='C1s')
    plt.legend(['Ground truth', 'PDS Estimate'])
    plt.show()
