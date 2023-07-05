r"""
This module implements Bayesian methods to estimate regularization parameters in inverse problems. Setting such
parameters if often a challenge in practice; this module aims to provide principled ways of setting them automatically.

**Example 1**

In this first example, we showcase the hierarchical Bayesian method
:py:class:`~pycsou.opt.solver.reg_param_est.RegParamMAP` that estimates regularization parameters via
maximum-a-posteriori estimation. We consider a denoising problem :math:`\mathbf{y}=\mathbf{x}_{\mathrm{GT}}+\mathbf{n}`
where :math:`\mathbf{y}` is the measured noisy image, :math:`\mathbf{x}_{\mathrm{GT}}` is the ground-truth image, and
:math:`\mathbf{n}` is additive i.i.d Gaussian noise with variance :math:`\sigma^2`. In Bayesian frameworks, one must
typically find the expression of the *posterior distribution* (conditional to :math:`\theta`), which, using Bayes’ rule,
is given by

.. math::
    p(\mathbf{x}|\mathbf{y},\theta) \propto p(\mathbf{y}|\mathbf{x}, \theta) p(\mathbf{x} | \theta),

where:

* :math:`\theta > 0` is the regularization parameter to be estimated. In *hierarchical Bayesian methods*, this
  parameter is modelled as a random variable defined via a hyper-prior distribution :math:`p(\theta)`.
* :math:`p(\mathbf{y}|\mathbf{x},\theta)` is the *likelihood* of the image :math:`\mathbf{x}`, which in an
  additive Gaussian noise model is given by :math:`p(\mathbf{y}|\mathbf{x},\theta) \propto \exp(- \frac{1}{2 \sigma^2}
  ||\mathbf{x} -\mathbf{y}||_2^2)`.
* :math:`p(\mathbf{x}|\theta)` is the *prior distribution*.

In this example, we assume an *isotropic total-variation (TV) prior* given by the distribution

.. math::
        p(\mathbf{x} | \theta) \propto \exp \Big( - \theta \mathrm{TV}(\mathbf{x}) \Big),

with :math:`\mathrm{TV}(\mathbf{x}) = || \nabla \mathbf{x} ||_{2, 1}` where :math:`|| \cdot ||_{2, 1}` is the mixed
:math:`L_{2, 1}` norm :py:class:`~pycsou.operator.func.norm.L21Norm` and :math:`\nabla` is the
:py:class:`~pycsou.operator.linop.diff.Gradient` operator. The posterior distribution is thus given by

.. math::
    p(\mathbf{x}|\mathbf{y},\boldsymbol{\theta}) \propto \exp \Big(-( \mathcal{F}(\mathbf{x}) + \theta \mathcal{H}
    ( \mathcal{K}(\mathbf{x})) \Big),

with:

* :math:`\mathcal{F}(\mathbf{x}) = \frac{1}{2 \sigma^2} ||\mathbf{x} -\mathbf{y}||_2^2`
* :math:`\mathcal{H}(\mathbf{x}) = || \mathbf{x} ||_{2,1}`
* :math:`\mathcal{K}(\mathbf{x}) = \nabla \mathbf{x}`.

We thus apply the :py:class:`~pycsou.opt.solver.reg_param_est.RegParamMAP` algorithm with the objective functional
:math:`\mathcal{F}(\mathbf{x}) + \theta \mathcal{H}(\mathcal{K}(\mathbf{x}))` to estimate the regularization parameter
:math:`\theta`, where the TV functional :math:`\mathrm{TV}(\mathbf{x}) = \mathcal{H}(\mathcal{K}(\mathbf{x}))` is
1-homogeneous since :math:`\mathrm{TV}(\lambda \mathbf{x}) = \lambda \mathrm{TV}(\mathbf{x})`. We plot the evolution of
the :math:`\theta` iterates throughout the algorithm to illustrate their convergence, as well as the denoised image
:math:`\mathbf{x}`.

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    import skimage as skim

    from pycsou.abc import Mode
    import pycsou.operator as pycop
    import pycsou.opt.solver as pycsol
    import pycsou.opt.stop as pycstop
    from pycsou.opt.solver.reg_param_est import RegParamMAP

    plt.rcParams["text.usetex"] = True

    gt = skim.data.shepp_logan_phantom()  # Ground-truth image
    sh_im = gt.shape
    N = np.prod(sh_im)

    # Noisy data
    rng = np.random.default_rng(seed=0)
    sigma = 1e-1
    y = gt.ravel() + sigma * rng.standard_normal(N)

    f = 1 / 2 * pycop.SquaredL2Norm(dim=y.size).asloss(y.ravel()) / sigma ** 2

    # Regularization
    K = pycop.Gradient(arg_shape=sh_im)
    K.lipschitz()
    h = pycop.L21Norm(arg_shape=(2, *sh_im))

    # MAP estimation parameters
    x0 = y
    theta0 = 1

    # Inner-loop solver (Condat-Vu) parameters
    rel_tol, max_iter, verb = 1e-4, 1e4, 1e2
    kwargs_CV = dict(f=f, g=None, h=h, K=K, verbosity=verb)
    stop_crit_CV = (pycstop.RelError(eps=rel_tol, var="x") & pycstop.RelError(eps=rel_tol, var="z")) |\
                pycstop.MaxIter(max_iter)
    fit_kwargs_CV = dict(x0=x0, stop_crit=stop_crit_CV)


    map_est = RegParamMAP(g=h * K, reg_key="h", homo_fact=1, solver=pycsol.CV, kwargs_solver=kwargs_CV)
    fit_kwargs = dict(mode=Mode.MANUAL, x0=x0, theta0=theta0, fit_kwargs_solver=fit_kwargs_CV)

    map_est.fit(**fit_kwargs)

    max_iter = int(1e1)
    theta_list = np.zeros(max_iter)
    it = 0
    for data in map_est.steps(n=max_iter):
        theta = data["theta"]
        x_opt = data["x"]
        theta_list[it] = theta
        it += 1

    theta_list = theta_list[:it]

    im_opt = x_opt.reshape(sh_im)

    fig, ax = plt.subplots()
    ax.plot(theta_list)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$\theta$")

    # Plot denoised image
    fig, ax = plt.subplots(1, 2)
    shw = ax[0].imshow(y.reshape(sh_im), vmin=0, vmax=1)
    ax[0].set_title('Noisy image')
    ax[0].axis('off')
    ax[1].imshow(im_opt, vmin=0, vmax=1)
    ax[1].set_title('Denoised image')
    ax[1].axis('off')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.subplots_adjust(right=0.8)
    fig.colorbar(shw, cax=cbar_ax)

**Example 2**

In the next example, we showcase another algorithm :py:class:`~pycsou.opt.solver.reg_param_est.RegParamMLE` that
estimates regularization parameters via maximum likelihood estimation. We consider a deconvolution problem
:math:`\mathbf{y}=\mathbf{H}\mathbf{x}_{\mathrm{GT}}+\mathbf{n}` where :math:`\mathbf{y}` is the blurry and noisy
measured image, :math:`\mathbf{H}` (forward model) is a convolution operator with a Gaussian kernel,
:math:`\mathbf{x}_{\mathrm{GT}}` is the ground-truth image, and :math:`\mathbf{n}` is additive i.i.d Gaussian noise with
variance :math:`\sigma^2`. In Bayesian frameworks, one must typically find the expression of the *posterior
distribution*, which, using Bayes’ rule, is given by

.. math::
    p(\mathbf{x}|\mathbf{y};\boldsymbol{\theta}) \propto p(\mathbf{y}|\mathbf{x}; \boldsymbol{\theta}) p(\mathbf{x};
    \boldsymbol{\theta}),

where:

* :math:`\boldsymbol{\theta} \in \mathbb{R}^K` are the model parameters to be estimated.
* :math:`p(\mathbf{y}|\mathbf{x};\boldsymbol{\theta})` is the *likelihood* of the image :math:`\mathbf{x}`, which in an
  additive Gaussian noise model is given by :math:`p(\mathbf{y}|\mathbf{x};\boldsymbol{\theta}) \propto
  \exp(- \frac{1}{2 \sigma^2}||\mathbf{H}\mathbf{x} -\mathbf{y}||_2^2)`.
* :math:`p(\mathbf{x};\boldsymbol{\theta})` is the *prior distribution*.

In this example, we assume that the noise variance :math:`\sigma^2` is unknown; it is thus considered as a model
parameter to be estimated, i.e.

.. math::
    p(\mathbf{y}|\mathbf{x};\boldsymbol{\theta}) \propto \exp \left(-\frac{\theta_0}{2}||\mathbf{H}\mathbf{x} -
    \mathbf{y}||_2^2 \right)

with :math:`\theta_0 = \frac{1}{\sigma^2}`. Next, since the ground-truth image is a deep-field image from the Hubble
space telescope that is mostly empty with a few bright galaxies, we consider an *elastic net* prior for the
reconstruction which is known to promote group sparsity, i.e.

.. math::
    p(\mathbf{x};\boldsymbol{\theta}) \propto \exp \left(-\theta_1 ||\mathbf{x}||_1 - \frac{\theta_2}{2}
    ||\mathbf{x}||_2^2 \right)

where :math:`\theta_1, \theta_2 > 0` determine the strength of the regularization. Hence, the posterior distribution
can be expressed as

.. math::
        p(\mathbf{x} | \mathbf{y}; \boldsymbol{\theta})) \propto \exp \left( - \sum_{k=0}^{2} \theta_k
        \mathcal{G}_k(\mathbf{x}) \right),

where:

* :math:`\mathcal{G}_0(\mathbf{x}) = \frac12 ||\mathbf{H}\mathbf{x} -\mathbf{y}||_2^2`
* :math:`\mathcal{G}_1(\mathbf{x}) = ||\mathbf{x}||_1`
* :math:`\mathcal{G}_2(\mathbf{x}) = \frac12 ||\mathbf{x}||_2^2`.

We thus apply the :py:class:`~pycsou.opt.solver.reg_param_est.RegParamMLE` algorithm with the objective functional
:math:`\sum_{k=0}^{2} \theta_k\mathcal{G}_k(\mathbf{x})` to estimate the parameters :math:`\boldsymbol{\theta}`. We plot
the evolution of the :math:`\boldsymbol{\theta}` iterates throughout the algorithm to illustrate their convergence.
In this simulated example, the true noise level and thus the true value :math:`\theta_0^{\mathrm{true}}` of
:math:`\theta_0` is known; we can observe that the algorithm is able to recover it accurately. We then compute the
maximum-a-posteriori (MAP) reconstruction obtained with the estimated parameters :math:`\boldsymbol{\theta}`, i.e. the
minimum of the objective functional :math:`\sum_{k=0}^{2} \theta_k\mathcal{G}_k(\mathbf{x})`. Although the theoretical
values of :math:`\theta_1` and :math:`\theta_2` are unknown, we observe that their estimates seem reasonable since the
MAP reconstructed image is visually satisfactory.

.. code-block:: python3

    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import skimage as sk

    from pycsou.abc import Mode
    import pycsou.operator as pycop
    import pycsou.opt.stop as pycstop
    import pycsou.opt.solver as pycsol
    from pycsou.opt.solver.reg_param_est import ProxFuncMoreau, RegParamMLE

    plt.rcParams["text.usetex"] = True

    # Load ground truth image
    sh_im = (256, ) * 2
    gt = sk.transform.resize(sk.color.rgb2gray(sk.data.hubble_deep_field()), sh_im)
    N = np.prod(sh_im)  # Problem dimension

    # Forward model (blurring operator)
    sigma_blur = 2
    filt = sp.ndimage._filters._gaussian_kernel1d(sigma=sigma_blur, order=0, radius=int(3*sigma_blur + 0.5))
    H = pycop.Stencil(kernel=(filt, filt),
                      center=(filt.size//2 + 1, filt.size//2 + 1),
                      arg_shape=sh_im)

    # Noisy data
    rng = np.random.default_rng(seed=0)
    sigma_gt = 1e-2
    y = H(gt.ravel()).reshape(sh_im) + sigma_gt * rng.standard_normal(sh_im)

    # Plot ground truth and noisy data
    fig, ax = plt.subplots()
    ax.imshow(gt)
    ax.set_title("Ground truth")
    ax.axis('off')

    fig, ax = plt.subplots()
    ax.imshow(y, vmin=0, vmax=1)
    ax.set_title("Measured data")
    ax.axis('off')

    # Data fidelity
    f = 1 / 2 * pycop.SquaredL2Norm(dim=N).asloss(y.ravel()) * H

    # Regularization
    g_L1 = pycop.L1Norm(dim=N)
    g_L2 = 1/2 * pycop.SquaredL2Norm(dim=N)

    # Initialize solver
    homo_factors = 2, 1, 2  # Homogeneity factors
    mu = 0.01  # Moreau envelope parameter for g_L1
    g_L1_moreau = ProxFuncMoreau(g_L1, mu)  # Differentiable approximation of g_L1

    sapg = RegParamMLE(g=[f, g_L1_moreau, g_L2], homo_factors=homo_factors)

    # Run solver
    theta0 = 1 / sigma_gt ** 2, 0.1, 1e2
    theta_min = theta0[0]/10, 1e0, theta0[2]/10
    theta_max = theta0[0]*10, 1e2, theta0[2]*10  # Valid interval for theta
    delta0 = 1e-3, 1e-3, 1e-3

    max_iter = int(3e3)
    stop_crit = pycstop.MaxIter(n=max_iter)  # Stopping criterion

    sapg.fit(mode=Mode.MANUAL, x0=np.zeros(N), theta0=theta0, theta_min=theta_min, theta_max=theta_max, delta0=delta0,
             warm_start=30, stop_crit=stop_crit, rng=rng)

    theta_list = np.zeros((len(theta0), max_iter))
    it = 0
    for data in sapg.steps():
        theta = data["theta"]
        theta_list[:, it] = theta
        it += 1

    # Plot convergence curves
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(np.log10(theta_list[0, :]))
    ax[0].axhline(y=np.log10(theta_max[0]), color='k', linestyle='--', label=r"$\log_{10}(\theta_0^{\max})$")
    ax[0].axhline(y=np.log10(1/sigma_gt**2), color='r', linestyle='--', label=r"$\log_{10}(\theta_0^{\mathrm{true}})$")
    ax[0].axhline(y=np.log10(theta_min[0]), color='b', linestyle='--', label=r"$\log_{10}(\theta_0^{\min})$")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel(r"$\log_{10}(\theta_0)$")
    ax[0].legend(loc="center right", bbox_to_anchor=(1, 0.7))

    ax[1].plot(np.log10(theta_list[1, :]))
    ax[1].axhline(y=np.log10(theta_max[1]), color='k', linestyle='--', label=r"$\log_{10}(\theta_1^{\max})$")
    ax[1].axhline(y=np.log10(theta_min[1]), color='b', linestyle='--', label=r"$\log_{10}(\theta_1^{\min})$")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel(r"$\log_{10}(\theta_1)$")
    ax[1].legend(loc="center right", bbox_to_anchor=(1, 0.7))

    ax[2].plot(np.log10(theta_list[2, :]))
    ax[2].axhline(y=np.log10(theta_max[2]), color='k', linestyle='--', label=r"$\log_{10}(\theta_2^{\max})$")
    ax[2].axhline(y=np.log10(theta_min[2]), color='b', linestyle='--', label=r"$\log_{10}(\theta_2^{\min})$")
    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel(r"$\log_{10}(\theta_2)$")
    ax[2].legend(loc="center right", bbox_to_anchor=(1, 0.7))
    fig.suptitle("Convergence plots of regularization parameters")
    fig.tight_layout()

    # Solve MAP problem with optimal theta
    pgd = pycsol.PGD(f=theta[0]*f + theta[2]*g_L2, g=theta[1] * g_L1)
    pgd.fit(x0=np.zeros(N))

    im_recon = pgd.solution().reshape(sh_im)

    # Plot MAP reconstruction
    fig, ax = plt.subplots()
    ax.imshow(im_recon, vmin=0, vmax=1)
    ax.set_title("MAP reconstruction with optimal regularization parameters")
    ax.axis('off')
    fig.tight_layout()
"""

import functools
import itertools
import operator
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou.sampler.sampler as pycs
import pycsou.util.ptype as pyct


class RegParamMAP(pyca.Solver):
    r"""
    Maximum-a-posteriori estimation algorithm that jointly recovers an estimate of the signal of interest and of the
    regularization parameter [MAP_RegParam]_.

    It can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\; \theta \mathcal{G}(\mathbf{x})}
       :label: eq:map

    while estimating a suitable regularization :math:`\theta > 0` using a Bayesian inference method (see `Notes`
    section).

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
      :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` are such that Eq. :math:numref:`eq:map` can
      be solved with a suitable external solver.

    * :math:`\mathcal{G}` is a :math:`{k}`-homogeneous functional, `i.e.`, :math:`\mathcal{G}(\lambda \mathbf{x}) =
      \lambda^{k} \mathcal{G}(\mathbf{x}) \quad \forall \mathcal{x} \in \mathbb{R}^N, \forall \eta > 0`.

    Notes
    -----
    This algorithm is a *hierarchical Bayesian method* for estimating regularization parameters. It models the
    regularization parameter :math:`\theta` as a random variable whose prior distribution is the gamma distribution

    .. math::
        p(\theta) = \frac{\beta^\alpha}{\Gamma(\alpha)} \theta^{\alpha - 1} \exp(- \beta \theta)
        \iota_{\mathbb{R}^+}(\theta)

    with parameters :math:`\alpha, \beta > 0`, and the posterior distribution conditional to :math:`\theta` is given by

    .. math::
        p(\mathbf{x}|\mathbf{y},\theta)\propto\exp\Big(-(\mathcal{F}(\mathbf{x})+\theta\mathcal{G}(\mathbf{x})\Big)

    The algorithm consists in maximizing the full posterior distribution :math:`p(\mathbf{x},\theta|\mathbf{y})` with
    two different approaches:

    * Computing the joint maximum-a-posteriori estimates

      .. math::
        \operatorname*{arg\,max}_{\mathbf{x}\in\mathbb{R}^N,\,\theta\in\mathbb{R}} \;p(\mathbf{x},\theta|\mathbf{y}),

      which corresponds to ``theta_update_method="joint"``.
    * Computing the marginalized maximum-a-posteriori estimate

      .. math::
        \operatorname*{arg\,max}_{\mathbf{x}\in\mathbb{R}^N} \int_{0}^{+ \infty} p(\mathbf{x},\theta|\mathbf{y})
        \mathrm{d} \theta,
      which corresponds to ``theta_update_method="marginalization"``.

    In both approaches, the algorithm consists in alternately optimizing with respect to :math:`\mathbf{x}` and
    :math:`\theta` via the following steps at iteration :math:`n`:

    .. math::
        &(1) \quad \mathbf{x}_n \in \operatorname*{arg\,min}_{\mathbf{x} \in \mathbb{R}^N} \;
        \mathcal{F}(\mathbf{x}) \;\;+\;\;  \theta_{n-1} \mathcal{G}(\mathbf{x}) \\
        &(2) \quad \theta_n = \begin{cases}
                              \frac{N / k + \alpha - 1}{\mathcal{G}(\mathbf{x}_n) + \beta} \quad
                              \text{for theta_update_method="joint"} \\
                              \frac{N / k + \alpha}{\mathcal{G}(\mathbf{x}_n) + \beta} \quad
                              \text{for theta_update_method="marginalization"} \\
                              \end{cases}

    Step (1) (the :math:`\mathbf{x}`-update) is performed using the inner-loop ``solver`` attribute instanciated with
    the ``kwargs_solver`` keyword arguments, and can be computationally intensive.

    Step (2) (the :math:`\theta`-update) is comparatively very cheap since it only involves a single evaluation of the
    :math:`\mathcal{G}` functional.

    **Remark 1:**

    There are no known theoretical convergence guarantees for this algorithm, only empirical evidence. Hence, the output
    may depend on the initialization parameters, in particular that of the regularization parameter :math:`\theta`.

    **Remark 2:**

    The ``reg_key`` argument specifies which functional within the ``kwargs_solver`` keyword arguments corresponds to
    the regularization term. Note that this functional does not necessarily coincide with :math:`\mathcal{G}`, which
    may be provided as a composition. For example, we may have :math:`\mathcal{G}(\mathbf{x}) = \mathcal{H}(\mathcal{K}
    (\mathbf{x}))` where :math:`\mathcal{H}` is a functional and :math:`\mathcal{K}` a linear operator within a
    :py:class:`~pycsou.opt.solver.pds.CP` solver, in which case we would have ``reg_key="h"``.

    **Remark 3:**

    In high-dimensional problems, both approaches ``theta_update_method="joint"`` and
    ``theta_update_method="marginalization"`` yield similar results since :math:`N/k \gg 1`.

    **Remark 4:**

    The :math:`\alpha` and :math:`\beta` parameters have little impact on the output of the algorithm; the default
    values should be suitable for most problems.

    **Initialization parameters of the class:**

    g: Func | None
        Regularization functional :math:`\mathcal{G}`, instance of :py:class:`~pycsou.abc.operator.Func`.
    reg_key: str
        Key corresponding to the regularization functional in the initialization parameters of ``solver``. Note that
        this functional does not necessarily coincide with :math:`\mathcal{G}`, which can be a composition
        (see `Remark 2`).
    solver: Solver
        Inner-loop solver, instance of :py:class:`~pycsou.abc.solver.Solver`
    kwargs_solver: dict
        Dictionary of keyword arguments that are passed as initialization parameters to ``solver``.

    **Parameterization of the ``fit()`` method:**

    x0: NDArray
        (..., N) initial point(s).
    theta0: Real | NDArray
        (..., 1) initial point(s) for the regularization parameter.
    fit_kwargs_solver: dict
        Dictionary of keyword arguments that are passed as initialization parameters to the ``solver.fit()`` method.
    alpha: Real
        Alpha parameter in the gamma distribution hyper-prior for the regularization parameter. Defaults to 1.
    beta: Real
        Beta parameter in the gamma distribution hyper-prior for the regularization parameter. Defaults to 1.
    theta_update_method: str
        Update method for the regularization parameter. Default to "joint".

    """

    def __init__(
        self, g: pyca.Func, reg_key: str, homo_fact: pyct.Real, solver: type[pyca.Solver], kwargs_solver: dict, **kwargs
    ):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x", "theta")),
        )
        super().__init__(**kwargs)
        self._g = g
        self._reg_key = reg_key
        self._homo_fact = homo_fact
        self._solver = solver
        self._kwargs_solver = kwargs_solver

    @pycrt.enforce_precision(i="x0")
    def m_init(
        self,
        x0: pyct.NDArray,
        theta0: typ.Union[pyct.Real, pyct.NDArray],
        fit_kwargs_solver,
        alpha: pyct.Real = 1,
        beta: pyct.Real = 1,
        theta_update_method: str = "joint",
    ):
        mst = self._mstate  # shorthand
        try:
            assert theta0 > 0 and alpha > 0 and beta > 0
            mst["theta"], mst["alpha"], mst["beta"] = theta0, alpha, beta
        except Exception:
            raise ValueError(f"theta, alpha, and beta must be positive, got {theta0}, {alpha}, and {beta}.")
        try:
            assert theta_update_method in ["joint", "marginalization"]
            self._theta_update_method = theta_update_method
        except Exception:
            raise ValueError("Unsupported theta update method.")

        mst["x"] = x0
        mst["fit_kwargs_solver"] = fit_kwargs_solver.copy()
        self._update_fit_kwargs_solver(dict(x=x0))
        mst["kwargs_solver"] = self._kwargs_solver.copy()
        self._instantiate_solver()

    def m_step(self):
        mst = self._mstate  # shorthand

        # Solve problem over x
        slv = mst["solver"]
        slv.fit(**mst["fit_kwargs_solver"])
        data, _ = slv.stats()
        mst["x"] = data["x"]
        self._update_fit_kwargs_solver(data)  # Warm start next iteration with result of current iteration
        self._update_theta()  # Update regularization parameter
        self._instantiate_solver()

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        from pycsou.opt.stop import RelError

        stop_crit_x = RelError(
            eps=1e-4,
            var="x",
        )
        stop_crit_theta = RelError(
            eps=1e-4,
            var="theta",
        )
        stop_crit = stop_crit_x | stop_crit_theta
        return stop_crit

    def objective_func(self) -> pyct.NDArray:
        return self._mstate["solver"].objective_func()

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: pyct.NDArray
            (..., N) solution.
        theta: pyct.NDArray
            (..., 1) regularisation parameter.
        """
        data, _ = self.stats()
        return data.get("x"), data.get("theta")

    def _instantiate_solver(self):
        mst = self._mstate  # shorthand
        # Update regularisation
        mst["kwargs_solver"][self._reg_key] = mst["theta"] * self._kwargs_solver.get(self._reg_key)
        mst["solver"] = self._solver(**mst["kwargs_solver"])

    def _update_fit_kwargs_solver(self, data):
        mst = self._mstate  # shorthand
        for k in data:
            # TODO does this have a risk of breaking ?
            mst["fit_kwargs_solver"][k + "0"] = data.get(k)  # Update starting point(s) of solver

    def _update_theta(self):
        mst = self._mstate  # shorthand
        x = mst["x"]
        if self._theta_update_method == "joint":
            mst["theta"] = (x.shape[-1] / self._homo_fact + mst["alpha"] - 1) / (self._g(x) + mst["beta"])
        elif self._theta_update_method == "marginalization":
            mst["theta"] = (x.shape[-1] / self._homo_fact + mst["alpha"]) / (self._g(x) + mst["beta"])


class RegParamMLE(pyca.Solver):
    r"""
    Maximum likelihood estimation (MLE) of regularization parameters.

    Estimates regularization parameters :math:`\boldsymbol{\theta} = (\theta_0, \ldots , \theta_{K-1})` for minimization
    problems of the form

    .. math::
        {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\; \sum_{k=0}^{K-1} \theta_k
        \mathcal{G}_k(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable* with :math:`\beta`-
      *Lipschitz-continuous* gradient.
    * :math:`\mathcal{G}_k:\mathbb{R}^N\rightarrow \mathbb{R}` for all :math:`0 \leq k \leq K-1` are *convex*,
      *differentiable* with :math:`\beta_k`-*Lipschitz-continuous* gradient, and :math:`\alpha_k`-*homogeneous*, i.e.
      :math:`\mathcal{G}_k(\lambda \mathbf{x}) = \lambda^{\alpha_k}\mathcal{G}_k(\mathbf{x})` for any :math:`\mathbf{x}
      \in \mathbb{R}^N` and :math:`\lambda \neq 0`.

    Notes
    -----
    This algorithm is based on the stochastic approximation proximal gradient (SAPG) algorithm described in [SAPG1]_. It
    consists in estimating the maximum of the likelihood of the regularization parameters :math:`\boldsymbol{\theta}`

    .. math::
        \mathcal{L}(\boldsymbol{\theta}) = p(\mathbf{y};\boldsymbol{\theta})= \int_{\mathbf{x}\in\mathbb{R}^N}
        p(\mathbf{y},\mathbf{x};\boldsymbol{\theta}) \mathrm{d}\mathbf{x},

    where :math:`\mathbf{y}` is the measured data and the joint distribution
    :math:`p(\mathbf{y},\mathbf{x};\boldsymbol{\theta})`, which is proportional to the *posterior distribution*
    :math:`p(\mathbf{x} | \mathbf{y} ; \boldsymbol{\theta}))` (using Bayes' rule), is determined by the objective
    functional of the original minimization problem via the relation

    .. math::
        p(\mathbf{y},\mathbf{x}; \boldsymbol{\theta})) \propto p(\mathbf{x} | \mathbf{y} ; \boldsymbol{\theta})) \propto
        \exp \left( -\mathcal{F}(\mathbf{x}) - \sum_{k=0}^{K-1} \theta_k \mathcal{G}_k(\mathbf{x}) \right).

    This algorithm iteratively updates :math:`\boldsymbol{\theta}` via projected gradient ascent on the log likelihood:

    .. math::
        \boldsymbol{\theta}_{n+1} = \mathrm{Proj}_\Theta \Big( \boldsymbol{\theta}_n + \boldsymbol{\delta}_n
        \nabla_{\boldsymbol{\theta}} \log ( \mathcal{L}(\boldsymbol{\theta})) \Big),

    where :math:`\boldsymbol{\delta}_n \in \mathbb{R}^K` are step sizes and :math:`\Theta = [\theta_0^\min,
    \theta_0^\max] \times \cdots \times [\theta_{K-1}^\min, \theta_{K-1}^\max]` is the set of feasible regularization
    parameters, determined by user-provided lower bounds :math:`\theta_k^\min` and upper bounds :math:`\theta_k^\max`
    for :math:`0 \leq k \leq K-1`.

    Since the likelihood is typically intractable in large-dimensional problems, it is estimated using Fisher's identity

    .. math::
        \frac{\mathrm{d}}{\mathrm{d} \theta_k} \log (\mathcal{L}(\boldsymbol{\theta})) =
        - \int_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}_k(\mathbf{x}) p(\mathbf{x}|\mathbf{y};\boldsymbol{\theta})
        \mathrm{d}\mathbf{x} - \frac{\mathrm{d}}{\mathrm{d} \theta_k} \log(Z_k(\theta_k)),
        :label: eq:Fisher

    where :math:`Z_k(\theta_k) = \int_{\mathbf{x}\in\mathbb{R}^N}\exp(-\theta_k\mathcal{G}_k(\mathbf{x}))\mathrm{d}
    \mathbf{x}` is the normalizing constant of the distribution :math:`p_k(\mathbf{x}; \theta_k) =
    \frac{\exp(-\theta_k\mathcal{G}_k(\mathbf{x}))}{Z_k(\theta_k)}`. The integral in Eq. :math:numref:`eq:Fisher` is
    approximated using the Monte-Carlo Markov chain (MCMC) method :py:class:`~pycsou.sampler.sampler.ULA` targeting the
    posterior distribution :math:`p(\mathbf{x}|\mathbf{y};\boldsymbol{\theta})`. The second term in
    :math:numref:`eq:Fisher` is computed exactly using the :math:`\alpha_k`-homogeneity of :math:`\mathcal{G}_k`, which
    yields :math:`\frac{\mathrm{d}}{\mathrm{d} \theta_k} \log(Z_k(\theta_k)) = - \frac{N}{\alpha_k \theta_k}`. Hence,
    the iterations of the algorithm are given by

    .. math::
        (\boldsymbol{\theta}_{n+1})_k = \mathrm{Proj}_\Theta \left( (\boldsymbol{\theta}_n)_k - (\boldsymbol{\delta}_n)_k
        \left( \frac1S \sum_{s=0}^{S-1}\mathcal{G}_k(\mathbf{x}_{n, s})-\frac{N}{\alpha_k (\boldsymbol{\theta}_n)_k}
        \right)\right),

    for :math:`0 \leq k \leq K-1`, where:

    * :math:`S > 0` is the batch size for the MCMC approximation of the integral.
    * :math:`\mathbf{x}_{n, s} \in \mathbb{R}^N` for :math:`n \in \mathbb{N}` and :math:`0 \leq s \leq S-1` are samples
      of a :py:class:`~pycsou.sampler.sampler.ULA` Markov chain targeting the posterior distribution
      :math:`p(\mathbf{x}|\mathbf{y};\boldsymbol{\theta}_n)`.

    **Remark 1:**

    The algorithm is still valid if :math:`\mathcal{F}` is zero.

    **Remark 2:**

    As opposed to purely maximum-a-posteriori formulations, in this Bayesian framework, multiplicative constants of
    the objective functional are important, since they affect the sharpness of the posterior distribution
    :math:`p(\mathbf{x} | \mathbf{y} ; \boldsymbol{\theta})) \propto \exp \left( -\mathcal{F}(\mathbf{x}) -
    \sum_{k=0}^{K-1} \theta_k \mathcal{G}_k(\mathbf{x}) \right)` that is being sampled from. When :math:`\mathcal{F}` is
    zero, this is not an issue since multiplicative constants are absorbed in the :math:`\boldsymbol{\theta}`
    parameters. However, when :math:`\mathcal{F}` is non-zero, its multiplicative constant should be selected with care.
    For example, in the case of inverse problems :math:`\mathbf{y}=\mathbf{H}\mathbf{x}+\mathbf{n}` where
    :math:`\mathbf{H}` is the forward model and :math:`\mathbf{n}` is additive i.i.d Gaussian noise with variance
    :math:`\sigma^2`, the likelihood of an image :math:`\mathbf{x}` is given by :math:`p(\mathbf{y})\propto
    \exp(- \frac{||\mathbf{H}\mathbf{x} - \mathbf{y}||_2^2}{2 \sigma^2})`, which implies that the objective functional
    should include the term :math:`\frac{1}{2 \sigma^2}||\mathbf{H}\mathbf{x} - \mathbf{y}||_2^2`. If the noise variance
    :math:`\sigma^2` is known, this term can be included in :math:`\mathcal{F}(\mathbf{x})`; however, if it is unknown,
    it should be included as :math:`\mathcal{G}_k(\mathbf{x}) = \frac{1}{2}||\mathbf{H}\mathbf{x} - \mathbf{y}||_2^2`
    for some :math:`k \geq 0`, and the noise variance will be estimated as :math:`\hat{\sigma}^2=\frac{1}
    {\hat{\theta_k}}`, where :math:`\hat{\theta_k}` is the estimated value of :math:`\theta_k` given by the algorithm
    (see top-level example of this module).

    **Remark 3:**

    This algorithm can be applied to non-differentiable proximable functionals :math:`\mathcal{G}_k` by using the
    utility class :py:class:`~pycsou.opt.solver.ProxFuncMoreau`. The gradient then calls that of the Moreau-Yosida
    envelope of the functional. This amounts to the SAPG algorithm described in [SAPG1]_. The envelope parameter
    :math:`\mu` is subject to a tradeoff between approximation bias (smaller values lead to lower bias) and convergence
    speed (smaller values lead to slower convergence) ; see [SAPG1]_ for more details.

    **Remark 4:**

    A theoretical analysis of the convergence of SAPG is given in [SAPG2]_. Note that in general, convergence is not
    guaranteed; however, there is ample empirical evidence of convergence for standard image-reconstruction problems
    (see top-level example of this module).

    **Remark 5:**

    A new :py:class:`~pycsou.sampler.sampler.ULA` chain is initialized at every iteration :math:`n` to account for the
    fact that the posterior distribution that is being targeted depends on the current iterate
    :math:`\boldsymbol{\theta}_n`. However for stability reasons, the ULA step size :math:`\gamma` is kept constant
    across iterations. By default, it is set conservatively so that the convergence guarantees of ULA are respected for
    any value of :math:`\boldsymbol{\theta}`, i.e. based on the Lipschitz constant of the gradient of :math:`\mathcal{F}
    (\mathbf{x}) + \sum_{k=0}^{K-1} \theta_{k}^\max \mathcal{G}_k(\mathbf{x})` (see [ULA]_). The first chain can be
    warm-started with the ``warm_start`` parameters of the ``fit()`` method, and/or with a starting point ``x0`` that is
    representative of the posterior distribution. All subsequent chains are warm-started with the last sample of the
    previous chain :math:`\mathbf{x}_{n-1, S-1}`.

    **Remark 6:**

    Following the recommendations of [SAPG1]_, the step sizes :math:`\boldsymbol{\delta}_n` are set to be
    :math:`\boldsymbol{\delta}_n = \boldsymbol{\delta}_0 (n + 1)^{-0.8}` for all :math:`n \in \mathbb{N}`. Although the
    choice of :math:`\boldsymbol{\delta}_0` is irrelevant in the theoretical analysis [SAPG2]_, in practice, it can
    drastically affect the convergence speed of the algorithm.

    **Initialization parameters of the class:**

    g: DiffFunc | list[DiffFunc]
        (K,) differentiable functionals :math:`[\mathcal{G}_0, \ldots, \mathcal{G}_{K-1}]`, where each
        :math:`\mathcal{G}_k` is an instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    homo_factors: Real | iterable[Real]
        (K,) homogeneity factors :math:`[\alpha_0, \ldots, \alpha_{K-1}]` corresponding to the functionals ``g``.
    f: DiffFunc | None
        Differentiable function :math:`\mathcal{F}`, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.

    **Parameterization** of the ``fit()`` method:

    x0: pyct.NDArray
        (N,) starting point of the ULA Markov chain (see `Notes`).
    theta0: Real | iterable[Real]
        (K,) starting points for the regularization parameters.
    theta_min: Real | iterable[Real]
        (K,) point-wise lower bound for the regularization parameters.
    theta_min: Real | iterable[Real]
        (K,) point-wise upper bound for the regularization parameters.
    delta0: Real | iterable[Real]
        Starting values for the gradient ascent step (see `Notes`).
    warm_start: int
        Number of warm-start iterations for the ULA Markov chain (see `Notes`). Defaults to 0.
    gamma: Real
        Discretization step of ULA (see `Notes` of :py:class:`~pycsou.sampler.sampler.ULA` documentation`).
    batch_size: int
        Batch size for Monte Carlo estimates (see `Notes`). Defaults to 1.
    log_scale: bool
        If True (default), perform the projected gradient ascent step (see `Notes`) in logarithmic scale.
    rng:
        Random number generator for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        g: typ.Union[pyca.DiffFunc, list[pyca.DiffFunc]],
        homo_factors: typ.Union[pyct.Real, typ.Iterable],
        f: pyca.DiffFunc = None,
        **kwargs,
    ):

        kwargs.update(
            log_var=kwargs.get("log_var", ("theta",)),
        )
        super().__init__(**kwargs)
        if isinstance(g, list):
            if len(g) > 1:
                assert len(g) == len(homo_factors)
            self._g = g
        else:
            self._g = [g]

        self._homo_factors = np.atleast_1d(homo_factors)
        self._f = pyco.NullFunc(dim=g[0].dim) if (f is None) else f

    def m_init(
        self,
        x0: pyct.NDArray,
        theta0: typ.Union[pyct.Real, typ.Iterable],
        theta_min: typ.Union[pyct.Real, typ.Iterable],
        theta_max: typ.Union[pyct.Real, typ.Iterable],
        delta0: pyct.Real = typ.Union[pyct.Real, typ.Iterable],
        warm_start: pyct.Integer = 0,
        gamma: pyct.Real = None,
        batch_size: pyct.Integer = 1,
        log_scale: bool = True,
        rng=None,
    ):
        mst = self._mstate  # shorthand
        mst["theta"], mst["theta_min"], mst["theta_max"] = tuple(np.atleast_1d(theta0, theta_min, theta_max))
        mst["theta"] = self._proj_interval(mst["theta"], mst["theta_min"], mst["theta_max"])

        delta0 = self._set_delta0(delta0)
        assert (
            len(mst["theta"]) == len(mst["theta_min"]) == len(mst["theta_max"]) == len(delta0) == len(self._g)
        ), "The number of hyperparameters must correspond to the number of functionals g."
        mst["delta"] = (delta0 / (k + 1) ** 0.8 for k in itertools.count(start=0))
        mst["batch_size"] = batch_size
        mst["log_scale"] = log_scale

        # Set gamma with most conservative Lipschitz constant
        self._update_moreau()
        gamma = self._set_gamma(gamma, dl=self._diff_lipschitz(mst["theta_max"]))
        mc = pycs.ULA(f=self._MAP_objective_func(), gamma=gamma)
        mst["gamma"] = mc._gamma
        mc_gen = mc.samples(x0=x0, rng=rng)
        for _ in range(warm_start):
            x0 = next(mc_gen)  # Warm-start Markov chain
        mst["x"] = x0
        mst["mc_gen"], mst["rng"] = mc_gen, mc._rng

    def m_step(self):
        mst = self._mstate  # shorthand
        delta = next(mst["delta"])
        # Compute MC expectation of g wrt to posterior distribution
        means = np.zeros_like(mst["theta"])
        for _ in range(mst["batch_size"]):
            x = next(mst["mc_gen"])
            for i in range(len(self._g)):
                means[i] += self._g[i](x)
        means /= mst["batch_size"]
        mst["x"] = x

        # Update theta
        grad = self._f.dim / (self._homo_factors * mst["theta"]) - means
        if mst["log_scale"]:
            eta = np.log(mst["theta"])
            eta += mst["theta"] * delta * grad
            mst["theta"] = np.exp(eta)
        else:
            mst["theta"] += delta * grad

        mst["theta"] = self._proj_interval(mst["theta"], mst["theta_min"], mst["theta_max"])

        # Update MC kernel with new theta iterate
        self._update_moreau()
        mc = pycs.ULA(f=self._MAP_objective_func(), gamma=mst["gamma"])
        mst["mc_gen"] = mc.samples(x0=mst["x"], rng=mst["rng"])

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        from pycsou.opt.stop import RelError

        stop_crit = RelError(
            eps=1e-4,
            var="theta",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def solution(self):
        return self._mstate["theta"]

    @staticmethod
    def _proj_interval(x, x_min, x_max):
        return np.maximum(np.minimum(x, x_max), x_min)

    def _set_delta0(self, delta: pyct.Real = None) -> pyct.Real:
        if delta is None:
            return 1 / (self._mstate["theta"] * self._f.dim)
        else:
            return np.array(delta, dtype=float)

    def _set_gamma(self, gamma: pyct.Real = None, dl: pyct.Real = 0) -> pyct.Real:
        if gamma is None:
            return pycrt.coerce(0.98 / dl)
        else:
            return pycrt.coerce(gamma)

    def _diff_lipschitz(self, theta):
        return self._f.diff_lipschitz() + sum([theta[i] * self._g[i].diff_lipschitz() for i in range(len(self._g))])

    def _MAP_objective_func(self):
        to_sum = [self._mstate["theta"][i] * self._g[i] for i in range(len(self._g))]
        return self._f + functools.reduce(operator.add, to_sum)

    def _update_moreau(self):
        for i in range(len(self._g)):
            if isinstance(self._g[i], ProxFuncMoreau):
                self._g[i].set_mu(self._g[i]._mu0 * self._mstate["theta"][i])


class ProxFuncMoreau(pyca.ProxDiffFunc):
    r"""
    Proximable function with Moreau-Yosida envelope approximation for the gradient.

    Utility class to make a proximable functional differentiable, by approximating its gradient with that of its
    Moreau-Yosida envelope. The ``apply()`` and ``prox()`` methods call that of the original functional. This class can
    be useful within solvers that require differentiable functionals, notably :py:class:`~pycsou.opt.solver.RegParamMLE`.
    """

    def __init__(self, f: pyca.ProxFunc, mu: pyct.Real):
        r"""
        Parameters
        ----------
        f: int
            Dimension size. (Default: domain-agnostic.)
        mu: Real
            Moreau envelope parameter.

        """
        super().__init__(shape=(1, f.dim))
        self._f = f
        self._mu0 = mu  # Specifically for RegParamMLE
        self._mu = mu
        self._moreau_envelope = None
        self.set_mu(mu)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._f.apply(arr)

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._moreau_envelope.grad(arr)

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return self._f.prox(arr, tau)

    def set_mu(self, mu: pyct.Real):
        self._mu = mu
        self._moreau_envelope = self._f.moreau_envelope(mu)
        self._diff_lipschitz = self._moreau_envelope.diff_lipschitz()
