r"""
This sampler module implements state-of-the-art algorithms that generate samples from probability distributions.  These
algorithms are particularly well-suited for high-dimensional distributions such as posterior distributions of inverse
problems in imaging, which is a notoriously difficult task.  The ability to sample from the posterior distribution is
extremely valuable, as it allows to explore the landscape of the objective function instead of having a single point
estimate (the maximum a posteriori solution, i.e. the mode of the posterior).  This is useful for uncertainty
quantification (UQ) purposes [UQ_MCMC]_, and it allows compute Monte Carlo estimates of expected values with respect to
the posterior.  For example, the mean of samples from the posterior is an approximation of the minimum mean-square error
(MMSE) estimator that can be used for image reconstruction.  Higher-order pixel-wise statistics (e.g., the variance) can
also be computed in an online fashion (see :py:mod:`~pyxu.experimental.sampler.statistics`) and provide useful
diagnostic tools for uncertainty quantification.

In the following example, we showcase the unajusted Langevin algorithm (:py:class:`~pyxu.experimental.sampler.ULA`)
applied to a total-variation denoising problem.  We show the MMSE estimator as well as the pixelwise variance of the
samples.  As expected, the variance is higher around edges than in the smooth regions, indicating that there is higher
uncertainty in these regions.

.. code-block:: python3

   import matplotlib.pyplot as plt
   import numpy as np
   import pyxu.experimental.sampler as pxe_sampler
   import pyxu.operator as pxo
   import pyxu.opt.solver as pxsl
   import skimage as skim

   sh_im = (128,) * 2
   gt = skim.transform.resize(skim.data.shepp_logan_phantom(), sh_im)  # Ground-truth image
   N = np.prod(sh_im)  # Number of pixels

   # Noisy data
   rng = np.random.default_rng(seed=0)
   sigma = 1e-1  # Noise standard deviation
   y = gt + sigma * rng.standard_normal(sh_im)  # Noisy image
   f = 1 / 2 * pxo.SquaredL2Norm(dim_shape=sh_im).argshift(-y) / sigma**2  # Data fidelity loss

   # Smoothed TV regularization
   g = pxo.L21Norm(dim_shape=(2, *sh_im)).moreau_envelope(1e-2) * pxo.Gradient(dim_shape=sh_im)
   theta = 10  # Regularization parameter

   # Compute MAP estimator
   pgd = pxsl.PGD(f=f + theta * g)
   pgd.fit(x0=y)
   im_MAP = pgd.solution()

   fig, ax = plt.subplots(1, 3)
   ax[0].imshow(gt)
   ax[0].set_title("Ground truth")
   ax[0].axis("off")
   ax[1].imshow(y, vmin=0, vmax=1)
   ax[1].set_title("Noisy image")
   ax[1].axis("off")
   ax[2].imshow(im_MAP, vmin=0, vmax=1)
   ax[2].set_title("MAP reconstruction")
   ax[2].axis("off")

   ula = pxe_sampler.ULA(f=f + theta * g)  # ULA sampler

   n = int(1e4)  # Number of samples
   burn_in = int(1e3)  # Number of burn-in iterations
   gen = ula.samples(x0=np.zeros(N), rng=rng)  # Generator for ULA samples
   # Objects for computing online statistics based on samples
   online_mean = pxe_sampler.OnlineMoment(order=1)
   online_var = pxe_sampler.OnlineVariance()

   i = 0  # Number of samples
   for sample in gen:  # Draw ULA sample
       i += 1
       if i > burn_in + n:
           break
       if i > burn_in:
           mean = online_mean.update(sample)  # Update online mean
           var = online_var.update(sample)  # Update online variance

   fig, ax = plt.subplots(1, 2)
   mean_im = ax[0].imshow(mean, vmin=0, vmax=1)
   fig.colorbar(mean_im, fraction=0.05, ax=ax[0])
   ax[0].set_title("Mean (MMSE estimator)")
   ax[0].axis("off")
   var_im = ax[1].imshow(var)
   fig.colorbar(var_im, fraction=0.05, ax=ax[1])
   ax[1].set_title("Variance")
   ax[1].axis("off")
   fig.suptitle("Pixel-wise statistics of ULA samples")
"""

import collections.abc as cabc
import math

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.util as pxu

__all__ = [
    "_Sampler",
    "ULA",
    "MYULA",
]


class _Sampler:
    """Abstract base class for samplers."""

    def samples(self, rng=None, **kwargs) -> cabc.Generator:
        """Returns a generator; samples are drawn by calling next(generator)."""
        self._sample_init(rng, **kwargs)

        def _generator():
            while True:
                yield self._sample()

        return _generator()

    def _sample_init(self, rng, **kwargs):
        """Optional method to set initial state of the sampler (e.g., a starting point)."""
        pass

    def _sample(self) -> pxt.NDArray:
        """Method to be implemented by subclasses that returns the next sample."""
        raise NotImplementedError


class ULA(_Sampler):
    r"""
    Unajusted Langevin algorithm (ULA).

    Generates samples from the distribution

    .. math::

       p(\mathbf{x})
       =
       \frac{\exp(-\mathcal{F}(\mathbf{x}))}{\int_{\mathbb{R}^N} \exp(-\mathcal{F}(\tilde{\mathbf{x}}))
       \mathrm{d} \tilde{\mathbf{x}} },

    where :math:`\mathcal{F}: \mathbb{R}^N \to \mathbb{R}` is *differentiable* with :math:`\beta`-*Lipschitz continuous*
    gradient.

    Notes
    -----
    ULA is a Monte-Carlo Markov chain (MCMC) method that derives from the discretization of overdamped Langevin
    diffusions.  More specifically, it relies on the Langevin stochastic differential equation (SDE):

    .. math::

       \mathrm{d} \mathbf{X}_t
       =
       - \nabla \mathcal{F}(\mathbf{X}_t) \mathrm{d}t + \sqrt{2} \mathrm{d} \mathbf{B}_t,

    where :math:`(\mathbf{B}_t)_{t \geq 0}` is a :math:`N`-dimensional Brownian motion.  It is well known that under
    mild technical assumptions, this SDE has a unique strong solution whose invariant distribution is
    :math:`p(\mathbf{x}) \propto \exp(-\mathcal{F}(\mathbf{x}))`.  The discrete-time Euler-Maruyama discretization of
    this SDE then yields the ULA Markov chain

    .. math::

       \mathbf{X}_{k+1} = \mathbf{X}_{k} - \gamma \nabla \mathcal{F}(\mathbf{X}_k) + \sqrt{2 \gamma} \mathbf{Z}_{k+1}

    for all :math:`k \in \mathbb{Z}`, where :math:`\gamma` is the discretization step size and :math:`(\mathbf{Z}_k)_{k
    \in \mathbb{Z}}` is a sequence of independant and identically distributed :math:`N`-dimensional standard Gaussian
    distributions.  When :math:`\mathcal{F}` is differentiable with :math:`\beta`-Lipschitz continuous gradient and
    :math:`\gamma \leq \frac{1}{\beta}`, the ULA Markov chain converges (see [ULA]_) to a unique stationary distribution
    :math:`p_\gamma` such that

    .. math::

       \lim_{\gamma \to 0} \Vert p_\gamma - p \Vert_{\mathrm{TV}} = 0.

    The discretization step :math:`\gamma` is subject to the bias-variance tradeoff: a larger step will lead to faster
    convergence of the Markov chain at the expense of a larger bias in the approximation of the distribution :math:`p`.
    Setting :math:`\gamma` as large as possible (default behavior) is recommended for large-scale problems, since
    convergence speed (rather than approximation bias) is then typically the main bottelneck.  See `Example` section
    below for a concrete illustration of this tradeoff.

    Remarks
    -------
    Like all MCMC sampling methods, ULA comes with the following challenges:

    * The first few samples of the chain may not be adequate for computing statistics, as they might be located in low
      probability regions.  This challenge can either be alleviated by selecting a representative starting point to the
      chain, or by having a `burn-in` phase where the first few samples are discarded.

    * Consecutive samples are typically correlated, which can deteriorate the Monte-Carlo estimation of quantities of
      interest.  This issue can be alleviated by `thinning` the chain, i.e., selecting only every :math:`k` samples, at
      the expense of an increased computational cost.  Useful diagnostic tools to quantify this correlation between
      samples include the pixel-wise autocorrelation function and the `effective sample size
      <https://mc-stan.org/docs/reference-manual/effective-sample-size.html>`_.

    Example
    -------
    We illustrate ULA on a 1D example (:math:`N = 1`) where :math:`\mathcal{F}(x) = \frac{x^2}{2}`; the target
    distribution :math:`p(x)` is thus the 1D standard Gaussian.  In this toy example, the biased distribution
    :math:`p_\gamma(x)` can be computed in closed form.  The ULA Markov chain is given by

    .. math::

       \mathbf{X}_{k+1} &= \mathbf{X}_{k} - \gamma \nabla\mathcal{F}(\mathbf{X}_k) + \sqrt{2\gamma}\mathbf{Z}_{k+1} \\
       &= \mathbf{X}_{k} (1 - \gamma) + \sqrt{2 \gamma} \mathbf{Z}_{k+1}.

    Assuming for simplicity that :math:`\mathbf{X}_0` is Gaussian with mean :math:`\mu_0` and variance
    :math:`\sigma_0^2`, :math:`\mathbf{X}_k` is Gaussian for any :math:`k \in \mathbb{Z}` as a linear combination of
    Gaussians.  Taking the expected value of the recurrence relation yields

    .. math::

       \mu_k := \mathbb{E}(\mathbf{X}_{k}) = \mathbb{E}(\mathbf{X}_{k-1}) (1 - \gamma) = \mu_0 (1 - \gamma)^k

    (geometric sequence).  Taking the expected value of the square of the recurrence relation yields

    .. math::

       \mu^{(2)}_k := \mathbb{E}(\mathbf{X}_{k}^2) = \mathbb{E}(\mathbf{X}_{k-1}^2) (1 - \gamma)^2 + 2 \gamma =
       (1 - \gamma)^{2k} (\sigma_0^2 - b) + b

    with :math:`b = \frac{2 \gamma}{1 - (1 - \gamma)^{2}} = \frac{1}{1-\frac{\gamma}{2}}` (arithmetico-geometric
    sequence) due to the independence of :math:`\mathbf{X}_{k-1}` and :math:`\mathbf{Z}_{k}`.  Hence,
    :math:`p_\gamma(x)` is a Gaussian with mean :math:`\mu_\gamma= \lim_{k \to \infty} \mu_k = 0` and variance
    :math:`\sigma_\gamma^2 = \lim_{k \to \infty} \mu^{(2)}_k - \mu_k^2 = \frac{1}{1-\frac{\gamma}{2}}`.  As expected, we
    have :math:`\lim_{\gamma \to 0} \sigma_\gamma^2 = 1`, which is the variance of the target distribution :math:`p(x)`.

    We plot the distribution of the samples of ULA for one large (:math:`\gamma_1 \approx 1`, i.e.
    :math:`\sigma_{\gamma_1}^2 \approx 2`) and one small (:math:`\gamma_2 = 0.1`, i.e. :math:`\sigma_{\gamma_2}^2
    \approx 1.05`) step size.  As expected, the larger step size :math:`\gamma_1` leads to a larger bias in the
    approximation of :math:`p(x)`.  To quantify the speed of convergence of the Markov chains, we compute the
    `Cramér-von Mises <https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion>`_ tests of goodness of fit
    of the empirical distributions to the stationary distributions of ULA :math:`p_{\gamma_1}(x)` and
    :math:`p_{\gamma_2}(x)`.  We observe that the larger step :math:`\gamma_1` leads to a better fit (lower Cramér-von
    Mises criterion), which illustrates the aforementioned bias-variance tradeoff for the choice of the step size.

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import pyxu.experimental.sampler as pxe_sampler
       import pyxu.operator as pxo
       import scipy as sp

       f = pxo.SquaredL2Norm(dim_shape=1) / 2  # To sample 1D normal distribution (mean 0, variance 1)
       ula = pxe_sampler.ULA(f=f)  # Sampler with maximum step size
       ula_lb = pxe_sampler.ULA(f=f, gamma=1e-1)  # Sampler with small step size

       gen_ula = ula.samples(x0=np.zeros(1))
       gen_ula_lb = ula_lb.samples(x0=np.zeros(1))
       n_burn_in = int(1e3)  # Number of burn-in iterations
       for i in range(n_burn_in):
           next(gen_ula)
           next(gen_ula_lb)

       # Online statistics objects
       mean_ula = pxe_sampler.OnlineMoment(order=1)
       mean_ula_lb = pxe_sampler.OnlineMoment(order=1)
       var_ula = pxe_sampler.OnlineVariance()
       var_ula_lb = pxe_sampler.OnlineVariance()

       n = int(1e4)  # Number of samples
       samples_ula = np.zeros(n)
       samples_ula_lb = np.zeros(n)
       for i in range(n):
           sample = next(gen_ula)
           sample_lb = next(gen_ula_lb)
           samples_ula[i] = sample
           samples_ula_lb[i] = sample_lb
           mean = float(mean_ula.update(sample))
           var = float(var_ula.update(sample))
           mean_lb = float(mean_ula_lb.update(sample_lb))
           var_lb = float(var_ula_lb.update(sample_lb))

       # Theoretical variances of biased stationary distributions of ULA
       biased_var = 1 / (1 - ula._gamma / 2)
       biased_var_lb = 1 / (1 - ula_lb._gamma / 2)

       # Quantify goodness of fit of empirical distribution with theoretical distribution (Cramér-von Mises test)
       cvm = sp.stats.cramervonmises(samples_ula, "norm", args=(0, np.sqrt(biased_var)))
       cvm_lb = sp.stats.cramervonmises(samples_ula_lb, "norm", args=(0, np.sqrt(biased_var_lb)))

       # Plots
       grid = np.linspace(-4, 4, 1000)

       plt.figure()
       plt.title(
           f"ULA samples (large step size) \n Empirical mean: {mean:.3f} (theoretical: 0) \n "
           f"Empirical variance: {var:.3f} (theoretical: {biased_var:.3f}) \n"
           f"Cramér-von Mises goodness of fit: {cvm.statistic:.3f}"
       )
       plt.hist(samples_ula, range=(min(grid), max(grid)), bins=100, density=True)
       plt.plot(grid, sp.stats.norm.pdf(grid), label=r"$p(x)$")
       plt.plot(grid, sp.stats.norm.pdf(grid, scale=np.sqrt(biased_var)), label=r"$p_{\gamma_1}(x)$")
       plt.legend()
       plt.show()

       plt.figure()
       plt.title(
           f"ULA samples (small step size) \n Empirical mean: {mean_lb:.3f} (theoretical: 0) \n "
           f"Empirical variance: {var_lb:.3f} (theoretical: {biased_var_lb:.3f}) \n"
           f"Cramér-von Mises goodness of fit: {cvm_lb.statistic:.3f}"
       )
       plt.hist(samples_ula_lb, range=(min(grid), max(grid)), bins=100, density=True)
       plt.plot(grid, sp.stats.norm.pdf(grid), label=r"$p(x)$")
       plt.plot(grid, sp.stats.norm.pdf(grid, scale=np.sqrt(biased_var_lb)), label=r"$p_{\gamma_2}(x)$")
       plt.legend()
       plt.show()
    """

    def __init__(self, f: pxa.DiffFunc, gamma: pxt.Real = None):
        r"""
        Parameters
        ----------
        f: :py:class:`~pyxu.abc.DiffFunc`
            Differentiable functional.
        gamma: Real
            Euler-Maruyama discretization step of the Langevin equation (see `Notes`).
        """
        self._f = f
        self._beta = f.diff_lipschitz
        self._gamma = self._set_gamma(gamma)
        self._rng = None
        self.x = None

    def _sample_init(self, rng, x0: pxt.NDArray):
        r"""
        Parameters
        ----------
        rng:
            Internal random generator.
        x0: NDArray
            Starting point of the Markov chain.
        """
        self.x = x0.copy()
        if rng is None:
            xp = pxu.get_array_module(x0)
            self._rng = xp.random.default_rng(None)
        else:
            self._rng = rng

    def _sample(self) -> pxt.NDArray:
        x = self.x.copy()
        x -= self._gamma * pxu.copy_if_unsafe(self._f.grad(self.x))
        x += math.sqrt(2 * self._gamma) * self._rng.standard_normal(size=self.x.shape, dtype=self.x.dtype)
        self.x = x
        return x

    def objective_func(self) -> pxt.Real:
        r"""
        Negative logarithm of the target ditribution (up to the a constant) evaluated at the current state of the Markov
        chain.

        Useful for diagnostics purposes to monitor whether the Markov chain is sufficiently warm-started.  If so, the
        samples should accumulate around the modes of the target distribution, i.e., toward the minimum of
        :math:`\mathcal{F}`.
        """
        return pxu.copy_if_unsafe(self._f.apply(self.x))

    def _set_gamma(self, gamma: pxt.Real = None) -> pxt.Real:
        if gamma is None:
            if math.isfinite(self._beta):
                return 0.98 / self._beta
            else:
                msg = "If f has unbounded Lipschitz gradient, the gamma parameter must be provided."
            raise ValueError(msg)
        else:
            try:
                assert gamma > 0
            except Exception:
                raise ValueError(f"gamma must be positive, got {gamma}.")
            return gamma


class MYULA(ULA):
    r"""
    Moreau-Yosida unajusted Langevin algorithm (MYULA).

    Generates samples from the distribution

    .. math::

       p(\mathbf{x}) = \frac{\exp(-\mathcal{F}(\mathbf{x}) - \mathcal{G}(\mathbf{x}))}{\int_{\mathbb{R}^N}
       \exp(-\mathcal{F}(\tilde{\mathbf{x}}) - \mathcal{G}(\tilde{\mathbf{x}})) \mathrm{d} \tilde{\mathbf{x}} },

    where :math:`\mathcal{F}: \mathbb{R}^N \to \mathbb{R}` is *convex* and *differentiable* with :math:`\beta`-
    *Lipschitz continuous* gradient, and :math:`\mathcal{G}: \mathbb{R}^N \to \mathbb{R}` is *proper*, *lower semi-
    continuous* and *convex* with *simple proximal operator*.

    Notes
    -----
    MYULA is an extension of :py:class:`~pyxu.experimental.sampler.ULA` to sample from distributions whose logarithm is
    nonsmooth.  It consists in applying ULA to the differentiable functional :math:`\mathcal{U}^\lambda = \mathcal{F} +
    \mathcal{G}^\lambda` for some :math:`\lambda > 0`, where

     .. math::

        \mathcal{G}^\lambda (\mathbf{x}) = \inf_{\tilde{\mathbf{x}} \in \mathbb{R}^N} \frac{1}{2 \lambda} \Vert
        \tilde{\mathbf{x}} - \mathbf{x} \Vert_2^2 + \mathcal{G}(\tilde{\mathbf{x}})

    is the Moreau-Yosida envelope of :math:`\mathcal{G}` with parameter :math:`\lambda`.  We then have

    .. math::

       \nabla \mathcal{U}^\lambda (\mathbf{x}) = \nabla \mathcal{F}(\mathbf{x}) + \frac{1}{\lambda} (\mathbf{x} -
       \mathrm{prox}_{\lambda \mathcal{G}}(\mathbf{x})),

    hence :math:`\nabla \mathcal{U}^\lambda` is :math:`(\beta + \frac{1}{\lambda})`-Lipschitz continuous, where
    :math:`\beta` is the Lipschitz constant of :math:`\nabla \mathcal{F}`.  Note that the target distribution of the
    underlying ULA Markov chain is not exactly :math:`p(\mathbf{x})`, but the distribution

    .. math::

       p^\lambda(\mathbf{x}) \propto \exp(-\mathcal{F}(\mathbf{x})-\mathcal{G}^\lambda(\mathbf{x})),

    which introduces some additional bias on top of the bias of ULA related to the step size :math:`\gamma` (see `Notes`
    of :py:class:`~pyxu.experimental.sampler.ULA` documentation).  MYULA is guaranteed to converges when :math:`\gamma
    \leq \frac{1}{\beta + \frac{1}{\lambda}}`, in which case it converges toward the stationary distribution
    :math:`p^\lambda_\gamma(\mathbf{x})` that satisfies

    .. math::

       \lim_{\gamma, \lambda \to 0} \Vert p^\lambda_\gamma - p \Vert_{\mathrm{TV}} = 0

    (see [MYULA]_).  The parameter :math:`\lambda` parameter is subject to a similar bias-variance tradeoff as
    :math:`\gamma`.  It is recommended to set it in the order of :math:`\frac{1}{\beta}`, so that the contributions of
    :math:`\mathcal{F}` and :math:`\mathcal{G}^\lambda` to the Lipschitz constant of :math:`\nabla \mathcal{U}^\lambda`
    is well balanced.
    """

    def __init__(
        self,
        f: pxa.DiffFunc = None,
        g: pxa.ProxFunc = None,
        gamma: pxt.Real = None,
        lamb: pxt.Real = None,
    ):
        r"""
        Parameters
        ----------
        f: :py:class:`~pyxu.abc.DiffFunc`, None
            Differentiable functional.
        g: :py:class:`~pyxu.abc.ProxFunc`, None
            Proximable functional.
        gamma: Real
            Euler-Maruyama discretization step of the Langevin equation (see `Notes` of
            :py:class:`~pyxu.experimental.sampler.ULA` documentation).
        lamb: Real
            Moreau-Yosida envelope parameter for `g`.
        """
        dim_shape = None
        if f is not None:
            dim_shape = f.dim_shape
        if g is not None:
            if dim_shape is None:
                dim_shape = g.dim_shape
            else:
                assert g.dim_shape == dim_shape
        if dim_shape is None:
            raise ValueError("One of f or g must be nonzero.")

        self._f_diff = pxo.NullFunc(dim_shape=dim_shape) if (f is None) else f
        self._g = pxo.NullFunc(dim_shape=dim_shape) if (g is None) else g

        self._lambda = self._set_lambda(lamb)
        f = self._f_diff + self._g.moreau_envelope(self._lambda)
        f.diff_lipschitz = f.estimate_diff_lipschitz()
        super().__init__(f, gamma)

    def _set_lambda(self, lamb: pxt.Real = None) -> pxt.Real:
        if lamb is None:
            if self._g._name == "NullFunc":
                return 1.0  # Lambda is irrelevant if g is a NullFunc, but it must be positive
            elif math.isfinite(dl := self._f_diff.diff_lipschitz):
                return 2.0 if dl == 0 else min(2.0, 1.0 / dl)
            else:
                msg = "If f has unbounded Lipschitz gradient, the lambda parameter must be provided."
            raise ValueError(msg)
        else:
            return lamb
