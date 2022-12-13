import math
import types
import typing as typ
import warnings

import pycsou.abc as pyca
import pycsou.abc.operator as pyco
import pycsou.operator.func as pycf
import pycsou.operator.linop as pyclo
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    *("CondatVu", "CV"),
    "PD3O",
    *("ChambollePock", "CP"),
    *("LorisVerhoeven", "LV"),
    *("DavisYin", "DY"),
    *("DouglasRachford", "DR"),
    "ADMM",
    *("ForwardBackward", "FB"),
    *("ProximalPoint", "PP"),
]


class _PrimalDualSplitting(pyca.Solver):
    r"""
    Base class for Primal Dual Splitting (PDS) solvers.
    """

    def __init__(
        self,
        f: typ.Optional[pyca.DiffFunc] = None,
        g: typ.Optional[pyca.ProxFunc] = None,
        h: typ.Optional[pyca.ProxFunc] = None,
        K: typ.Optional[pyca.DiffMap] = None,
        beta: typ.Optional[pyct.Real] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
        super().__init__(**kwargs)
        if (f is None) and (g is None) and (h is None):
            msg = " ".join(
                [
                    "Cannot minimize always-0 functional.",
                    "At least one of Parameter[f, g, h] must be specified.",
                ]
            )
            raise ValueError(msg)

        if f is not None:
            primal_dim = f.dim
        elif g is not None:
            primal_dim = g.dim
        else:
            primal_dim = h.dim

        if h is not None:
            dual_dim = h.dim
        elif K is not None:
            dual_dim = K.shape[0]
        else:
            dual_dim = primal_dim

        self._f = pyclo.NullFunc(dim=primal_dim) if (f is None) else f
        self._g = pyclo.NullFunc(dim=primal_dim) if (g is None) else g
        self._h = pyclo.NullFunc(dim=dual_dim) if (h is None) else h
        self._beta = self._set_beta(beta)
        if h is not None:
            self._K = pyclo.IdentityOp(dim=h.dim) if (K is None) else K
        else:
            if K is None:
                K_dim = f.dim if f is not None else g.dim
                self._K = pyclo.NullOp(shape=(K_dim, K_dim))
            else:
                raise ValueError("Optional argument ``h`` mut be specified if ``K`` is not None.")
        self._objective_func = self._f + self._g + (self._h * self._K)

    @pycrt.enforce_precision(i=["x0", "z0", "tau", "sigma", "rho"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray] = None,
        tau: typ.Optional[pyct.Real] = None,
        sigma: typ.Optional[pyct.Real] = None,
        rho: typ.Optional[pyct.Real] = None,
        tuning_strategy: typ.Literal[1, 2, 3] = 1,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = x0 if x0.ndim > 1 else x0.reshape(1, -1)
        mst["z"] = self._set_dual_variable(z0)
        self._tuning_strategy = tuning_strategy
        gamma = self._set_gamma(tuning_strategy)
        mst["tau"], mst["sigma"], delta = self._set_step_sizes(tau, sigma, gamma)
        mst["rho"] = self._set_momentum_term(rho, delta)

    def m_step(self):
        raise NotImplementedError

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        stop_crit_x = pycos.RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        stop_crit_z = pycos.RelError(
            eps=1e-4,
            var="z",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit_x & stop_crit_z if self._h._name != "NullFunc" else stop_crit_x

    def solution(self, which: typ.Literal["primal", "dual"] = "primal") -> pyct.NDArray:
        data, _ = self.stats()
        if which == "primal":
            assert "x" in data.keys(), "Primal variable x was not logged (declare it in log_var to log it)."
        elif which == "dual":
            assert "z" in data.keys(), "Dual variable z was not logged (declare it in log_var to log it)."
        else:
            raise ValueError(f"Parameter which must be one of ['primal', 'dual'] got: {which}.")
        return data.get("x") if which == "primal" else data.get("z")

    def objective_func(self) -> pyct.NDArray:
        return self._objective_func(self._mstate["x"])

    @pycrt.enforce_precision(i=["beta"], allow_None=True)
    def _set_beta(self, beta: typ.Optional[pyct.Real]) -> pyct.Real:
        r"""
        Sets the Lipschitz constant.

        Returns
        -------
        float
            Lipschitz constant.
        """
        if beta is None:
            if math.isfinite(dl := self._f._diff_lipschitz):
                return pycrt.coerce(dl)
            else:
                msg = "beta: automatic inference not supported for operators with unbounded Lipschitz gradients."
            raise ValueError(msg)
        else:
            return beta

    def _set_dual_variable(self, z: typ.Optional[pyct.NDArray]) -> pyct.NDArray:
        r"""
        Initialize the dual variable if it is ```None``` by mapping the primal variable through the operator K.

        Returns
        -------
        NDArray
            Initialized dual variable.
        """
        if z is None:
            return self._K(self._mstate["x"].copy())
        else:
            return z if z.ndim > 1 else z.reshape(1, -1)

    def _set_gamma(self, tuning_strategy: typ.Literal[1, 2, 3]) -> pyct.Real:
        r"""
        Sets the gamma parameter according to the tuning strategy.

        Returns
        -------
        float
            Gamma parameter.
        """
        return pycrt.coerce(self._beta) if tuning_strategy != 2 else pycrt.coerce(self._beta / 2)

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: pyct.Real
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        raise NotImplementedError

    def _set_momentum_term(self, rho: typ.Optional[pyct.Real], delta: pyct.Real) -> pyct.Real:
        r"""
        Sets the momentum term according to Theorem 8.2 in [PSA]_.

        Returns
        -------
        float
            Momentum term.

        Notes
        -----
        The :math:`O(1/\sqrt(k))` objective functional convergence rate of (Theorem 1 of [dPSA]_) is  for `\rho=1`.
        """
        if rho is None:
            rho = 1.0 if self._tuning_strategy != 3 else delta - 0.1
        else:
            assert rho <= delta, f"Parameter rho must be smaller than delta: {rho} > {delta}."
        return pycrt.coerce(rho)


_PDS = _PrimalDualSplitting


class CondatVu(_PrimalDualSplitting):
    r"""
    Condat-Vu (CV) primal-dual splitting algorithm.

    This class is also accessible via the alias ``CV()``.

    The *Condat Vu (CV)* primal-dual method is described in [CVS]_ (this particular implementation is based on the pseudo-code Algorithm 7.1 provided in [FuncSphere]_ Chapter 7, Section1).

    It can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathcal{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator* :math:`\mathbf{K}`), with **operator norm**:

    .. math::

       \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    The algorithm has convergence guarantees for the case in which :math:`\mathcal{H}` is composed with a
    *linear operator* :math:`\mathbf{K}`. When :math:`\mathcal{F}=0`, convergence can be proven for *non-linear differentiable maps* :math:`\mathcal{K}` (see [NLCP]_).
    Note that this class does not support yet automatic selection of hyperparameters for the case of *non-linear differentiable maps* :math:`\mathcal{K}`.

    **Remark 3:**

    Assume that the following holds:

    * :math:`\beta>0` and:

      - :math:`\gamma \geq \frac{\beta}{2}`,
      - :math:`\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\geq \gamma`,
      - :math:`\rho \in ]0,\delta[`, where :math:`\delta:=2-\frac{\beta}{2}\gamma^{-1}\in[1,2[` (:math:`\delta=2` is possible when :math:`\mathcal{F}` is *quadratic*
        and :math:`\gamma \geq \beta`, see [PSA]_).

    * or :math:`\beta=0` and:

      - :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2\leq 1`
      - :math:`\rho \in ]0,2[`.

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M` solution s.t. the primal and dual sequences
    of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively, i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.


    **Initialization parameters of the class:**

    f: DiffFunc | None
        Differentiable function :math:`\mathcal{F}`, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    g: ProxFunc | None
        Proximable function :math:`\mathcal{G}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    h: ProxFunc | None
        Proximable function :math:`\mathcal{H}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`, composed with a differentiable map
        :math:`\mathcal{K}`.
    K: DiffMap | None
        Differentiable map :math:`\mathcal{K}` instance of :py:class:`~pycsou.abc.operator.DiffMap`, or a linear
        operator :math:`\mathbf{K}` instance of :py:class:`~pycsou.abc.operator.LinOp`.
    beta: float | None
        Lipschitz constant :math:`\beta` of the gradient of :math:`\mathcal{F}`. If not provided, it will be automatically estimated.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    sigma: Real | None
        Dual step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See section below for more details.

    **Default values of the hyperparameters.**

    This class supports three strategies for automaticly setting the hyperparameters (see [PSA]_ for more details and numerical experiments
    comparing the performances of the three strategies):

        - ``tuning_strategy == 1``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=1` (no relaxation).
          This is the most standard way of setting the parameters in the literature, does not leverage relaxation.
        - ``tuning_strategy == 2``: :math:`\gamma = \beta/1.9` (large step sizes) and :math:`\rho=1` (no relaxation).
          This strategy favours large step sizes forbidding the use of overrelaxation. When :math:`\beta=0`, same as first strategy.
        - ``tuning_strategy == 3``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=\delta - 0.1 > 1` (overrelaxation).
          This strategy chooses smaller step sizes, but performs overrelaxation.

    Once :math:`\gamma` chosen, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and :math:`\tau` as
    large as possible and relatively well-balanced --so that both the primal and dual variables converge at the same pace.
    Whenever possible, we therefore choose perfectly balanced parameters :math:`\sigma=\tau` saturating the convergence inequalities for a given value of :math:`\gamma`.

    * For :math:`\beta>0` and :math:`\mathcal{H}\neq 0` this yields:

        .. math::

           \frac{1}{\tau}-\tau\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad -\tau^2\Vert\mathbf{K}\Vert_{2}^2-\gamma\tau+1=0,

        which admits one positive root

        .. math::

           \tau=\sigma=\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}\left(-\frac{\gamma}{2}+\sqrt{\frac{\gamma^2}{4}+\Vert\mathbf{K}\Vert_{2}^2}\right).

    * For :math:`\beta>0` and :math:`\mathcal{H}=0` this yields: :math:`\tau=1/\gamma.`

    * For :math:`\beta=0` this yields:

        .. math::

           \tau=\sigma=\Vert\mathbf{K}\Vert_{2}^{-1}.

    When :math:`\tau` is provided (:math:`\tau = \tau_{1}`), but not :math:`\sigma`, the latter is chosen as:

    .. math::

       \frac{1}{\tau_{1}}-\sigma\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad \sigma=\left(\frac{1}{\tau_{1}}-\gamma\right)\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}.

    When :math:`\sigma` is provided (:math:`\sigma = \sigma_{1}`), but not :math:`\tau`, the latter is chosen as:

    .. math::

       \frac{1}{\tau}-\sigma_{1}\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad \tau=\frac{1}{\left(\gamma+\sigma_{1}\Vert\mathbf{K}\Vert_{2}^2\right)}.

    Warnings
    --------
    When values are provided for both :math:`\tau` and :math:`\sigma` it is assumed that the latter satisfy the convergence inequalities,
    but this check is not explicitly performed. Automatic selection of hyperparameters for the case of non-linear differentiable maps :math:`\mathcal{K}` is not supported yet.

    Examples
    --------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.`
    This problem can be solved via PDS with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`, :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,`
    :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1` and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::

       >>> import matplotlib.pyplot as plt
       >>> import numpy as np
       >>> from pycsou.opt.solver.pds import CV
       >>> from pycsou._dev import FirstDerivative, DownSampling, SquaredL2Norm, L1Norm

       >>> x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
       >>> D = FirstDerivative(size=x.size, kind="forward")
       >>> D.lipschitz(tol=1e-3)
       >>> downsampling = DownSampling(size=x.size, downsampling_factor=3)
       >>> downsampling.lipschitz()
       >>> y = downsampling(x)
       >>> l22_loss = (1 / 2) * SquaredL2Norm().argshift(-y)
       >>> fidelity = l22_loss * downsampling
       >>> H = 0.1 * L1Norm()

       >>> G = 0.01 * L1Norm()
       >>> cv = CV(f=fidelity, g=G, h=H, K=D)
       >>> x0, z0 = x * 0, x * 0
       >>> cv.fit(x0=x0, z0=z0)

       >>> estimate = cv.solution()
       >>> x_recons = estimate[0]
       >>>
       >>> plt.figure()
       >>> plt.stem(x, linefmt="C0-", markerfmt="C0o")
       >>> mask_ids = np.where(downsampling.downsampling_mask)[0]
       >>> markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
       >>> markerline.set_markerfacecolor("none")
       >>> plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
       >>> plt.legend(["Ground truth", "Observation", "CV Estimate"])
       >>> plt.show()

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CV`,
    :py:class:`~pycsou.opt.solver.pds.PD3O`,
    :py:class:`~pycsou.opt.solver.pds.ChambollePock`,
    :py:class:`~pycsou.opt.solver.pds.DouglasRachford`
    """

    def m_step(self):
        mst = self._mstate
        x_temp = self._g.prox(
            mst["x"] - mst["tau"] * self._f.grad(mst["x"]) - mst["tau"] * self._K.jacobian(mst["x"]).adjoint(mst["z"]),
            tau=mst["tau"],
        )
        if not self._h._name == "NullFunc":
            u = 2 * x_temp - mst["x"]
            z_temp = self._h.fenchel_prox(mst["z"] + mst["sigma"] * self._K(u), sigma=mst["sigma"])
            mst["z"] = mst["rho"] * z_temp + (1 - mst["rho"]) * mst["z"]
        mst["x"] = mst["rho"] * x_temp + (1 - mst["rho"]) * mst["x"]

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: typ.Optional[pyct.Real]
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of the parameter :math:`delta`.
        """

        if not issubclass(self._K.__class__, pyca.LinOp):
            msg = (
                f"Automatic selection of parameters is only supported in the case in which K is a linear operator. "
                f"Got operator of type {self._K.__class__}."
            )
            raise ValueError(msg)
        tau = None if tau == 0 else tau
        sigma = None if sigma == 0 else sigma

        if (tau is not None) and (sigma is None):
            assert tau > 0, f"Parameter tau must be positive, got {tau}."
            if self._h._name == "NullFunc":
                assert tau <= 1 / gamma, f"Parameter tau must be smaller than 1/gamma: {tau} > {1 / gamma}."
                sigma = 0
            else:
                if math.isfinite(self._K._lipschitz):
                    sigma = ((1 / tau) - gamma) * (1 / self._K._lipschitz**2)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is not None):
            assert sigma > 0
            if self._h._name == "NullFunc":
                tau = 1 / gamma
            else:
                if math.isfinite(self._K._lipschitz):
                    tau = 1 / (gamma + (sigma * self._K._lipschitz**2))
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is None):
            if self._beta > 0:
                if self._h._name == "NullFunc":
                    tau = 1 / gamma
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau = sigma = (1 / (self._K._lipschitz) ** 2) * (
                            (-gamma / 2) + math.sqrt((gamma**2 / 4) + self._K._lipschitz**2)
                        )
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
            else:
                if self._h._name == "NullFunc":
                    tau = 1
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau = sigma = 1 / self._K._lipschitz
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
        delta = (
            2
            if (self._beta == 0 or (isinstance(self._f, pycf.QuadraticFunc) and gamma <= self._beta))
            else 2 - self._beta / (2 * gamma)
        )
        return pycrt.coerce(tau), pycrt.coerce(sigma), pycrt.coerce(delta)


CV = CondatVu  #: Alias of :py:class:`~pycsou.opt.solver.pds.CondatVu`.


class PD3O(_PrimalDualSplitting):
    r"""
    Primal Dual Three-Operator Splitting (PD3O) algorithm.

    The *Primal Dual three Operator splitting (PD3O)* method is described in [PD3O]_.

    It can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\Psi(\mathbf{x}):=\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathcal{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator* :math:`\mathbf{K}`), with **operator norm**:

    .. math::

       \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    The algorithm has convergence guarantees for the case in which :math:`\mathcal{H}` is composed with a
    *linear operator* :math:`\mathbf{K}`. When :math:`\mathcal{F}=0`, convergence can be proven for *non-linear differentiable maps* :math:`\mathcal{K}` (see [NLCP]_).
    Note that this class does not support yet automatic selection of hyperparameters for the case of *non-linear differentiable maps* :math:`\mathcal{K}`.

    **Remark 3:**

    Assume that the following holds:

    * :math:`\gamma\geq\frac{\beta}{2}`,
    * :math:`\tau \in ]0, \frac{1}{\gamma}[`,
    * :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2 \leq 1`,
    * :math:`\delta = 2-\beta\tau/2 \in [1, 2[` and :math:`\rho \in (0, \delta]`,

    Then, there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M` solution
    s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}`
    *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star` respectively (Theorem 8.2 of [PSA]_), i.e.

    .. math::

       \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    Futhermore, when :math:`\rho=1`, the objective functional sequence :math:`\left(\Psi(\mathbf{x}_n)\right)_{n\in\mathbb{N}}` can be shown to converge towards
    its minimum :math:`\Psi^\ast` with rate :math:`o(1/\sqrt{n})` (Theorem 1 of [dPSA]_):

    .. math::

       \Psi(\mathbf{x}_n) - \Psi^\ast = o(1/\sqrt{n}).

    **Initialization parameters of the class:**

    f: DiffFunc | None
        Differentiable function :math:`\mathcal{F}`, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    g: ProxFunc | None
        Proximable function :math:`\mathcal{G}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    h: ProxFunc | None
        Proximable function :math:`\mathcal{H}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`, composed with a linear operator
        :math:`\mathbf{K}`.
    K: DiffMap | None
        Differentiable map :math:`\mathcal{K}` instance of :py:class:`~pycsou.abc.operator.DiffMap`, or a linear
        operator :math:`\mathbf{K}` instance of :py:class:`~pycsou.abc.operator.LinOp`.
    beta: float | None
        Lipschitz constant :math:`\beta` of the gradient of :math:`\mathcal{F}`. If not provided, it will be automatically estimated.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray | None
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    sigma: Real | None
        Dual step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See section below for more details.


    **Default values of the hyperparameters.**

    This class supports three strategies for automaticly setting the hyperparameters (see [PSA]_ for more details and numerical experiments
    comparing the performances of the three strategies):

        - ``tuning_strategy == 1``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=1` (no relaxation).
          This is the most standard way of setting the parameters in the literature, does not leverage relaxation.
        - ``tuning_strategy == 2``: :math:`\gamma = \beta/1.9` (large step sizes) and :math:`\rho=1` (no relaxation).
          This strategy favours large step sizes forbidding the use of overrelaxation. When :math:\beta=0`, same as first strategy.
        - ``tuning_strategy == 3``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=\delta - 0.1 > 1` (overrelaxation).
          This strategy chooses smaller step sizes, but performs overrelaxation.

    Once :math:`\gamma` chosen, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and :math:`\tau` as
    large as possible and relatively well-balanced --so that both the primal and dual variables converge at the same pace.
    Whenever possible, we therefore choose perfectly balanced parameters :math:`\sigma=\tau` saturating the convergence inequalities for a given value of :math:`\gamma`.

    In practice, the following linear programming optimization problem is solved:

    .. math::

       (\tau, \, \sigma) = \operatorname{arg} \max_{(\tau^{*}, \,  \sigma^{*})} \quad & \operatorname{log}(\tau^{*}) + \operatorname{log}(\sigma^{*})\\
       \text{s.t.} \quad & \operatorname{log}(\tau^{*}) + \operatorname{log}(\sigma^{*}) \leq 2\operatorname{log}(\Vert\mathbf{K}\Vert_{2})\\
       & \operatorname{log}(\tau^{*}) \leq -\operatorname{log}(\gamma)\\
       & \operatorname{log}(\tau^{*}) = \operatorname{log}(\sigma^{*}).

    When :math:`\tau \leq 1/\gamma` is given (i.e., :math:`\tau=\tau_{1}`), but not :math:`\sigma`, the latter is chosen as:

    .. math::

       \tau_{1}\sigma\Vert\mathbf{K}\Vert_{2}^2= 1 \quad\Longleftrightarrow\quad \sigma=\frac{1}{\tau_{1}\Vert\mathbf{K}\Vert_{2}^{2}}.

    When :math:`\sigma` is given (i.e., :math:`\sigma=\sigma_{1}`), but not :math:`\tau`, the latter is chosen as:

    .. math::

       \tau = \min \left\{\frac{1}{\gamma}, \frac{1}{\sigma_{1}\Vert\mathbf{K}\Vert_{2}^{2}}\right\}.

    Warnings
    --------
    When values are provided for both :math:`\tau` and :math:`\sigma` it is assumed that the latter satisfy the convergence inequalities,
    but this check is not explicitly performed. Automatic selection of hyperparameters for the case of non-linear differentiable maps :math:`\mathcal{K}` is not supported yet.


    Examples
    --------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.`
    This problem can be solved via PD3O with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`, :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,`
    :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1` and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::

       >>> import matplotlib.pyplot as plt
       >>> import numpy as np
       >>> from pycsou.opt.solver.pds import CV
       >>> from pycsou._dev import FirstDerivative, DownSampling, SquaredL2Norm, L1Norm

       >>> x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
       >>> D = FirstDerivative(size=x.size, kind="forward")
       >>> D.lipschitz(tol=1e-3)
       >>> downsampling = DownSampling(size=x.size, downsampling_factor=3)
       >>> downsampling.lipschitz()
       >>> y = downsampling(x)
       >>> l22_loss = (1 / 2) * SquaredL2Norm().argshift(-y)
       >>> fidelity = l22_loss * downsampling
       >>> H = 0.1 * L1Norm()

       >>> G = 0.01 * L1Norm()
       >>> pd3o = PD3O(f=fidelity, g=G, h=H, K=D)
       >>> x0, z0 = x * 0, x * 0
       >>> pd3o.fit(x0=x0, z0=z0)

       >>> estimate = pd3o.solution()
       >>> x_recons = estimate[0]
       >>>
       >>> plt.figure()
       >>> plt.stem(x, linefmt="C0-", markerfmt="C0o")
       >>> mask_ids = np.where(downsampling.downsampling_mask)[0]
       >>> markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
       >>> markerline.set_markerfacecolor("none")
       >>> plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
       >>> plt.legend(["Ground truth", "Observation", "PD3O Estimate"])
       >>> plt.show()
    """

    @pycrt.enforce_precision(i=["x0", "z0", "tau", "sigma", "rho"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray] = None,
        tau: typ.Optional[pyct.Real] = None,
        sigma: typ.Optional[pyct.Real] = None,
        rho: typ.Optional[pyct.Real] = None,
        tuning_strategy: typ.Literal[1, 2, 3] = 1,
    ):
        super().m_init(x0=x0, z0=z0, tau=tau, sigma=sigma, rho=rho, tuning_strategy=tuning_strategy)
        self._mstate["u"] = x0 if x0.ndim > 1 else x0.reshape(1, -1)

    def m_step(
        self,
    ):  # Slightly more efficient rewriting of iterations (216) of [PSA] with M=1. Faster than (185) since only one call to the adjoint and the gradient per iteration.
        mst = self._mstate
        mst["x"] = self._g.prox(mst["u"] - mst["tau"] * self._K.jacobian(mst["u"]).adjoint(mst["z"]), tau=mst["tau"])
        u_temp = mst["x"] - mst["tau"] * self._f.grad(mst["x"])
        if not self._h._name == "NullFunc":
            z_temp = self._h.fenchel_prox(
                mst["z"] + mst["sigma"] * self._K(mst["x"] + u_temp - mst["u"]), sigma=mst["sigma"]
            )
            mst["z"] = (1 - mst["rho"]) * mst["z"] + mst["rho"] * z_temp
        mst["u"] = (1 - mst["rho"]) * mst["u"] + mst["rho"] * u_temp

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: pyct.Real
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of :math:`\delta`.
        """

        if not issubclass(self._K.__class__, pyca.LinOp):
            msg = (
                f"Automatic selection of parameters is only supported in the case in which K is a linear operator. "
                f"Got operator of type {self._K.__class__}."
            )
            raise ValueError(msg)
        tau = None if tau == 0 else tau
        sigma = None if sigma == 0 else sigma

        if (tau is not None) and (sigma is None):
            assert 0 < tau <= 1 / gamma, f"tau must be positive and smaller than 1/gamma."
            if self._h._name == "NullFunc":
                sigma = 0
            else:
                if math.isfinite(self._K._lipschitz):
                    sigma = 1 / (tau * self._K._lipschitz**2)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is not None):
            assert sigma > 0, f"sigma must be positive, got {sigma}."
            if self._h._name == "NullFunc":
                tau = 1 / gamma
            else:
                if math.isfinite(self._K._lipschitz):
                    tau = min(1 / (sigma * self._K._lipschitz**2), 1 / gamma)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is None):
            if self._beta > 0:
                if self._h._name == "NullFunc":
                    tau = 1 / gamma
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau, sigma = self._optimize_step_sizes(gamma)
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
            else:
                if self._h._name == "NullFunc":
                    tau = 1
                    sigma = 0
                else:
                    if math.isfinite(self._K._lipschitz):
                        tau = sigma = 1 / self._K._lipschitz
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'lipschitz()'"
                        raise ValueError(msg)
        delta = 2 if self._beta == 0 else 2 - self._beta * tau / 2
        return pycrt.coerce(tau), pycrt.coerce(sigma), pycrt.coerce(delta)

    @pycrt.enforce_precision()
    def _optimize_step_sizes(self, gamma: pyct.Real) -> pyct.Real:
        r"""
        Optimize the primal/dual step sizes.

        Parameters
        ----------
        gamma: Real
            Gamma parameter.

        Returns
        -------
        Tuple[Real, Real]
            Sensible primal/dual step sizes.
        """
        import numpy as np
        from scipy.optimize import linprog

        c = np.array([-1, -1])
        A_ub = np.array([[1, 1], [1, 0]])
        b_ub = np.array([np.log(0.99) - 2 * np.log(self._K._lipschitz), np.log(1 / gamma)])
        A_eq = np.array([[1, -1]])
        b_eq = np.array([0])
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
        if not result.success:
            warnings.warn("Automatic parameter selection has not converged.", UserWarning)
        return np.exp(result.x)


def ChambollePock(
    g: typ.Optional[pyca.ProxFunc] = None,
    h: typ.Optional[pyca.ProxFunc] = None,
    K: typ.Optional[pyca.DiffMap] = None,
    base: typ.Type[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Chambolle and Pock primal-dual splitting method.

    This class is also accessible via the alias ``CP()``.

    The *Chambolle and Pock (CP) primal-dual splitting* method can be used to solve problems of the form:

    .. math::
      {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator* :math:`\mathbf{K}`), with **operator norm**:

    .. math::
         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.


    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.


    **Remark 2:**

    Automatic selection of parameters is not supported for *non-linear differentiable maps* :math:`\mathcal{K}`.

    **Remark 3:**

    The *Chambolle and Pock (CP) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{F}=0` in the :py:class:`~pycsou.opt.solver.pds.CondatVu` or :py:class:`~pycsou.opt.solver.pds.PD3O`
    algorithms. Chambolle and Pock originally introduced the algorithm without relaxation (:math:`\rho=1`) [CPA]_. Relaxed versions have been proposed afterwards [PSA]_.
    Chambolle and Pock's algorithm is also known as the *Primal-Dual Hybrid Gradient (PDHG)* algorithm. It can be seen as a preconditionned ADMM method [CPA]_.


    **Initialization parameters of the class:**

    g: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    h: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`, composed with a linear operator
        :math:`\mathbf{K}`.
    K: DiffMap | None
        Differentiable map :math:`\mathcal{K}` instance of :py:class:`~pycsou.abc.operator.DiffMap`, or a linear
        operator :math:`\mathbf{K}` instance of :py:class:`~pycsou.abc.operator.LinOp`.
    base: PrimalDual | None
        Specifies the base primal-dual algorithm (:py:class:`~pycsou.opt.solver.pds.CondatVu` (default)
        or :py:class:`~pycsou.opt.solver.pds.PD3O`). Both yield the same iterates but the rules for setting the hyperparameters may differ slightly.

    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray | None
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    sigma: Real | None
        Dual step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See base class for more details.

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CondatVu`, :py:class:`~pycsou.opt.solver.pds.PD3O`, :py:func:`~pycsou.opt.solver.pds.DouglasRachford`
    """
    kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
    obj = base(
        f=None,
        g=g,
        h=h,
        K=K,
        beta=0,
        **kwargs,
    )
    obj.__repr__ = lambda _: "ChambollePock"
    return obj


CP = ChambollePock  #: Alias of :py:class:`~pycsou.opt.solver.pds.ChambollePock`.


class LorisVerhoeven(PD3O):
    r"""
    Loris Verhoeven splitting algorithm.

    This class is also accessible via the alias ``LV()``.

     The *Loris Verhoeven (LV) primal-dual splitting* can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and *convex function* with *simple proximal operator*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator* :math:`\mathbf{K}`), with **operator norm**:

    .. math::
         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{F}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    Automatic selection of parameters is not supported for *non-linear differentiable maps* :math:`\mathcal{K}`.

    **Remark 3:**

    The *Loris and Verhoeven (CP) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{G}=0` in the :py:class:`~pycsou.opt.solver.pds.PD3O`
    algorithm.

    **Remark 4:**

    In the specific case when :math:`\mathcal{F}` is *quadratic*, then one can set :math:`\rho \in ]0,\delta[` with
    :math:`\delta=2` (see Theorem 4.3 in [PSA]_).

    **Initialization parameters of the class:**

    f: DiffFunc | None
        Differentiable function :math:`\mathcal{F}`, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    h: ProxFunc | None
        Proximable function :math:`\mathcal{H}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`, composed with a differentiable map
        :math:`\mathcal{K}`.
    K: DiffMap | None
        Differentiable map :math:`\mathcal{K}` instance of :py:class:`~pycsou.abc.operator.DiffMap`, or a linear
        operator :math:`\mathbf{K}` instance of :py:class:`~pycsou.abc.operator.LinOp`.
    beta: float | None
        Lipschitz constant :math:`\beta` of the gradient of :math:`\mathcal{F}`. If not provided, it will be automatically estimated.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray | None
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    sigma: Real | None
        Dual step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See base class for more details.

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.PD3O`, :py:class:`~pycsou.opt.solver.pds.DavisYin`, :py:class:`~pycsou.opt.solver.pgd.PGD`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`, :py:func:`~pycsou.opt.solver.pds.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pyca.DiffFunc] = None,
        h: typ.Optional[pyca.ProxFunc] = None,
        K: typ.Optional[pyca.DiffMap] = None,
        beta: typ.Optional[pyct.Real] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
        super().__init__(
            f=f,
            g=None,
            h=h,
            K=K,
            beta=beta,
            **kwargs,
        )

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: typ.Optional[pyct.Real]
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of the parameter :math:`delta`.
        """
        tau, sigma, _ = super()._set_step_sizes(tau=tau, sigma=sigma, gamma=gamma)
        delta = 2 if (self._beta == 0 or isinstance(self._f, pycf.QuadraticFunc)) else 2 - self._beta / (2 * gamma)
        return pycrt.coerce(tau), pycrt.coerce(sigma), pycrt.coerce(delta)


LV = LorisVerhoeven  #: Alias of :py:class:`~pycsou.opt.solver.pds.LorisVerhoeven`.


class DavisYin(PD3O):
    r"""
    Davis-Yin primal-dual splitting method.

    This class is also accessible via the alias ``DY()``.

    The *Davis and Yin (DY) primal-dual splitting* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{x}).}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient,
      for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.

    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**

    The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**

    The *Davis and Yin (DY) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{K}=\mathbf{I}`
    (identity) and :math:`\tau=1/\sigma` in the :py:class:`~pycsou.opt.solver.pds.PD3O` algorithm [PSA]_ (provided a suitable change of variable).

    **Initialization parameters of the class:**

    f: DiffFunc | None
        Differentiable function :math:`\mathcal{F}`, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    g: ProxFunc | None
        Proximable function :math:`\mathcal{G}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    h: ProxFunc | None
        Proximable function :math:`\mathcal{H}`, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    beta: float | None
        Lipschitz constant :math:`\beta` of the gradient of :math:`\mathcal{F}`. If not provided, it will be automatically estimated.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray | None
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    sigma: Real | None
        Dual step size. In this class, this parameter is overlooked and redefined by the automatic parameter selection.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See base class for more details.

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.PD3O`, :py:class:`~pycsou.opt.solver.pds.LorisVerhoeven`, :py:class:`~pycsou.opt.solver.pgd.PGD`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`, :py:func:`~pycsou.opt.solver.pds.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pyca.DiffFunc] = None,
        g: typ.Optional[pyca.ProxFunc] = None,
        h: typ.Optional[pyca.ProxFunc] = None,
        beta: typ.Optional[pyct.Real] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
        super().__init__(
            f=f,
            g=g,
            h=h,
            K=None,
            beta=beta,
            **kwargs,
        )

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: pyct.Real
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of :math:`\delta`.
        """
        if tau is not None:
            assert 0 < tau <= 1 / gamma, f"tau must be positive and smaller than 1/gamma."
        else:
            tau = 1.0 if self._beta == 0 else 1 / gamma

        delta = 2.0 if self._beta == 0 else 2 - self._beta * tau / 2

        return pycrt.coerce(tau), pycrt.coerce(1 / tau), pycrt.coerce(delta)


DY = DavisYin  #: Alias of :py:class:`~pycsou.opt.solver.pds.DavisYin`.


def DouglasRachford(
    g: typ.Optional[pyca.ProxFunc] = None,
    h: typ.Optional[pyca.ProxFunc] = None,
    base: typ.Type[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Douglas Rachford splitting algorithm.

    This class is also accessible via the alias ``DR()``.

    The *Douglas Rachford (DR) primal-dual splitting* can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{x}).}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are two *proper*, *lower semicontinuous* and *convex functions* with *simple proximal operators*.
    * The problem is *feasible* --i.e. there exists at least one solution.


    **Remark 1:**
    The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**
    The *Douglas Rachford (DR) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{F}=0`, :math:`\mathbf{K}=\mathbf{Id}` and :math:`\tau=1/\sigma`  in the :py:class:`~pycsou.opt.solver.pds.CondatVu` or :py:class:`~pycsou.opt.solver.pds.PD3O`
    algorithms. Douglas and Rachford originally introduced the algorithm without relaxation (:math:`\rho=1`), but relaxed versions have been proposed afterwards [PSA]_.
    When :math:`\rho=1`, Douglas Rachford's algorithm is *functionally equivalent* to ADMM (up to a change of variable, see [PSA]_ for a derivation).

    **Initialization parameters of the class:**

    g: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    h: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    base: PrimalDual | None
        Specifies the base primal-dual algorithm (:py:class:`~pycsou.opt.solver.pds.CondatVu` (default)
        or :py:class:`~pycsou.opt.solver.pds.PD3O`). Both yield identical algorithms.

    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray | None
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size. Defaults to 1.


    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CondatVu`, :py:class:`~pycsou.opt.solver.pds.PD3O`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`, :py:func:`~pycsou.opt.solver.pds.ForwardBackward`"""
    kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
    obj = base(f=None, g=g, h=h, K=None, beta=0, **kwargs)
    obj.__repr__ = lambda _: "DouglasRachford"

    def _set_step_sizes_custom(
        tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: pyct.Real
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        tau = 1.0 if tau is None else tau
        delta = 2.0
        return pycrt.coerce(tau), pycrt.coerce(1 / tau), pycrt.coerce(delta)

    obj._set_step_sizes = types.MethodType(_set_step_sizes_custom, obj)
    return obj


DR = DouglasRachford  #: Alias of :py:class:`~pycsou.opt.solver.pds.DouglasRachford`.


class ADMM(_PDS):
    r"""
    Alternating Direction Method of Multipliers algorithm.

    TODO add example and test

    The *Alternating Direction Method of Multipliers (ADMM)* method can be used to solve problems of the form:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N} \quad \mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K}\mathbf{x}),

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}_+` is a *convex functional* (see Remark 2 for additional
      requirements),
    * :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and
      *convex functional* with *simple proximal operator*,
    * :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *linear operator* with **operator norm**:
      :math:`\Vert{\mathbf{K}}\Vert_2`,
    * The problem is *feasible* --i.e. there exists at least one solution.

    **Remark 1:**
    The algorithm is still valid if one of the terms :math:`\mathcal{F}` or :math:`\mathcal{H}` is zero.

    **Remark 2:**
    This is an implementation of the algorithm described in Section 5.4 of [PSA]_, which handles the non-smooth
    composite term :math:`\mathcal{H}(\mathbf{K}\mathbf{x})` by means of a change of variable and an infimal
    postcomposition trick. This algorithm involves the :math:`\mathbf{x}`-minimization step

    .. math::
        :label: eq:x_minimization

        \mathcal{V} = \operatorname*{arg\,min}_{x \in \mathbb{R^N}} \quad \mathcal{F}(\mathbf{x}) + \frac1{2 \tau}
        \Vert \mathbf{K} \mathbf{x} - \mathbf{a} \Vert_2^2,

    which must be solved at *every* iteration of ADMM, where :math:`\tau` is the primal step size and
    :math:`\mathbf{a} \in \mathbb{R}^M` is an iteration-dependant vector. The following cases are covered in this
    implementation:

    * The user may provide a callable solver :math:`s: \mathbb{R}^M \times \mathbb{R} \to \mathbb{R}^N` to solve
      :math:numref:`eq:x_minimization`, so that :math:`s(\mathbf{a}, \tau) \in \mathcal{V}`. If ADMM is
      initialized with such a solver, then the latter is used to solve :math:numref:`eq:x_minimization` regardless of
      whether one of the following cases is met.

    * :math:`\mathbf{K}` is None (`i.e.`, the identity operator) and :math:`\mathcal{F}` is a
      :py:class:`~pycsou.abc.operator.ProxFunc`. Then, :math:numref:`eq:x_minimization` amounts to an application of the
      proximal operator of :math:`\mathcal{F}`. This case amounts to the classical ADMM algorithm described in Section
      5.3 of [PSA]_ (without the postcomposition trick).

    * :math:`\mathcal{F}` is a :py:class:`~pycsou.operator.func.quadratic.QuadraticFunc`. Then,
      :math:numref:`eq:x_minimization` amounts to solving the linear system

      .. math::

          \underbrace{\Big( \mathbf{Q} + \frac1\tau \mathbf{K}^* \mathbf{K} \Big)}_{\mathbf{A}} \mathbf{x} =
          \mathbf{b}

      over :math:`\mathbf{x} \in \mathbb{R}^N`, where :math:`\mathbf{Q} \in \mathbb{R}^{N \times N}` is the Hessian of
      :math:`\mathcal{F}` and :math:`\mathbf{b} \in \mathbb{R}^N` is an iteration-dependant vector. This linear system
      is solved via an inner-loop :py:class:`~pycsou.opt.solver.cg.CG` algorithm that involves the repeated
      application of the operator :math:`\mathbf{A}`, which consists of :math:`\mathbf{Q}` and of the Gramian of
      :math:`\mathbf{K}`. Hence, this scenario can be costly if these operators cannot be evaluated with fast algorithms
      .

    * :math:`\mathcal{F}` is a :py:class:`~pycsou.abc.operator.DiffFunc`. Then, :math:numref:`eq:x_minimization` is
      solved with an inner-loop :py:class:`~pycsou.opt.solver.nlcg.NLCG` algorithm, which involves the repeated
      application of the gradient of :math:`\mathcal{F}` and of the Gramian of :math:`\mathbf{K}`. Hence, this scenario
      can be costly if these operators cannot be evaluated with fast algorithms.

    **Remark 3:**
    Note that this algorithm **does not require** the diff-lipschitz constant of :math:`\mathcal{F}` to be known!

    **Initialization parameters of the class:**

    f: Func | None
        Functional, instance of :py:class:`~pycsou.abc.operator.Func`.
    h: ProxFunc | None
        Proximable functional, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    K: LinOp | None
        Linear operator, instance of :py:class:`~pycsou.abc.operator.LinOp`.
    solver: Callable[ndarray, float] | None
        Callable that solves the x-minimization step :math:numref:`eq:x_minimization`.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See base class for more details.
    cg_kwargs: dict | None
        Initialization parameters of the inner-loop :py:class:`~pycsou.opt.solver.cg.CG` algorithm (see Remark 2).
    cg_fit_kwargs: dict | None
        Parameters of the ``fit()`` method of the inner-loop :py:class:`~pycsou.opt.solver.cg.CG` algorithm (see Remark
        2).
    nlcg_kwargs: dict | None
        Initialization parameters of the inner-loop :py:class:`~pycsou.opt.solver.nlcg.NLCG` algorithm (see Remark 2).
    nlcg_fit_kwargs: dict | None
        Parameters of the ``fit()`` method of the inner-loop :py:class:`~pycsou.opt.solver.nlcg.NLCG` algorithm (see
        Remark 2).

    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CondatVu`, :py:class:`~pycsou.opt.solver.pds.PD3O`,
    :py:class:`~pycsou.opt.solver.pgd.PGD`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`,
    :py:func:`~pycsou.opt.solver.pds.DouglasRachford`"""

    def __init__(
        self,
        f: typ.Optional[pyca.Func] = None,
        h: typ.Optional[pyca.ProxFunc] = None,
        K: typ.Optional[pyca.DiffMap] = None,
        solver: typ.Callable[[pyct.NDArray, float], pyct.NDArray] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "u", "z")))

        x_update_method = "solver"  # Method for the x-minimization step
        g = None
        beta = 1  # The value of beta is irrelevant in the cg and nlcg scenarios
        if solver is None:
            if f.has(pyco.Property.PROXIMABLE) and K is None:
                x_update_method = "prox"
                g = f  # In this case, f corresponds to g in the _PDS terminology
                f = None
                beta = 0  # Beta does not apply to the prox x_update_method since f is None
            elif isinstance(f, pycf.QuadraticFunc):
                x_update_method = "cg"
                warnings.warn(
                    "An inner-loop conjugate gradient algorithm will be applied for the x-minimization step "
                    "of ADMM. This might lead to slow convergence.",
                    UserWarning,
                )
            elif f.has(pyco.Property.DIFFERENTIABLE_FUNCTION):
                x_update_method = "nlcg"
                warnings.warn(
                    "An inner-loop non-linear conjugate gradient algorithm will be applied for the "
                    "x-minimization step of ADMM. This might lead to slow convergence.",
                    UserWarning,
                )
            else:
                raise TypeError(
                    "Unsupported scenario: f must either be a ProxFunc (in which case K must be ``None``), a"
                    "QuadraticFunc, or a DiffMap. If neither of these scenarios hold, a solver must be provided for the"
                    "x-minimization step of ADMM."
                )
        self.solver = solver
        self.x_update_method = x_update_method

        super().__init__(
            f=f,
            g=g,
            h=h,
            K=K,
            beta=beta,
            **kwargs,
        )

    @pycrt.enforce_precision(i=["x0", "z0", "tau", "rho"], allow_None=True)
    def m_init(
        self,
        x0: pyct.NDArray,
        z0: typ.Optional[pyct.NDArray] = None,
        tau: typ.Optional[pyct.Real] = None,
        rho: typ.Optional[pyct.Real] = None,
        tuning_strategy: typ.Literal[1, 2, 3] = 1,
        **kwargs,
    ):
        super().m_init(x0=x0, z0=z0, tau=tau, sigma=None, rho=rho, tuning_strategy=tuning_strategy)
        mst = self._mstate  # shorthand
        mst["u"] = self._K(x0) if x0.ndim > 1 else self._K(x0).reshape(1, -1)
        # Conjugate gradient parameters
        mst["cg_kwargs"] = kwargs.get("cg_kwargs", dict(show_progress=False))
        mst["cg_fit_kwargs"] = kwargs.get("cg_fit_kwargs", dict())
        # Nonlinear conjugate gradient parameters
        mst["nlcg_kwargs"] = kwargs.get("nlcg_kwargs", dict(show_progress=False))
        mst["nlcg_fit_kwargs"] = kwargs.get("nlcg_fit_kwargs", dict())

    def m_step(
        self,
    ):  # Algorithm (130) in [PSA]. Paper -> code correspondence: L -> K, K -> -Id, c -> 0, y -> u, v -> z, g -> h
        mst = self._mstate
        mst["x"] = self._x_update(mst["u"] - mst["z"], tau=mst["tau"])
        z_temp = mst["z"] + self._K(mst["x"]) - mst["u"]
        if not self._h._name == "NullFunc":
            mst["u"] = self._h.prox(self._K(mst["x"]) + z_temp, tau=mst["tau"])
        mst["z"] = z_temp + (mst["rho"] - 1) * (self._K(mst["x"]) - mst["u"])

    def _x_update(self, arr: pyct.NDArray, tau: float) -> pyct.NDArray:
        if self.x_update_method == "solver":
            return self.solver(arr, tau)
        elif self.x_update_method == "prox":
            return self._g.prox(arr, tau=tau)
        elif self.x_update_method == "cg":
            from pycsou.opt.solver import CG

            b = (1 / tau) * self._K.adjoint(arr) - self._f._c.grad(arr)
            A = self._f._Q + (1 / tau) * self._K.gram()
            slvr = CG(A=A, **self._mstate["cg_kwargs"])
            slvr.fit(b=b, **self._mstate["cg_fit_kwargs"])
            return slvr.solution()
        elif self.x_update_method == "nlcg":
            from pycsou.opt.solver import NLCG

            quad_func = pycf.QuadraticFunc(self._K.gram(), pyca.LinFunc.from_array(-self._K.adjoint(arr)))
            slvr = NLCG(f=self._f + (1 / tau) * quad_func, **self._mstate["nlcg_kwargs"])
            slvr.fit(x0=self._mstate["x"], **self._mstate["nlcg_fit_kwargs"])  # Initialize NLCG with previous iterate
            return slvr.solution()

    def solution(self, which: typ.Literal["primal", "primal_h", "dual"] = "primal") -> pyct.NDArray:
        data, _ = self.stats()
        if which == "primal":
            assert "x" in data.keys(), "Primal variable x in dom(g) was not logged (declare it in log_var to log it)."
            return data.get("x")
        elif which == "primal_h":
            assert "u" in data.keys(), "Primal variable u in dom(h) was not logged (declare it in log_var to log it)."
            return data.get("u")
        elif which == "dual":
            assert "z" in data.keys(), "Dual variable z was not logged (declare it in log_var to log it)."
            return data.get("z") / self._mstate["tau"]
        else:
            raise ValueError(f"Parameter which must be one of ['primal', 'primal_h', 'dual'] got: {which}.")

    def _set_step_sizes(
        self, tau: typ.Optional[pyct.Real], sigma: typ.Optional[pyct.Real], gamma: pyct.Real
    ) -> typ.Tuple[pyct.Real, pyct.Real, pyct.Real]:
        if tau is not None:
            assert tau > 0, f"Parameter tau must be positive, got {tau}."
        else:
            tau = 1.0
        delta = 2.0
        return pycrt.coerce(tau), pycrt.coerce(1 / tau), pycrt.coerce(delta)


class ForwardBackward(CondatVu):
    r"""
    Forward-backward splitting algorithm.

    This class is also accessible via the alias ``FB()``.

    The *Forward-backward (FB) splitting* method can be used to solve problems of the form:

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

    The *Forward-backward (FB) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{H}=0` in the :py:class:`~pycsou.opt.solver.pds.CondatVu`
    algorithm. Mercier originally introduced the algorithm without relaxation (:math:`\rho=1`) [FB]_. Relaxed versions have been proposed afterwards [PSA]_.
    The Forward-backward algorithm is also known as the *Proximal Gradient Descent (PGD)* algorithm.
    For the accelerated version of PGD, see :py:class:`~pycsou.opt.solver.pgd`.


    **Initialization parameters of the class:**

    f: DiffFunc | None
        Differentiable function, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    g: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.
    beta: float | None
        Lipschitz constant of the gradient of :math:`\mathcal{F}`. If not provided, it will be automatically estimated.


    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    z0: NDArray
        (..., N) initial point(s) for the dual variable.
        If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    tau: Real | None
        Primal step size.
    rho: Real | None
        Momentum parameter.
    tuning_strategy: [1, 2, 3]
        Strategy to be employed when setting the hyperparameters (default to 1). See base class for more details.


    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CondatVu`, :py:class:`~pycsou.opt.solver.pds.PD3O`,:py:class:`~pycsou.opt.solver.pgd.PGD`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`, :py:func:`~pycsou.opt.solver.pds.DouglasRachford`"""

    def __init__(
        self,
        f: typ.Optional[pyca.DiffFunc] = None,
        g: typ.Optional[pyca.ProxFunc] = None,
        beta: typ.Optional[pyct.Real] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x",)))
        super().__init__(
            f=f,
            g=g,
            h=None,
            K=None,
            beta=beta,
            **kwargs,
        )


FB = ForwardBackward  #: Alias of :py:class:`~pycsou.opt.solver.pds.ForwardBackward`.


def ProximalPoint(
    g: typ.Optional[pyca.ProxFunc] = None,
    base: typ.Optional[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Proximal-point method algorithm.

    This class is also accessible via the alias ``PP()``.

    The *Proximal-point (PP)* method can be used to solve problems of the form:

    .. math::
       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{G}(\mathbf{x}).}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is *proper*, *lower semicontinuous* and *convex function* with *simple proximal operator*.
    * The problem is *feasible* --i.e. there exists at least one solution.

    The *Proximal-point (PP)* algorithm can be obtained by choosing :math:`\mathcal{F}=0` and :math:`\mathcal{G}=0` in the :py:class:`~pycsou.opt.solver.pds.CondatVu` or :py:class:`~pycsou.opt.solver.pds.PD3O`
    algorithms. The original version of the algorithm was introduced without relaxation (:math:`\rho=1`) [PP]_. Relaxed versions have been proposed afterwards [PSA]_.

    **Initialization parameters of the class:**

    g: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.

    **Parameterization** of the ``fit()`` method:

    x0: NDArray
        (..., N) initial point(s) for the primal variable.
    tau: Real | None
        Primal step size.
    rho: Real | None
        Momentum parameter.
    base: PrimalDual | None
        Primal dual base algorithm from which inherit mathematical iterative updates and default parameterization.
        Currently, the existing base classes are :py:class:`~pycsou.opt.solver.pds.CondatVu` (default), and
        :py:class:`~pycsou.opt.solver.pds.PD3O`


    See Also
    --------
    :py:class:`~pycsou.opt.solver.pds.CondatVu`, :py:class:`~pycsou.opt.solver.pds.PD3O`,:py:class:`~pycsou.opt.solver.pgd.PGD`, :py:func:`~pycsou.opt.solver.pds.ChambollePock`, :py:func:`~pycsou.opt.solver.pds.DouglasRachford`"""
    kwargs.update(log_var=kwargs.get("log_var", ("x",)))
    obj = base.__init__(
        f=None,
        g=g,
        h=None,
        K=None,
        beta=None,
        **kwargs,
    )

    obj.__repr__ = lambda _: "ProximalPoint"
    return obj


PP = ProximalPoint  #: Alias of :py:class:`~pycsou.opt.solver.pds.ProximalPoint`.
