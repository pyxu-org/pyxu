import math
import types
import typing as typ
import warnings

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.opt.stop as pxst

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


class _PrimalDualSplitting(pxa.Solver):
    r"""
    Base class for Primal-Dual Splitting (PDS) solvers.
    """
    TuningSpec = typ.Literal[1, 2, 3]  #: valid tuning_parameter values

    def __init__(
        self,
        f: typ.Optional[pxa.DiffFunc] = None,
        g: typ.Optional[pxa.ProxFunc] = None,
        h: typ.Optional[pxa.ProxFunc] = None,
        K: typ.Optional[pxa.DiffMap] = None,
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
            primal_dim_shape = f.dim_shape
        elif g is not None:
            primal_dim_shape = g.dim_shape
        else:
            primal_dim_shape = h.dim_shape

        if h is not None:
            dual_dim_shape = h.dim_shape
        elif K is not None:  # todo: isn't this elif-clause useless? For which solver is it triggered?
            dual_dim_shape = K.codim_shape
        else:
            dual_dim_shape = primal_dim_shape

        self._f = pxo.NullFunc(dim_shape=primal_dim_shape) if (f is None) else f
        self._g = pxo.NullFunc(dim_shape=primal_dim_shape) if (g is None) else g
        self._h = pxo.NullFunc(dim_shape=dual_dim_shape) if (h is None) else h
        if h is not None:
            self._K = pxo.IdentityOp(dim_shape=h.dim_shape) if (K is None) else K
        else:
            if K is None:
                K_dim_shape = f.dim_shape if f is not None else g.dim_shape
                self._K = pxo.NullOp(dim_shape=K_dim_shape, codim_shape=K_dim_shape)
            else:
                raise ValueError("Optional argument ``h`` mut be specified if ``K`` is not None.")
        self._objective_func = self._f + self._g + (self._h * self._K)

    def m_init(
        self,
        x0: pxt.NDArray,
        z0: typ.Optional[pxt.NDArray] = None,
        tau: typ.Optional[pxt.Real] = None,
        sigma: typ.Optional[pxt.Real] = None,
        rho: typ.Optional[pxt.Real] = None,
        beta: typ.Optional[pxt.Real] = None,
        tuning_strategy: TuningSpec = 1,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = x0
        mst["z"] = self._set_dual_variable(z0)
        self._tuning_strategy = int(tuning_strategy)
        self._beta = self._set_beta(beta)
        gamma = self._set_gamma(tuning_strategy)
        mst["tau"], mst["sigma"], delta = self._set_step_sizes(tau, sigma, gamma)
        mst["rho"] = self._set_momentum_term(rho, delta)

    def m_step(self):
        raise NotImplementedError

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        stop_crit_x = pxst.RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        stop_crit_z = pxst.RelError(
            eps=1e-4,
            var="z",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit_x & stop_crit_z if self._h._name != "NullFunc" else stop_crit_x

    def solution(self, which: typ.Literal["primal", "dual"] = "primal") -> pxt.NDArray:
        data, _ = self.stats()
        if which == "primal":
            assert "x" in data.keys(), "Primal variable x was not logged (declare it in log_var to log it)."
        elif which == "dual":
            assert "z" in data.keys(), "Dual variable z was not logged (declare it in log_var to log it)."
        else:
            raise ValueError(f"Parameter which must be one of ['primal', 'dual'] got: {which}.")
        return data.get("x") if which == "primal" else data.get("z")

    def objective_func(self) -> pxt.NDArray:
        return self._objective_func(self._mstate["x"])

    def _set_beta(self, beta: typ.Optional[pxt.Real]) -> pxt.Real:
        r"""
        Sets the Lipschitz constant.

        Returns
        -------
        beta: Real
            Lipschitz constant.
        """
        if beta is None:
            if math.isfinite(dl := self._f.diff_lipschitz):
                return dl
            else:
                msg = "beta: automatic inference not supported for operators with unbounded Lipschitz gradients."
            raise ValueError(msg)
        else:
            return beta

    def _set_dual_variable(self, z: typ.Optional[pxt.NDArray]) -> pxt.NDArray:
        r"""
        Initialize the dual variable if it is :py:obj:`None` by mapping the primal variable through the operator K.

        Returns
        -------
        NDArray
            Initialized dual variable.
        """
        if z is None:
            return self._K(self._mstate["x"].copy())
        else:
            return z

    def _set_gamma(self, tuning_strategy: typ.Literal[1, 2, 3]) -> pxt.Real:
        r"""
        Sets the gamma parameter according to the tuning strategy.

        Returns
        -------
        gamma: Real
            Gamma parameter.
        """
        return self._beta if tuning_strategy != 2 else self._beta / 1.9

    def _set_step_sizes(
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: pxt.Real,
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        raise NotImplementedError

    def _set_momentum_term(
        self,
        rho: typ.Optional[pxt.Real],
        delta: pxt.Real,
    ) -> pxt.Real:
        r"""
        Sets the momentum term according to Theorem 8.2 in [PSA]_.

        Returns
        -------
        pho: Real
            Momentum term.

        Notes
        -----
        The :math:`O(1/\sqrt(k))` objective functional convergence rate of (Theorem 1 of [dPSA]_) is for `\rho=1`.
        """
        if rho is None:
            rho = 1.0 if self._tuning_strategy != 3 else delta - 0.1
        else:
            assert rho <= delta, f"Parameter rho must be smaller than delta: {rho} > {delta}."
        return rho


_PDS = _PrimalDualSplitting  #: shorthand


class CondatVu(_PrimalDualSplitting):
    r"""
    Condat-Vu primal-dual splitting algorithm.

    This solver is also accessible via the alias :py:class:`~pyxu.opt.solver.CV`.

    The *Condat-Vu (CV)* primal-dual method is described in [CVS]_.  (This particular implementation is based on the
    pseudo-code Algorithm 7.1 provided in [FuncSphere]_ Chapter 7, Section1.)

    It can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathcal{K} \mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
      :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are *proper*, *lower semicontinuous* and
      *convex functions* with *simple proximal operators*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator*
      :math:`\mathbf{K}`), with **operator norm**:

      .. math::

         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or
      :math:`\mathcal{H}` is zero.

    * The algorithm has convergence guarantees when :math:`\mathcal{H}` is composed with a *linear operator*
      :math:`\mathbf{K}`.  When :math:`\mathcal{F}=0`, convergence can be proven for *non-linear differentiable maps*
      :math:`\mathcal{K}` (see [NLCP]_).  Note that :py:class:`~pyxu.opt.solver.CondatVu` does not yet support automatic
      selection of hyperparameters for the case of *non-linear differentiable maps* :math:`\mathcal{K}`.

    * Assume that either of the following holds:

      * :math:`\beta>0` and:

        - :math:`\gamma \geq \frac{\beta}{2}`,
        - :math:`\frac{1}{\tau}-\sigma\Vert\mathbf{K}\Vert_{2}^2\geq \gamma`,
        - :math:`\rho \in ]0,\delta[`, where :math:`\delta:=2-\frac{\beta}{2}\gamma^{-1}\in[1,2[` (:math:`\delta=2` is
          possible when :math:`\mathcal{F}` is *quadratic* and :math:`\gamma \geq \beta`, see [PSA]_).

      * :math:`\beta=0` and:

        - :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2\leq 1`,
        - :math:`\rho \in ]0,2[`.

      Then there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M` solution
      s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and
      :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star`
      respectively, i.e.

      .. math::

         \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and}
         \quad
         \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:obj:`None`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **g** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{G}`.
    * **h** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{H}`.
    * **K** (:py:class:`~pyxu.abc.DiffMap`, :py:class:`~pyxu.abc.LinOp`, :py:obj:`None`)
      --
      Differentiable map or linear operator :math:`\mathcal{K}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M) initial point(s) for the dual variable.
      If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **sigma** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Dual step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **beta** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Lipschitz constant :math:`\beta` of :math:`\nabla\mathcal{F}`.
      If not provided, it will be automatically estimated.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See section below for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    .. rubric:: Default hyperparameter values

    This class supports three strategies to automatically set the hyperparameters (see [PSA]_ for more details and
    numerical experiments comparing the performance of the three strategies):

    - ``tuning_strategy == 1``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=1` (no relaxation).

      This is the most standard way of setting the parameters in the literature.
    - ``tuning_strategy == 2``: :math:`\gamma = \beta/1.9` (large step sizes) and :math:`\rho=1` (no relaxation).

      This strategy favours large step sizes forbidding the use of overrelaxation.  When :math:`\beta=0`, coincides with
      the first strategy.
    - ``tuning_strategy == 3``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=\delta - 0.1 > 1` (overrelaxation).

      This strategy chooses smaller step sizes, but performs overrelaxation.

    Once :math:`\gamma` chosen, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and
    :math:`\tau` as large as possible and relatively well-balanced -- so that both the primal and dual variables
    converge at the same pace.  Whenever possible, we therefore choose perfectly balanced parameters :math:`\sigma=\tau`
    saturating the convergence inequalities for a given value of :math:`\gamma`.

    * For :math:`\beta>0` and :math:`\mathcal{H}\neq 0` this yields:

      .. math::

         \frac{1}{\tau}-\tau\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad -\tau^2\Vert\mathbf{K}\Vert_{2}^2-\gamma\tau+1=0,

      which admits one positive root

      .. math::

         \tau=\sigma=\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}\left(-\frac{\gamma}{2}+\sqrt{\frac{\gamma^2}{4}+\Vert\mathbf{K}\Vert_{2}^2}\right).

    * For :math:`\beta>0` and :math:`\mathcal{H}=0` this yields: :math:`\tau=1/\gamma.`

    * For :math:`\beta=0` this yields: :math:`\tau=\sigma=\Vert\mathbf{K}\Vert_{2}^{-1}`.

    When :math:`\tau` is provided (:math:`\tau = \tau_{1}`), but not :math:`\sigma`, the latter is chosen as:

    .. math::

       \frac{1}{\tau_{1}}-\sigma\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad \sigma=\left(\frac{1}{\tau_{1}}-\gamma\right)\frac{1}{\Vert\mathbf{K}\Vert_{2}^2}.

    When :math:`\sigma` is provided (:math:`\sigma = \sigma_{1}`), but not :math:`\tau`, the latter is chosen as:

    .. math::

       \frac{1}{\tau}-\sigma_{1}\Vert\mathbf{K}\Vert_{2}^2= \gamma \quad\Longleftrightarrow\quad \tau=\frac{1}{\left(\gamma+\sigma_{1}\Vert\mathbf{K}\Vert_{2}^2\right)}.

    Warning
    -------
    When values are provided for both :math:`\tau` and :math:`\sigma`, it is assumed that the latter satisfy the
    convergence inequalities, but no check is explicitly performed.  Automatic selection of hyperparameters for the case
    of non-linear differentiable maps :math:`\mathcal{K}` is not supported yet.

    Example
    -------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and
    :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.` This problem can
    be solved via PDS with :math:`\mathcal{F}(\mathbf{x})=
    \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`,
    :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,` :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1`
    and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import pyxu.operator as pxo
       from pyxu.operator import SubSample, PartialDerivative
       from pyxu.opt.solver import CV

       x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
       N = x.size

       D = PartialDerivative.finite_difference(dim_shape=(N,), order=(1,))

       downsample = SubSample(N, slice(None, None, 3))
       y = downsample(x)
       loss = (1 / 2) * pxo.SquaredL2Norm(y.size).argshift(-y)
       F = loss * downsample

       cv = CV(f=F, g=0.01 * pxo.L1Norm(N), h=0.1 * pxo.L1Norm((N)), K=D)
       x0, z0 = np.zeros((2, N))
       cv.fit(x0=x0, z0=z0)
       x_recons = cv.solution()

       plt.figure()
       plt.stem(x, linefmt="C0-", markerfmt="C0o")
       mask_ids = np.where(downsample.adjoint(np.ones_like(y)))[0]
       markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
       markerline.set_markerfacecolor("none")
       plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
       plt.legend(["Ground truth", "Observation", "CV Estimate"])
       plt.show()

    See Also
    --------
    :py:class:`~pyxu.opt.solver.CV`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.ChambollePock`,
    :py:class:`~pyxu.opt.solver.DouglasRachford`
    """

    def m_step(self):
        mst = self._mstate
        x_temp = self._g.prox(
            mst["x"] - mst["tau"] * self._f.grad(mst["x"]) - mst["tau"] * self._K.jacobian(mst["x"]).adjoint(mst["z"]),
            tau=mst["tau"],
        )
        if not self._h._name == "NullFunc":
            u = 2 * x_temp - mst["x"]
            z_temp = self._h.fenchel_prox(
                mst["z"] + mst["sigma"] * self._K(u),
                sigma=mst["sigma"],
            )
            mst["z"] = mst["rho"] * z_temp + (1 - mst["rho"]) * mst["z"]
        mst["x"] = mst["rho"] * x_temp + (1 - mst["rho"]) * mst["x"]

    def _set_step_sizes(
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: typ.Optional[pxt.Real],
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Sensible primal/dual step sizes and value of the parameter :math:`delta`.
        """
        # todo: this method in general is quite dirty.
        if not issubclass(self._K.__class__, pxa.LinOp):
            # todo: wrong! Must raise only if tau/sigma = None
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
                if math.isfinite(self._K.lipschitz):
                    sigma = ((1 / tau) - gamma) * (1 / self._K.lipschitz**2)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is not None):
            assert sigma > 0
            if self._h._name == "NullFunc":
                tau = 1 / gamma
            else:
                if math.isfinite(self._K.lipschitz):
                    tau = 1 / (gamma + (sigma * self._K.lipschitz**2))
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is None):
            if self._beta > 0:
                if self._h._name == "NullFunc":
                    tau = 1 / gamma
                    sigma = 0
                else:
                    if math.isfinite(self._K.lipschitz):
                        tau = sigma = (1 / (self._K.lipschitz) ** 2) * (
                            (-gamma / 2) + math.sqrt((gamma**2 / 4) + self._K.lipschitz**2)
                        )
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                        raise ValueError(msg)
            else:
                if self._h._name == "NullFunc":
                    tau = 1
                    sigma = 0
                else:
                    if math.isfinite(self._K.lipschitz):
                        tau = sigma = 1 / self._K.lipschitz
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                        raise ValueError(msg)
        delta = (
            2
            if (self._beta == 0 or (isinstance(self._f, pxa.QuadraticFunc) and gamma <= self._beta))
            else 2 - self._beta / (2 * gamma)
        )
        return tau, sigma, delta


CV = CondatVu  #: Alias of :py:class:`~pyxu.opt.solver.CondatVu`.


class PD3O(_PrimalDualSplitting):
    r"""
    Primal-Dual Three-Operator Splitting (PD3O) algorithm.

    The *Primal-Dual three Operator splitting (PD3O)* method is described in [PD3O]_.

    It can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\Psi(\mathbf{x}):=\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathcal{K} \mathbf{x}),}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
      :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are *proper*, *lower semicontinuous* and
      *convex functions* with *simple proximal operators*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator*
      :math:`\mathbf{K}`), with **operator norm**:

      .. math::

         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    Remarks
    -------
    * The problem is *feasible* -- i.e. there exists at least one solution.

    * The algorithm is still valid if one or more of the terms :math:`\mathcal{F}`, :math:`\mathcal{G}` or
      :math:`\mathcal{H}` is zero.

    * The algorithm has convergence guarantees for the case in which :math:`\mathcal{H}` is composed with a *linear
      operator* :math:`\mathbf{K}`.  When :math:`\mathcal{F}=0`, convergence can be proven for *non-linear
      differentiable maps* :math:`\mathcal{K}`. (See [NLCP]_.) Note that this class does not yet support automatic
      selection of hyperparameters for the case of *non-linear differentiable maps* :math:`\mathcal{K}`.

    * Assume that the following holds:

      * :math:`\gamma\geq\frac{\beta}{2}`,
      * :math:`\tau \in ]0, \frac{1}{\gamma}[`,
      * :math:`\tau\sigma\Vert\mathbf{K}\Vert_{2}^2 \leq 1`,
      * :math:`\delta = 2-\beta\tau/2 \in [1, 2[` and :math:`\rho \in (0, \delta]`.

      Then there exists a pair :math:`(\mathbf{x}^\star,\mathbf{z}^\star)\in\mathbb{R}^N\times \mathbb{R}^M` solution
      s.t. the primal and dual sequences of  estimates :math:`(\mathbf{x}_n)_{n\in\mathbb{N}}` and
      :math:`(\mathbf{z}_n)_{n\in\mathbb{N}}` *converge* towards :math:`\mathbf{x}^\star` and :math:`\mathbf{z}^\star`
      respectively (Theorem 8.2 of [PSA]_), i.e.

      .. math::

         \lim_{n\rightarrow +\infty}\Vert\mathbf{x}^\star-\mathbf{x}_n\Vert_2=0, \quad \text{and} \quad  \lim_{n\rightarrow +\infty}\Vert\mathbf{z}^\star-\mathbf{z}_n\Vert_2=0.

      Futhermore, when :math:`\rho=1`, the objective functional sequence
      :math:`\left(\Psi(\mathbf{x}_n)\right)_{n\in\mathbb{N}}` can be shown to converge towards its minimum
      :math:`\Psi^\ast` with rate :math:`o(1/\sqrt{n})` (Theorem 1 of [dPSA]_):

      .. math::

         \Psi(\mathbf{x}_n) - \Psi^\ast = o(1/\sqrt{n}).

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:obj:`None`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **g** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{G}`.
    * **h** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{H}`.
    * **K** (:py:class:`~pyxu.abc.DiffMap`, :py:class:`~pyxu.abc.LinOp`, :py:obj:`None`)
      --
      Differentiable map or linear operator :math:`\mathcal{K}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M) initial point(s) for the dual variable.
      If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **sigma** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Dual step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **beta** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Lipschitz constant :math:`\beta` of :math:`\nabla\mathcal{F}`.
      If not provided, it will be automatically estimated.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See section below for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    .. rubric:: Default hyperparameter values

    This class supports three strategies to automatically set the hyperparameters (see [PSA]_ for more details and
    numerical experiments comparing the performance of the three strategies):

    - ``tuning_strategy == 1``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=1` (no relaxation).

      This is the most standard way of setting the parameters in the literature.
    - ``tuning_strategy == 2``: :math:`\gamma = \beta/1.9` (large step sizes) and :math:`\rho=1` (no relaxation).

      This strategy favours large step sizes forbidding the use of overrelaxation.  When :math:`\beta=0`, coincides with
      the first strategy.
    - ``tuning_strategy == 3``: :math:`\gamma = \beta` (safe step sizes) and :math:`\rho=\delta - 0.1 > 1` (overrelaxation).

      This strategy chooses smaller step sizes, but performs overrelaxation.

    Once :math:`\gamma` chosen, the convergence speed  of the algorithm is improved by choosing :math:`\sigma` and
    :math:`\tau` as large as possible and relatively well-balanced -- so that both the primal and dual variables
    converge at the same pace.  Whenever possible, we therefore choose perfectly balanced parameters :math:`\sigma=\tau`
    saturating the convergence inequalities for a given value of :math:`\gamma`.

    In practice, the following linear programming optimization problem is solved:

    .. math::

       (\tau, \, \sigma) = \operatorname{arg} \max_{(\tau^{*}, \,  \sigma^{*})} \quad & \operatorname{log}(\tau^{*}) + \operatorname{log}(\sigma^{*})\\
       \text{s.t.} \quad & \operatorname{log}(\tau^{*}) + \operatorname{log}(\sigma^{*}) \leq 2\operatorname{log}(\Vert\mathbf{K}\Vert_{2})\\
       & \operatorname{log}(\tau^{*}) \leq -\operatorname{log}(\gamma)\\
       & \operatorname{log}(\tau^{*}) = \operatorname{log}(\sigma^{*}).

    When :math:`\tau \leq 1/\gamma` is given (i.e., :math:`\tau=\tau_{1}`), but not :math:`\sigma`, the latter is chosen
    as:

    .. math::

       \tau_{1}\sigma\Vert\mathbf{K}\Vert_{2}^2= 1 \quad\Longleftrightarrow\quad \sigma=\frac{1}{\tau_{1}\Vert\mathbf{K}\Vert_{2}^{2}}.

    When :math:`\sigma` is given (i.e., :math:`\sigma=\sigma_{1}`), but not :math:`\tau`, the latter is chosen as:

    .. math::

       \tau = \min \left\{\frac{1}{\gamma}, \frac{1}{\sigma_{1}\Vert\mathbf{K}\Vert_{2}^{2}}\right\}.

    Warning
    -------
    When values are provided for both :math:`\tau` and :math:`\sigma`, it is assumed that the latter satisfy the
    convergence inequalities, but no check is explicitly performed.  Automatic selection of hyperparameters for the case
    of non-linear differentiable maps :math:`\mathcal{K}` is not supported yet.

    Example
    -------
    Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and
    :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.` This problem can
    be solved via PD3O with :math:`\mathcal{F}(\mathbf{x})=
    \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`,
    :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,` :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1`
    and :math:`\mathbf{K}=\mathbf{D}`.

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import pyxu.operator as pxo
       from pyxu.operator import SubSample, PartialDerivative
       from pyxu.opt.solver import PD3O

       x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
       N = x.size

       D = PartialDerivative.finite_difference(dim_shape=(N,), order=(1,))

       downsample = SubSample(N, slice(None, None, 3))
       y = downsample(x)
       loss = (1 / 2) * pxo.SquaredL2Norm(y.size).argshift(-y)
       F = loss * downsample

       pd3o = PD3O(f=F, g=0.01 * pxo.L1Norm(N), h=0.1 * pxo.L1Norm((N)), K=D)
       x0, z0 = np.zeros((2, N))
       pd3o.fit(x0=x0, z0=z0)
       x_recons = pd3o.solution()

       plt.figure()
       plt.stem(x, linefmt="C0-", markerfmt="C0o")
       mask_ids = np.where(downsample.adjoint(np.ones_like(y)))[0]
       markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
       markerline.set_markerfacecolor("none")
       plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
       plt.legend(["Ground truth", "Observation", "PD3O Estimate"])
       plt.show()
    """

    def m_init(
        self,
        x0: pxt.NDArray,
        z0: typ.Optional[pxt.NDArray] = None,
        tau: typ.Optional[pxt.Real] = None,
        sigma: typ.Optional[pxt.Real] = None,
        rho: typ.Optional[pxt.Real] = None,
        tuning_strategy: _PDS.TuningSpec = 1,
    ):
        super().m_init(
            x0=x0,
            z0=z0,
            tau=tau,
            sigma=sigma,
            rho=rho,
            tuning_strategy=tuning_strategy,
        )

        # if x0 == u0 the first step wouldn't change x and the solver would stop at the first iteration
        if self._g._name == self._h._name == "NullFunc":
            self._mstate["u"] = x0 * 1.01
        else:
            self._mstate["u"] = x0.copy()

    def m_step(self):
        # Slightly more efficient rewriting of iterations (216) of [PSA] with M=1. Faster than (185) since only one call to the adjoint and the gradient per iteration.
        mst = self._mstate
        mst["x"] = self._g.prox(
            mst["u"] - mst["tau"] * self._K.jacobian(mst["u"]).adjoint(mst["z"]),
            tau=mst["tau"],
        )
        u_temp = mst["x"] - mst["tau"] * self._f.grad(mst["x"])
        if not self._h._name == "NullFunc":
            z_temp = self._h.fenchel_prox(
                mst["z"] + mst["sigma"] * self._K(mst["x"] + u_temp - mst["u"]),
                sigma=mst["sigma"],
            )
            mst["z"] = (1 - mst["rho"]) * mst["z"] + mst["rho"] * z_temp
        mst["u"] = (1 - mst["rho"]) * mst["u"] + mst["rho"] * u_temp

    def _set_step_sizes(
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: pxt.Real,
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of :math:`\delta`.
        """

        if not issubclass(self._K.__class__, pxa.LinOp):
            msg = (
                f"Automatic selection of parameters is only supported in the case in which K is a linear operator. "
                f"Got operator of type {self._K.__class__}."
            )
            raise ValueError(msg)
        tau = None if tau == 0 else tau
        sigma = None if sigma == 0 else sigma

        if (tau is not None) and (sigma is None):
            assert 0 < tau <= 1 / gamma, "tau must be positive and smaller than 1/gamma."
            if self._h._name == "NullFunc":
                sigma = 0
            else:
                if math.isfinite(self._K.lipschitz):
                    sigma = 1 / (tau * self._K.lipschitz**2)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is not None):
            assert sigma > 0, f"sigma must be positive, got {sigma}."
            if self._h._name == "NullFunc":
                tau = 1 / gamma
            else:
                if math.isfinite(self._K.lipschitz):
                    tau = min(1 / (sigma * self._K.lipschitz**2), 1 / gamma)
                else:
                    msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                    raise ValueError(msg)
        elif (tau is None) and (sigma is None):
            if self._beta > 0:
                if self._h._name == "NullFunc":
                    tau = 1 / gamma
                    sigma = 0
                else:
                    if math.isfinite(self._K.lipschitz):
                        tau, sigma = self._optimize_step_sizes(gamma)
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                        raise ValueError(msg)
            else:
                if self._h._name == "NullFunc":
                    tau = 1
                    sigma = 0
                else:
                    if math.isfinite(self._K.lipschitz):
                        tau = sigma = 1 / self._K.lipschitz
                    else:
                        msg = "Please compute the Lipschitz constant of the linear operator K by calling its method 'estimate_lipschitz()'"
                        raise ValueError(msg)
        delta = 2 if self._beta == 0 else 2 - self._beta * tau / 2
        return tau, sigma, delta

    def _optimize_step_sizes(self, gamma: pxt.Real) -> pxt.Real:
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
        b_ub = np.array([np.log(0.99) - 2 * np.log(self._K.lipschitz), np.log(1 / gamma)])
        A_eq = np.array([[1, -1]])
        b_eq = np.array([0])
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(None, None),
        )
        if not result.success:
            warnings.warn("Automatic parameter selection has not converged.", UserWarning)
        return np.exp(result.x)


def ChambollePock(
    g: typ.Optional[pxa.ProxFunc] = None,
    h: typ.Optional[pxa.ProxFunc] = None,
    K: typ.Optional[pxa.DiffMap] = None,
    base: typ.Type[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Chambolle-Pock primal-dual splitting method.

    Parameters
    ----------
    g: :py:class:`~pyxu.abc.ProxFunc`, None
        Proximable function :math:`\mathcal{G}`.
    h: :py:class:`~pyxu.abc.ProxFunc`, None
        Proximable function :math:`\mathcal{H}`.
    K: :py:class:`~pyxu.abc.DiffMap`, :py:class:`~pyxu.abc.LinOp`, None
        Differentiable map or linear operator :math:`\mathcal{K}`.
    base: :py:class:`~pyxu.opt.solver.CondatVu`, :py:class:`~pyxu.opt.solver.PD3O`
        Specifies the base primal-dual algorithm.
        (Default = :py:class:`~pyxu.opt.solver.CondatVu`)
    \*\*kwargs: ~collections.abc.Mapping
        Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.


    The *Chambolle-Pock (CP) primal-dual splitting* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}),}

    where:

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
      :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` are *proper*, *lower semicontinuous* and
      *convex functions* with *simple proximal operators*.
    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator*
      :math:`\mathbf{K}`), with **operator norm**:

      .. math::

         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    * Automatic selection of parameters is not supported for *non-linear differentiable maps* :math:`\mathcal{K}`.

    * The *Chambolle-Pock (CP) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{F}=0` in
      :py:class:`~pyxu.opt.solver.CondatVu` or :py:class:`~pyxu.opt.solver.PD3O`.  Chambolle and Pock originally
      introduced the algorithm without relaxation (:math:`\rho=1`) [CPA]_.  Relaxed versions have been proposed
      afterwards [PSA]_.  Chambolle-Pock's algorithm is also known as the *Primal-Dual Hybrid Gradient (PDHG)*
      algorithm.  It can be seen as a preconditionned ADMM method [CPA]_.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M) initial point(s) for the dual variable.
      If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **sigma** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Dual step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See `base` for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:func:`~pyxu.opt.solver.CP`,
    :py:class:`~pyxu.opt.solver.CondatVu`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """
    kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
    obj = base(
        f=None,
        g=g,
        h=h,
        K=K,
        **kwargs,
    )
    obj.__repr__ = lambda _: "ChambollePock"
    return obj


CP = ChambollePock  #: Alias of :py:func:`~pyxu.opt.solver.ChambollePock`.


class LorisVerhoeven(PD3O):
    r"""
    Loris-Verhoeven splitting algorithm.

    This solver is also accessible via the alias :py:class:`~pyxu.opt.solver.LV`.

    The *Loris-Verhoeven (LV) primal-dual splitting* can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K} \mathbf{x}),}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and
      *convex function* with *simple proximal operator*.

    * :math:`\mathcal{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *differentiable map* (e.g. a *linear operator*
      :math:`\mathbf{K}`), with **operator norm**:

      .. math::

         \Vert{\mathcal{K}}\Vert_2=\sup_{\mathbf{x}\in\mathbb{R}^N,\Vert\mathbf{x}\Vert_2=1} \Vert\mathcal{K}(\mathbf{x})\Vert_2.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one of the terms :math:`\mathcal{F}` or :math:`\mathcal{H}` is zero.

    * Automatic selection of parameters is not supported for *non-linear differentiable maps* :math:`\mathcal{K}`.

    * The *Loris-Verhoeven (CP) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{G}=0` in
      :py:class:`~pyxu.opt.solver.PD3O`.

    * In the specific case where :math:`\mathcal{F}` is *quadratic*, then one can set :math:`\rho \in ]0,\delta[` with
      :math:`\delta=2`. (See Theorem 4.3 in [PSA]_.)

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:obj:`None`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **h** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{H}`.
    * **K** (:py:class:`~pyxu.abc.DiffMap`, :py:class:`~pyxu.abc.LinOp`, :py:obj:`None`)
      --
      Differentiable map or linear operator :math:`\mathcal{K}`.
    * **beta** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Lipschitz constant :math:`\beta` of :math:`\nabla\mathcal{F}`.
      If not provided, it will be automatically estimated.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M) initial point(s) for the dual variable.
      If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **sigma** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Dual step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See :py:class:`~pyxu.opt.solver.PD3O` for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.DavisYin`,
    :py:class:`~pyxu.opt.solver.PGD`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pxa.DiffFunc] = None,
        h: typ.Optional[pxa.ProxFunc] = None,
        K: typ.Optional[pxa.DiffMap] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
        super().__init__(
            f=f,
            g=None,
            h=h,
            K=K,
            **kwargs,
        )

    def _set_step_sizes(
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: typ.Optional[pxt.Real],
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of the parameter :math:`delta`.
        """
        tau, sigma, _ = super()._set_step_sizes(tau=tau, sigma=sigma, gamma=gamma)
        delta = 2 if (self._beta == 0 or isinstance(self._f, pxa.QuadraticFunc)) else 2 - self._beta / (2 * gamma)
        return tau, sigma, delta


LV = LorisVerhoeven  #: Alias of :py:class:`~pyxu.opt.solver.LorisVerhoeven`.


class DavisYin(PD3O):
    r"""
    Davis-Yin primal-dual splitting method.

    This solver is also accessible via the alias :py:class:`~pyxu.opt.solver.DY`.

    The *Davis-Yin (DY) primal-dual splitting* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{x}),}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
      :math:`\mathcal{H}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` are *proper*, *lower semicontinuous* and
      *convex functions* with *simple proximal operators*.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    * The *Davis-Yin primal-dual splitting* method can be obtained by choosing :math:`\mathcal{K}=\mathbf{I}` (identity)
      and :math:`\tau=1/\sigma` in :py:class:`~pyxu.opt.solver.PD3O`, provided a suitable change of variable [PSA]_.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **g** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{G}`.
    * **h** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{H}`.
    * **beta** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Lipschitz constant :math:`\beta` of :math:`\nabla\mathcal{F}`.
      If not provided, it will be automatically estimated.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., N) initial point(s) for the dual variable.
      If ``None`` (default), then use ``x0`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **sigma** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Dual step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See :py:class:`~pyxu.opt.solver.PD3O` for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.LorisVerhoeven`,
    :py:class:`~pyxu.opt.solver.PGD`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pxa.DiffFunc],
        g: typ.Optional[pxa.ProxFunc] = None,
        h: typ.Optional[pxa.ProxFunc] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
        super().__init__(
            f=f,
            g=g,
            h=h,
            K=None,
            **kwargs,
        )

    def _set_step_sizes(
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: pxt.Real,
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        r"""
        Set the primal/dual step sizes.

        Returns
        -------
        Tuple[Real, Real, Real]
            Sensible primal/dual step sizes and value of :math:`\delta`.
        """
        if tau is not None:
            assert 0 < tau <= 1 / gamma, "tau must be positive and smaller than 1/gamma."
        else:
            tau = 1.0 if self._beta == 0 else 1 / gamma

        delta = 2.0 if self._beta == 0 else 2 - self._beta * tau / 2

        return tau, 1 / tau, delta


DY = DavisYin  #: Alias of :py:class:`~pyxu.opt.solver.DavisYin`.


def DouglasRachford(
    g: typ.Optional[pxa.ProxFunc] = None,
    h: typ.Optional[pxa.ProxFunc] = None,
    base: typ.Type[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Douglas-Rachford splitting algorithm.

    Parameters
    ----------
    g: :py:class:`~pyxu.abc.ProxFunc`, None
        Proximable function :math:`\mathcal{G}`.
    h: :py:class:`~pyxu.abc.ProxFunc`, None
        Proximable function :math:`\mathcal{H}`.
    base: :py:class:`~pyxu.opt.solver.CondatVu`, :py:class:`~pyxu.opt.solver.PD3O`
        Specifies the base primal-dual algorithm.
        (Default = :py:class:`~pyxu.opt.solver.CondatVu`)
    \*\*kwargs: ~collections.abc.Mapping
        Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.


    The *Douglas-Rachford (DR) primal-dual splitting* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \mathcal{G}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{x}),}

    where :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` and
    :math:`\mathcal{H}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` are *proper*, *lower semicontinuous* and
    *convex functions* with *simple proximal operators*.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one of the terms :math:`\mathcal{G}` or :math:`\mathcal{H}` is zero.

    * The *Douglas-Rachford (DR) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{F}=0`,
      :math:`\mathbf{K}=\mathbf{Id}` and :math:`\tau=1/\sigma` in :py:class:`~pyxu.opt.solver.CondatVu` or
      :py:class:`~pyxu.opt.solver.PD3O`.  Douglas and Rachford originally introduced the algorithm without relaxation
      (:math:`\rho=1`), but relaxed versions have been proposed afterwards [PSA]_.  When :math:`\rho=1`,
      Douglas-Rachford's algorithm is *functionally equivalent* to ADMM (up to a change of variable, see [PSA]_ for a
      derivation).

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., N) initial point(s) for the dual variable.
      If ``None`` (default), then use ``x0`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size. Defaults to 1.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:class:`~pyxu.opt.solver.CondatVu`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.ForwardBackward`
    """
    kwargs.update(log_var=kwargs.get("log_var", ("x", "z")))
    obj = base(f=None, g=g, h=h, K=None, **kwargs)
    obj.__repr__ = lambda _: "DouglasRachford"

    def _set_step_sizes_custom(
        obj: typ.Type[_PrimalDualSplitting],
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: pxt.Real,
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        tau = 1.0 if tau is None else tau
        delta = 2.0
        return tau, 1 / tau, delta

    obj._set_step_sizes = types.MethodType(_set_step_sizes_custom, obj)
    return obj


DR = DouglasRachford  #: Alias of :py:func:`~pyxu.opt.solver.DouglasRachford`.


class ADMM(_PDS):
    r"""
    Alternating Direction Method of Multipliers.

    The *Alternating Direction Method of Multipliers (ADMM)* can be used to solve problems of the form:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N} \quad \mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{H}(\mathbf{K}\mathbf{x}),

    where (see below for additional details on the assumptions):

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and
      *convex functional* potentially with *simple proximal operator* or *Lipschitz-differentiable*,
    * :math:`\mathcal{H}:\mathbb{R}^M\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower semicontinuous* and
      *convex functional* with *simple proximal operator*,
    * :math:`\mathbf{K}:\mathbb{R}^N\rightarrow \mathbb{R}^M` is a *linear operator* with **operator norm**
      :math:`\Vert{\mathbf{K}}\Vert_2`.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if either :math:`\mathcal{F}` or :math:`\mathcal{H}` is zero.

    * When :math:`\mathbf{K} = \mathbf{I}_{N}`, ADMM is equivalent to the :py:func:`~pyxu.opt.solver.DouglasRachford`
      method (up to a change of variable, see [PSA]_ for a derivation).

    * This is an implementation of the algorithm described in Section 5.4 of [PSA]_, which handles the non-smooth
      composite term :math:`\mathcal{H}(\mathbf{K}\mathbf{x})` by means of a change of variable and an infimal
      postcomposition trick.  The update equation of this algorithm involves the following
      :math:`\mathbf{x}`-minimization step:

      .. math::
         :label: eq:x_minimization

         \mathcal{V} = \operatorname*{arg\,min}_{\mathbf{x} \in \mathbb{R}^N} \quad \mathcal{F}(\mathbf{x}) + \frac1{2 \tau}
         \Vert \mathbf{K} \mathbf{x} - \mathbf{a} \Vert_2^2,

      where :math:`\tau` is the primal step size and :math:`\mathbf{a} \in \mathbb{R}^M` is an iteration-dependant
      vector.

      The following cases are covered in this implementation:

      - :math:`\mathbf{K}` is ``None`` (i.e. the identity operator) and :math:`\mathcal{F}` is a
        :py:class:`~pyxu.abc.ProxFunc`.  Then, :math:numref:`eq:x_minimization` has a single solution provided by
        :math:`\operatorname*{prox}_{\tau \mathcal{F}}(\mathbf{a})`.  This yields the *classical ADMM algorithm*
        described in Section 5.3 of [PSA]_ (i.e. without the postcomposition trick).

      - :math:`\mathcal{F}` is a :py:class:`~pyxu.abc.QuadraticFunc`, i.e.  :math:`\mathcal{F}(\mathbf{x})=\frac{1}{2}
        \langle\mathbf{x}, \mathbf{Q}\mathbf{x}\rangle + \mathbf{c}^T\mathbf{x} + t`.  Then the unique solution to
        :math:numref:`eq:x_minimization` is obtained by solving a linear system of the form:

        .. math::
           :label: eq:linear_system

           \Big( \mathbf{Q} + \frac1\tau \mathbf{K}^* \mathbf{K} \Big) \mathbf{x} =
           \frac1\tau \mathbf{K}^\ast\mathbf{a}-\mathbf{c}, \qquad \mathbf{x} \in \mathbb{R}^N.

        This linear system is solved via a sub-iterative :py:class:`~pyxu.opt.solver.CG` algorithm involving the
        repeated application of :math:`\mathbf{Q}` and :math:`\mathbf{K}^{*}\mathbf{K}`.  This sub-iterative scheme may
        be computationally intensive if these operators do not admit fast matrix-free implementations.

      - :math:`\mathcal{F}` is a :py:class:`~pyxu.abc.DiffFunc`.  Then, :math:numref:`eq:x_minimization` is solved via a
        sub-iterative :py:class:`~pyxu.opt.solver.NLCG` algorithm involving repeated evaluation of
        :math:`\nabla\mathcal{F}` and :math:`\mathbf{K}^{*}\mathbf{K}`.  This sub-iterative scheme may be costly if
        these operators cannot be evaluated with fast algorithms.  In this scenario, the use of multiple initial points
        in :py:meth:`~pyxu.abc.Solver.fit` is not supported.

      The user may also provide a *custom* callable solver :math:`s: \mathbb{R}^M \times \mathbb{R} \to \mathbb{R}^N`,
      taking as input :math:`(\mathbf{a}, \tau)` and solving :math:numref:`eq:x_minimization`, i.e. :math:`s(\mathbf{a},
      \tau) \in \mathcal{V}`. (See example below.) If :py:class:`~pyxu.opt.solver.ADMM` is initialized with such a
      solver, then the latter is used to solve :math:numref:`eq:x_minimization` regardless of whether one of the
      above-mentioned cases is met.

    * Unlike traditional implementations of ADMM, :py:class:`~pyxu.opt.solver.ADMM` supports relaxation, i.e.
      :math:`\rho\neq 1`.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Differentiable or proximable function :math:`\mathcal{F}`.
    * **h** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{H}`.
    * **K** (:py:class:`~pyxu.abc.LinOp`, :py:obj:`None`)
      --
      Linear operator :math:`\mathbf{K}`.
    * **solver** (:py:class:`~collections.abc.Callable`, :py:obj:`None`)
      --
      Custom callable to solve the :math:`\mathbf{x}`-minimization step :math:numref:`eq:x_minimization`.

      If provided, `solver` must have the `NumPy signature
      <https://numpy.org/neps/nep-0020-gufunc-signature-enhancement.html>`_ ``(n), (1) -> (n)``.
    * **solver_kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Keyword parameters passed to the ``__init__()`` method of sub-iterative :py:class:`~pyxu.opt.solver.CG` or
      :py:class:`~pyxu.opt.solver.NLCG` solvers.

      `solver_kwargs` is ignored if `solver` provided.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., M) initial point(s) for the dual variable.
      If ``None`` (default), then use ``K(x0)`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter for relaxation.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See base class for more details.
    * **solver_kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Keyword parameters passed to the ``fit()`` method of sub-iterative
      :py:class:`~pyxu.opt.solver.CG` or :py:class:`~pyxu.opt.solver.NLCG` solvers.

      `solver_kwargs` is ignored if `solver` was provided in ``__init__()``.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    Warning
    -------
    ``tuning_strategy`` docstring says to look at base class for details, but nothing mentioned there!

    Example
    -------
    Consider the following optimization problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda \|\mathbf{D}\mathbf{x}\|_1,

    with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete second-order derivative operator,
    :math:`\mathbf{G}\in\mathbb{R}^{M\times N}, \, \mathbf{y}\in\mathbb{R}^M, \lambda>0.` This problem can be solved via
    ADMM with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`,
    :math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{D}\mathbf{x}\|_1,` and :math:`\mathbf{K}=\mathbf{D}`.  The
    functional :math:`\mathcal{F}` being quadratic, the :math:`\mathbf{x}`-minimization step consists in solving a
    linear system of the form :math:numref:`eq:linear_system`.  Here, we demonstrate how to provide a custom solver,
    which consists in applying a matrix inverse, for this step.  Otherwise, a sub-iterative
    :py:class:`~pyxu.opt.solver.CG` algorithm would have been used automatically instead.

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import pyxu.abc as pxa
       import pyxu.operator as pxo
       import scipy as sp
       from pyxu.opt.solver import ADMM

       N = 100  # Dimension of the problem

       # Generate piecewise-linear ground truth
       x_gt = np.array([10, 25, 60, 90])  # Knot locations
       a_gt = np.array([2, -4, 3, -2])  # Amplitudes of the knots
       gt = np.zeros(N)  # Ground-truth signal
       for n in range(len(x_gt)):
           gt[x_gt[n] :] += a_gt[n] * np.arange(N - x_gt[n]) / N

       # Generate data (noisy samples at random locations)
       M = 20  # Number of data points
       rng = np.random.default_rng(seed=0)
       x_samp = rng.choice(np.arange(N // M), size=M) + np.arange(N, step=N // M)  # sampling locations
       sigma = 2 * 1e-2  # noise variance
       y = gt[x_samp] + sigma * rng.standard_normal(size=M)  # noisy data points

       # Data-fidelity term
       subsamp_mat = sp.sparse.lil_matrix((M, N))
       for i in range(M):
           subsamp_mat[i, x_samp[i]] = 1
       G = pxa.LinOp.from_array(subsamp_mat.tocsr())
       F = 1 / 2 * pxo.SquaredL2Norm(dim=y.size).argshift(-y) * G
       F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

       # Regularization term (promotes sparse second derivatives)
       deriv_mat = sp.sparse.diags(diagonals=[1, -2, 1], offsets=[0, 1, 2], shape=(N - 2, N))
       D = pxa.LinOp.from_array(deriv_mat)
       _lambda = 1e-1  # regularization parameter
       H = _lambda * pxo.L1Norm(dim=D.codim)

       # Solver for ADMM
       tau = 1 / _lambda  # internal ADMM parameter
       # Inverse operator to solve the linear system
       A_inv = sp.linalg.inv(G.gram().asarray() + (1 / tau) * D.gram().asarray())


       def solver_ADMM(arr, tau):
           b = (1 / tau) * D.adjoint(arr) + G.adjoint(y)
           return A_inv @ b.squeeze()


       # Solve optimization problem
       admm = ADMM(f=F, h=H, K=D, solver=solver_ADMM,show_progress=False)  # with solver
       admm.fit(x0=np.zeros(N), tau=tau)
       x_opt = admm.solution()  # reconstructed signal

       # Plots
       plt.figure()
       plt.plot(np.arange(N), gt, label="Ground truth")
       plt.plot(x_samp, y, "kx", label="Noisy data points")
       plt.plot(np.arange(N), x_opt, label="Reconstructed signal")
       plt.legend()

    See Also
    --------
    :py:class:`~pyxu.opt.solver.CondatVu`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.PGD`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pxa.Func] = None,
        h: typ.Optional[pxa.ProxFunc] = None,
        K: typ.Optional[pxa.DiffMap] = None,
        solver: typ.Callable[[pxt.NDArray, float], pxt.NDArray] = None,
        solver_kwargs: typ.Optional[dict] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", "u", "z")))

        x_update_solver = "custom"  # Method for the x-minimization step
        g = None
        if solver is None:
            if f is None:
                if h is None:
                    msg = " ".join(
                        [
                            "Cannot minimize always-0 functional.",
                            "At least one of Parameter[f, h] must be specified.",
                        ]
                    )
                    raise ValueError(msg)
                if K is None:  # Prox scenario (classical ADMM)
                    f = pxo.NullFunc(h.dim_shape)
                else:  # Sub-iterative CG scenario
                    f = pxa.QuadraticFunc(
                        dim_shape=h.dim_shape,
                        codim_shape=(1,),
                        Q=pxo.NullOp(dim_shape=h.dim_shape, codim_shape=h.dim_shape),
                        c=pxo.NullFunc(dim_shape=h.dim_shape),
                    )
            if f.has(pxa.Property.PROXIMABLE) and K is None:
                x_update_solver = "prox"
                g = f  # In this case, f corresponds to g in the _PDS terminology
                f = None
            elif isinstance(f, pxa.QuadraticFunc):
                x_update_solver = "cg"
                self._K_gram = K.gram()
                warnings.warn(
                    "A sub-iterative conjugate gradient algorithm is used for the x-minimization step "
                    "of ADMM. This might be computationally expensive.",
                    UserWarning,
                )
            elif f.has(pxa.Property.DIFFERENTIABLE_FUNCTION):
                x_update_solver = "nlcg"
                self._K_gram = K.gram()
                warnings.warn(
                    "A sub-iterative non-linear conjugate gradient algorithm is used for the "
                    "x-minimization step of ADMM. This might be computationally expensive.",
                    UserWarning,
                )
            else:
                raise TypeError(
                    "Unsupported scenario: f must either be a ProxFunc (in which case K must be None), a "
                    "QuadraticFunc, or a DiffMap. If neither of these scenarios hold, a solver must be provided for "
                    "the x-minimization step of ADMM."
                )
        self._solver = solver
        self._x_update_solver = x_update_solver
        self._init_kwargs = solver_kwargs if solver_kwargs is not None else dict(show_progress=False)

        super().__init__(
            f=f,
            g=g,
            h=h,
            K=K,
            **kwargs,
        )

    def m_init(
        self,
        x0: pxt.NDArray,
        z0: typ.Optional[pxt.NDArray] = None,
        tau: typ.Optional[pxt.Real] = None,
        rho: typ.Optional[pxt.Real] = None,
        tuning_strategy: _PDS.TuningSpec = 1,
        solver_kwargs: typ.Optional[dict] = None,
        **kwargs,
    ):
        super().m_init(
            x0=x0,
            z0=z0,
            tau=tau,
            sigma=None,
            rho=rho,
            tuning_strategy=tuning_strategy,
        )
        mst = self._mstate  # shorthand
        mst["u"] = self._K(x0)

        # Fit kwargs of sub-iterative solver
        if solver_kwargs is None:
            solver_kwargs = dict()
        self._fit_kwargs = solver_kwargs

    def m_step(self):
        # Algorithm (145) in [PSA]. Paper -> code correspondence: L -> K, K -> -Id, c -> 0, y -> u, v -> z, g -> h
        mst = self._mstate
        mst["x"] = self._x_update(mst["u"] - mst["z"], tau=mst["tau"])
        z_temp = mst["z"] + self._K(mst["x"]) - mst["u"]
        if not self._h._name == "NullFunc":
            mst["u"] = self._h.prox(self._K(mst["x"]) + z_temp, tau=mst["tau"])
        mst["z"] = z_temp + (mst["rho"] - 1) * (self._K(mst["x"]) - mst["u"])

    def _x_update(self, arr: pxt.NDArray, tau: float) -> pxt.NDArray:
        if self._x_update_solver == "custom":
            return self._solver(arr, tau)
        elif self._x_update_solver == "prox":
            return self._g.prox(arr, tau=tau)
        elif self._x_update_solver == "cg":
            from pyxu.opt.solver import CG

            b = (1 / tau) * self._K.adjoint(arr) - self._f._c.grad(arr)
            A = self._f._Q + (1 / tau) * self._K_gram
            slvr = CG(A=A, **self._init_kwargs)
            slvr.fit(b=b, x0=self._mstate["x"].copy(), **self._fit_kwargs)  # Initialize CG with previous iterate
            return slvr.solution()
        elif self._x_update_solver == "nlcg":
            from pyxu.opt.solver import NLCG

            c = pxa.LinFunc.from_array(-self._K.adjoint(arr))
            quad_func = pxa.QuadraticFunc(dim_shape=self._f.dim_shape, codim_shape=(1,), Q=2 * self._K_gram, c=c)
            slvr = NLCG(f=self._f + (1 / tau) * quad_func, **self._init_kwargs)
            slvr.fit(x0=self._mstate["x"].copy(), **self._fit_kwargs)  # Initialize NLCG with previous iterate
            return slvr.solution()

    def solution(self, which: typ.Literal["primal", "primal_h", "dual"] = "primal") -> pxt.NDArray:
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
        self,
        tau: typ.Optional[pxt.Real],
        sigma: typ.Optional[pxt.Real],
        gamma: pxt.Real,
    ) -> tuple[pxt.Real, pxt.Real, pxt.Real]:
        if tau is not None:
            assert tau > 0, f"Parameter tau must be positive, got {tau}."
        else:
            tau = 1.0
        delta = 2.0
        return tau, 1 / tau, delta


class ForwardBackward(CondatVu):
    r"""
    Forward-backward splitting algorithm.

    This solver is also accessible via the alias :py:class:`~pyxu.opt.solver.FB`.

    The *Forward-backward (FB) splitting* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x}),}

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is *proper*, *lower semicontinuous* and
      *convex function* with *simple proximal operator*.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The algorithm is still valid if one of the terms :math:`\mathcal{F}` or :math:`\mathcal{G}` is zero.

    * The *Forward-backward (FB) primal-dual splitting* method can be obtained by choosing :math:`\mathcal{H}=0` in
      :py:class:`~pyxu.opt.solver.CondatVu`.  Mercier originally introduced the algorithm without relaxation
      (:math:`\rho=1`) [FB]_.  Relaxed versions have been proposed afterwards [PSA]_.  The Forward-backward algorithm is
      also known as the *Proximal Gradient Descent (PGD)* algorithm.  For the accelerated version of PGD, use
      :py:class:`~pyxu.opt.solver.PGD`.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:obj:`None`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **g** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{G}`.
    * **beta** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Lipschitz constant :math:`\beta` of :math:`\nabla\mathcal{F}`.
      If not provided, it will be automatically estimated.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **z0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., N) initial point(s) for the dual variable.
      If ``None`` (default), then use ``x0`` as the initial point for the dual variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **tuning_strategy** (1, 2, 3)
      --
      Strategy to be employed when setting the hyperparameters (default to 1).
      See :py:class:`~pyxu.opt.solver.CondatVu` for more details.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:class:`~pyxu.opt.solver.CondatVu`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.PGD`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """

    def __init__(
        self,
        f: typ.Optional[pxa.DiffFunc] = None,
        g: typ.Optional[pxa.ProxFunc] = None,
        **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x",)))
        super().__init__(
            f=f,
            g=g,
            h=None,
            K=None,
            **kwargs,
        )


FB = ForwardBackward  #: Alias of :py:class:`~pyxu.opt.solver.ForwardBackward`.


def ProximalPoint(
    g: typ.Optional[pxa.ProxFunc] = None,
    base: typ.Optional[_PrimalDualSplitting] = CondatVu,
    **kwargs,
):
    r"""
    Proximal-point method.

    Parameters
    ----------
    g: :py:class:`~pyxu.abc.ProxFunc`
        Proximable function :math:`\mathcal{G}`.
    base: :py:class:`~pyxu.opt.solver.CondatVu`, :py:class:`~pyxu.opt.solver.PD3O`
        Specifies the base primal-dual algorithm from which mathematical updates are inherited.
        (Default = :py:class:`~pyxu.opt.solver.CondatVu`)
    \*\*kwargs: ~collections.abc.Mapping
        Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.


    The *Proximal-point (PP)* method can be used to solve problems of the form:

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{G}(\mathbf{x}),}

    where :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is *proper*, *lower semicontinuous* and
    *convex function* with *simple proximal operator*.

    Remarks
    -------
    * The problem is *feasible*, i.e. there exists at least one solution.

    * The *Proximal-point* algorithm can be obtained by choosing :math:`\mathcal{F}=0` and :math:`\mathcal{H}=0` in
      :py:class:`~pyxu.opt.solver.CondatVu` or :py:class:`~pyxu.opt.solver.PD3O`.  The original version of the algorithm
      was introduced without relaxation (:math:`\rho=1`) [PP]_.  Relaxed versions have been proposed afterwards [PSA]_.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s) for the primal variable.
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Primal step size.
    * **rho** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Momentum parameter.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    See Also
    --------
    :py:class:`~pyxu.opt.solver.PP`,
    :py:class:`~pyxu.opt.solver.CondatVu`,
    :py:class:`~pyxu.opt.solver.PD3O`,
    :py:class:`~pyxu.opt.solver.PGD`,
    :py:func:`~pyxu.opt.solver.ChambollePock`,
    :py:func:`~pyxu.opt.solver.DouglasRachford`
    """
    kwargs.update(log_var=kwargs.get("log_var", ("x",)))
    obj = base(
        f=None,
        g=g,
        h=None,
        K=None,
        **kwargs,
    )

    obj.__repr__ = lambda _: "ProximalPoint"
    return obj


PP = ProximalPoint  #: Alias of :py:func:`~pyxu.opt.solver.ProximalPoint`.
