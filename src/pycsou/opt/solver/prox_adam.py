import warnings

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "ProxAdam",
]


class ProxAdam(pyca.Solver):
    r"""
    Proximal Adam solver.

    ProxAdam minimizes

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower
      semicontinuous* and *convex function* with a *simple proximal operator*.

    It does so by estimating the mean and the variance of :math:`\mathbf{g}_t=\nabla\mathcal{F}` at each
    iteration by :math:`\phi_t=\phi(\mathbf{g}_1, \ldots, \mathbf{g}_t)` and
    :math:`\psi_t=\psi(\mathbf{g}_1^2, \ldots, \mathbf{g}_t^2)` respectively first, then solving the proximal
    sub-problem:

    ..math::

       {\min_{\mathbf{z}\in\mathbb{R}^N} \;\mathcal{G}(\mathbf{z})\;\;+\;\;\frac{1}{2\alpha}||\mathbf{z}-\mathbf{x}_t||_H^2},

    where :math:`H=\text{diag}(\psi_t)`, which can be solved through Proximal Gradient Descent.
    Three variants of this algorithm are implemented, corresponding to three distinct :math:`\phi` and :math:`psi` pairs:

    * Adam:

    ..math::

       \phi_t
       =
       \frac{
         \mathbf{m}_t
       }{
         1-\beta_1^t
       } \\
       \psi_t
       =
       \sqrt{
         \frac{
           \mathbf{v}_t
         }{
           1-\beta_2^t
         }
       } + \epsilon

    * AMSGrad:

    ..math::

       \phi_t = \mathbf{m}_t \\
       \psi_t = \sqrt{\hat{\mathbf{v}}_t}

    * PAdam:

    ..math::

       \phi_t = \mathbf{m}_t \\
       \psi_t = \hat{\mathbf{v}}_t^p

    where in all cases:

    ..math::

       \mathbf{m}_t
       =
       \beta_1\mathbf{m}_{t-1}
       +
       (1-\beta_1)\mathbf{g}_t \\
       \mathbf{v}_t
       =
       \beta_2\mathbf{v}_{t-1}
       +
       (1-\beta_2)\mathbf{g}_t^2\\
       \hat{\mathbf{v}}_t
       =
       \max(\hat{\mathbf{v}}_{t-1}, \mathbf{v}_t)

    with :math:`\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}`.

    Further details in [ProxAdam]_.

    **Remark 1:**
    The algorithm is still valid if :math:`\mathcal{G}` is zero. It is not if :math:`\mathcal{F}` is zero.

    **Remark 2:**
    The convergence is guaranteed for step sizes :math:`\alpha\leq 2/\beta`.

    **Remark 3:**
    The relative norm change of the primal variable is used as the default stopping criterion.
    By default, the algorithm stops when the norm of the difference between two consecutive
    iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` is smaller than 1e-4.
    Different stopping criteria can be used. (see :py:mod:`~pycsou.opt.solver.stop`.)
    By default, the same stopping criterion is used for the proximal sub-problem.

    ``ProxAdam.fit()`` **Parameterization**

    x0: pyct.NDArray
        (..., N) initial point(s).
    variant: str
        Name of the ProxAdam variant to use.
        Use "adam" for Adam. (default)
        Use "amsgrad" for AMSGrad.
        Use "padam" for PAdam.
    a: pyct.Real
        Gradient step size.
        Defaults to :math:`1 / \beta` if unspecified.
    b1: pyct.Real
        Factor for mean gradient computation.
    b2: pyct.Real
        Factor for gradient variance computation.
    m0: pyct.NDArray
        (..., N) first mean estimate of the gradient corresponding to
        each initial point, defaults to the zero vector.
    v0: pyct.NDArray
        (..., N) first variance estimate of the gradient corresponding to
        each initial point, defaults to the zero vector.
    stop_crit_sub: pyca.solver.StoppingCriterion
        Sub-problem stopping criterion, defaults to the main problem's.
    **kwargs
        p: pyct.Real
            Power used in the variance estimate. Must be specified for PAdam, unused else. Should be between 0 and 0.5.
        eps_adam: pyct.Real
            Noise term used exclusively in Adam, unused else. Defaults to 1e-6.

    Example
    --------
    Consider the following optimisation problem:

    ..math:

       \min_{\mathbf{x}\in\mathbb{R}^N} \Vert{\mathbf{x}-\mathbf{1}}\Vert_2^2 + \Vert{\mathbf{x}-\mathbf{1}}\Vert_1
    
    >>> import numpy as np

    >>> from pycsou.operator import shift_loss
    >>> from pycsou.operator.func import L2Norm, SquaredL2Norm
    >>> from pycsou.opt.solver import ProxAdam

    >>> N = 3
    >>> f = SquaredL2Norm(dim=N).asloss(np.ones((N,)))
    >>> g = L2Norm(dim=N).asloss(np.ones((N,)))

    >>> prox_adam = ProxAdam(f, g)
    >>> x0 = np.zeros((N,))
    >>> prox_adam.fit(x0=x0, variant="padam", p=0.25)
    >>> x_star = prox_adam.solution()
    >>> assert np.allclose(x_star, np.ones((N,)))

    """
    __eps_adam_default = 1e-6

    def __init__(self, f: pyca.DiffFunc, g: pyca.ProxFunc, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f
        self._g = g

    @pycrt.enforce_precision(i=("x0", "a", "b1", "b2", "m0", "v0"))
    def m_init(
        self,
        x0: pyct.NDArray,
        variant: str = "adam",
        a: pyct.Real = None,
        b1: pyct.Real = 0.9,  # default values from:
        b2: pyct.Real = 0.999,  # https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.
        m0: pyct.NDArray = None,  # warm start for mean
        v0: pyct.NDArray = None,  # warm start for variance
        stop_crit_sub: pyca.solver.StoppingCriterion = None,  # sub-problem stopping criterion
        **kwargs,  # p for padam, eps_adam for adam with eps_adam defaulting to some value > 0
    ):
        mst = self._mstate  # shorthand

        mst["x"] = x0

        if a is None:
            try:
                mst["a"] = pycrt.coerce(1 / self._f.diff_lipschitz())
            except ZeroDivisionError as exc:
                # _f is constant-valued: a is a free parameter.
                mst["a"] = 1.0
                msg = "\n".join(
                    [
                        rf"[ProxAdam] The gradient/proximal step size a is auto-set to {mst['a']}.",
                        r"           Choosing a manually may lead to faster convergence.",
                    ]
                )
                warnings.warn(msg, pycuw.AutoInferenceWarning)
        else:
            try:
                assert a > 0
                mst["a"] = a
            except:
                raise ValueError(f"[ProxAdam] a must be positive, got {a}.")

        mst["variant"] = self.__parse_variant(variant)

        if mst["variant"] == "padam":
            if (p := kwargs.get("p")) is None:
                raise ValueError("[ProxAdam] To use PAdam, specify the power p as a kwarg in ProxAdam.fit()")
            else:
                mst["padam_p"] = p

        if mst["variant"] == "adam":
            eps_adam = kwargs.get("eps_adam")
            mst["eps_adam"] = self.__eps_adam_default if eps_adam is None else eps_adam

        xp = pycu.get_array_module(x0)

        if m0 is not None:
            assert m0.shape == x0.shape
            mst["mean"] = m0
        else:
            mst["mean"] = xp.full(shape=x0.shape, fill_value=0.0, dtype=x0.dtype)

        if v0 is not None:
            assert v0.shape == x0.shape
            mst["variance"] = v0
        else:
            mst["variance"] = xp.full(shape=x0.shape, fill_value=0.0, dtype=x0.dtype)

        if stop_crit_sub is None:
            stop_crit_sub = self.default_stop_crit()

        mst["subproblem_stop_crit"] = stop_crit_sub

        mst["b1"] = b1
        mst["b2"] = b2

        mst["variance_hat"] = mst["variance"]

        mst["phi"] = self.__compute__phi(1)
        mst["psi"] = self.__compute__psi(1)

    def m_step(self):
        from pycsou.operator import SquaredL2Norm
        from pycsou.operator.linop import DiagonalOp
        from pycsou.opt.solver import PGD

        mst = self._mstate  # shorthand
        x, m, v = mst["x"], mst["mean"], mst["variance"]

        g = self._f.grad(x)

        b1 = mst["b1"]
        # In-place implementation of -----------------
        #   m = b1 * m + (1 - b1) * g
        m = b1 * m
        m += (1 - b1) * g
        # --------------------------------------------

        b2 = mst["b2"]
        # In-place implementation of -----------------
        #   v = b2 * v + (1 - b2) * (g ** 2)
        v = b2 * v
        v += (1 - b2) * (g**2)
        # --------------------------------------------

        mst["mean"], mst["variance"] = m, v
        xp = pycu.get_array_module(x)
        mst["variance_hat"] = xp.maximum(mst["variance_hat"], v)

        phi = self.__compute__phi(self._astate["idx"])
        psi = self.__compute__psi(self._astate["idx"])

        a = mst["a"]
        x = x - a * (phi / psi)

        xp = pycu.get_array_module(x)
        gamma = pycrt.coerce(a / xp.max(psi))

        sqrt_psi = xp.sqrt(psi)
        h = (0.5 / a) * SquaredL2Norm().asloss((sqrt_psi * x).ravel()) * DiagonalOp(sqrt_psi.ravel())
        pgd_sub = PGD(h, self._g, show_progress=False)
        pgd_sub.fit(x0=x.ravel(), tau=gamma, stop_crit=mst["subproblem_stop_crit"])
        x = pgd_sub.solution().reshape(x.shape)

        mst["x"], mst["phi"], mst["psi"] = x, phi, psi

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        from pycsou.opt.stop import RelError

        # Used in https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.py as well as corresp. paper
        stop_crit = RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pyct.NDArray:
        func = lambda x: self._f.apply(x) + self._g.apply(x)
        y = func(self._mstate["x"])
        return y

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: pyct.NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")

    def __compute__phi(self, t):
        mst = self._mstate
        v = mst["variant"]
        m = mst["mean"]
        if v == "adam":
            return m / (1 - (mst["b1"] ** t))
        elif v in ["amsgrad", "padam"]:
            return m

    def __compute__psi(self, t):
        mst = self._mstate
        xp = pycu.get_array_module(mst["x"])
        v = mst["variant"]
        if v == "adam":
            return xp.sqrt(mst["variance"] / (1 - (mst["b2"] ** t))) + mst["eps_adam"]
        elif v == "amsgrad":
            return xp.sqrt(mst["variance_hat"])
        elif v == "padam":
            return mst["variance_hat"] ** mst["padam_p"]

    def __parse_variant(self, variant: str) -> str:
        supported_variants = {"adam", "amsgrad", "padam"}
        if (v := variant.lower().strip()) not in supported_variants:
            raise ValueError(f"Unsupported variant '{variant}'.")
        return v
