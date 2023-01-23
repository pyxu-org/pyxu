import warnings

import pycsou.abc as pyca
import pycsou.operator.func as pycof
import pycsou.operator.linop as pycl
import pycsou.opt.solver as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "ProxAdam",
]


class ProxAdam(pyca.Solver):
    r"""
    Proximal Adam solver [ProxAdam]_.

    ProxAdam minimizes

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower
      semicontinuous* and *convex function* with a *simple proximal operator*.

    ProxAdam is a suitable alternative to Proximal Gradient Descent
    (:py:class:`~pycsou.opt.solvel.pgd.PGD`) when:

    * computing :math:`\beta` to optimally choose the step size is infeasible, and
    * line-search methods to estimate step sizes are too expensive.

    Compared to PGD, ProxAdam:

    * auto-tunes gradient updates based on stochastic estimates of
      :math:`\phi_{t} = \mathbb{E}[\nabla\mathcal{F}]` and :math:`\psi_{t} =
      \mathbb{E}[\nabla\mathcal{F}^{2}]` respectively;
    * uses a modified proximity operator at each iteration to update coordinates at varying scales:

    .. math::

       \text{prox}_{\alpha\mathcal{G}}(\mathbf{x}_{t})
       =
       \min_{\mathbf{z}\in\mathbb{R}^N} \;\mathcal{G}(\mathbf{z})\;\;+\;\;\frac{1}{2\alpha}||\mathbf{z}-\mathbf{x}_t||_H^2,

    where :math:`H=\text{diag}(\psi_t)`.
    The modified proximity operator is computed by solving a PGD sub-problem.
    ProxAdam has many named variants for particular choices of :math:`\phi` and :math:`\psi`:

    * Adam:

    .. math::

       \phi_t
       =
       \frac{
         \mathbf{m}_t
       }{
         1-\beta_1^t
       }
       \qquad
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

    .. math::

       \phi_t = \mathbf{m}_t
       \qquad
       \psi_t = \sqrt{\hat{\mathbf{v}}_t}

    * PAdam:

    .. math::

       \phi_t = \mathbf{m}_t
       \qquad
       \psi_t = \hat{\mathbf{v}}_t^p,

    where in all cases:

    .. math::

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
       \max(\hat{\mathbf{v}}_{t-1}, \mathbf{v}_t),

    with :math:`\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}`.

    **Remark 1:**
    The algorithm is still valid if :math:`\mathcal{G}` is zero.

    **Remark 2:**
    The convergence is guaranteed for step sizes :math:`\alpha\leq 2/\beta`.

    **Remark 3:**
    The default stopping criterion is the relative norm change of the primal variable.
    By default, the algorithm stops when the norm of the difference between two consecutive iterates
    :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` is smaller than 1e-4.
    Different stopping criteria can be used. (see :py:mod:`~pycsou.opt.solver.stop`.) It is recommended
    to change the stopping criterion when using the PAdam and AMSGrad variants to avoid premature stops.
    By default, the same stopping criterion is used for the proximal sub-problem.

    ``ProxAdam.fit()`` **Parameterization**

    x0: pyct.NDArray
        (..., N) initial point(s).
    variant: "adam", "amsgrad", "padam"
        Name of the ProxAdam variant to use.
        Defaults to "adam"
    a: pyct.Real
        Max normalized gradient step size.
        Defaults to :math:`1 / \beta` if unspecified.
    b1: pyct.Real
        1st-order gradient exponential decay :math:`\beta_{1} \in [0, 1)`.
    b2: pyct.Real
        2nd-order gradient exponential decay :math:`\beta_{2} \in [0, 1)`.
    m0: pyct.NDArray
        (..., N) initial 1st-order gradient estimate corresponding to each initial point.
        Defaults to the null vector if unspecified.
    v0: pyct.NDArray
        (..., N) initial 2nd-order gradient estimate corresponding to each initial point.
        Defaults to the null vector if unspecified.
    kwargs_sub: dict[str]
        Keyword parameters used to initialize :py:meth:`~pycsou.opt.solver.pgd.PGD.__init__` in
        sub-problems. This is an advanced option: use it with care.
    stop_crit_sub: pyca.solver.StoppingCriterion
        Sub-problem stopping criterion.
        Default: use same stopping criterion as main problem.
    p: pyct.Real
        PAdam power parameter :math:`p \in (0, 0.5]`.
        Must be specified for PAdam, unused otherwise.
    eps_adam: pyct.Real
        Adam noise parameter :math:`\epsilon`.
        This term is used exclusively if `variant="adam"`.
        Defaults to 1e-6.
    eps_var: pyct.Real
        Avoids division by zero if estimated gradient variance is too small.
        Defaults to 1e-6.

    **Remark 4:**
    If provided, 'm0' and 'v0' must be broadcastable with 'x0'.

    Example
    --------
    Consider the following optimization problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N} \Vert{\mathbf{x}-\mathbf{1}}\Vert_2^2 + \Vert{\mathbf{x}-\mathbf{1}}\Vert_1

    .. code-block:: python3

       import numpy as np

       from pycsou.operator.func import L1Norm, SquaredL2Norm
       from pycsou.opt.solver import ProxAdam

       N = 3
       f = SquaredL2Norm(dim=N).asloss(1)
       g = L1Norm(dim=N).asloss(1)

       prox_adam = ProxAdam(f, g)
       prox_adam.fit(
           x0=np.zeros((N,)),
           variant="padam",
           p=0.25,
       )
       x_opt = prox_adam.solution()
       np.allclose(x_opt, 1)  # True
    """

    def __init__(
        self,
        f: pyca.DiffFunc,
        g: pyca.ProxFunc = None,
        **kwargs,
    ):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f
        # If f is domain-agnostic and g is unspecified, cannot auto-infer NullFunc dimension.
        # Solution: delay initialization of g to m_init(), where x0's shape can be used.
        self._prox_required = g is not None
        self._g = g

    @pycrt.enforce_precision(i=("x0", "a", "b1", "b2", "m0", "v0", "p", "eps_adam", "eps_var"))
    def m_init(  # default values from https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.py
        self,
        x0: pyct.NDArray,
        variant: str = "adam",
        a: pyct.Real = None,
        b1: pyct.Real = 0.9,
        b2: pyct.Real = 0.999,
        m0: pyct.NDArray = None,
        v0: pyct.NDArray = None,
        kwargs_sub: dict = None,
        stop_crit_sub: pyca.solver.StoppingCriterion = None,
        p: pyct.Real = 0.5,
        eps_adam: pyct.Real = 1e-6,
        eps_var: pyct.Real = 1e-6,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = x0

        if self._g is None:
            self._g = pycof.NullFunc(dim=x0.shape[-1])

        if a is None:
            try:
                mst["a"] = pycrt.coerce(1 / self._f.diff_lipschitz())
            except ZeroDivisionError as exc:
                # _f is constant-valued: a is a free parameter.
                mst["a"] = pycrt.coerce(1)
                msg = "\n".join(
                    [
                        rf"The gradient step size `a` is auto-set to {mst['a']}.",
                        r"Choosing a manually may lead to faster convergence.",
                    ]
                )
                warnings.warn(msg, pycuw.AutoInferenceWarning)
        else:
            try:
                assert a > 0
                mst["a"] = a
            except:
                raise ValueError(f"`a` must be positive, got {a}.")

        mst["variant"] = self.__parse_variant(variant)

        assert 0 < p <= 0.5, f"p: expected value in (0, 0.5], got {p}."
        mst["padam_p"] = p

        assert eps_adam > 0, f"eps_adam: expected positive value, got {eps_adam}."
        mst["eps_adam"] = eps_adam

        xp = pycu.get_array_module(x0)

        if m0 is None:
            mst["mean"] = xp.zeros_like(x0)
        elif m0.shape == x0.shape:
            # No broadcasting involved
            mst["mean"] = m0
        else:
            x0, m0 = xp.broadcast_arrays(x0, m0)
            mst["mean"] = m0.copy()

        if v0 is None:
            mst["variance"] = xp.zeros_like(x0)
        elif v0.shape == x0.shape:
            # No broadcasting involved
            mst["variance"] = v0
        else:
            x0, v0 = xp.broadcast_arrays(x0, v0)
            mst["variance"] = v0.copy()
        mst["variance_hat"] = mst["variance"]

        if kwargs_sub is None:
            kwargs_sub = dict()
        mst["subproblem_init_kwargs"] = kwargs_sub
        if stop_crit_sub is None:
            stop_crit_sub = self.default_stop_crit()
        mst["subproblem_stop_crit"] = stop_crit_sub

        assert 0 <= b1 < 1, f"b1: expected value in [0, 1), got {b1}."
        mst["b1"] = b1

        assert 0 <= b2 < 1, f"b2: expected value in [0, 1), got {b2}."
        mst["b2"] = b2

        assert eps_var > 0, f"eps_var: expected positive value, got {eps_var}."
        mst["eps_variance"] = eps_var

    def m_step(self):
        mst = self._mstate  # shorthand

        x, a = mst["x"], mst["a"]
        xp = pycu.get_array_module(x)
        gm = self._f.grad(x)
        gv = gm.copy()

        ## Part 1: evaluate phi/psi ============================
        m, b1 = mst["mean"], mst["b1"]
        # In-place implementation of -----------------
        #   m = b1 * m + (1 - b1) * g
        m *= b1
        gm *= 1 - b1
        m += gm
        # --------------------------------------------
        mst["mean"] = m

        v, b2 = mst["variance"], mst["b2"]
        # In-place implementation of -----------------
        #   v = b2 * v + (1 - b2) * (g ** 2)
        v *= b2
        gv **= 2
        gv *= 1 - b2
        v += gv
        # --------------------------------------------
        mst["variance"] = v.clip(mst["eps_variance"], None)  # avoid division-by-zero
        mst["variance_hat"] = xp.maximum(mst["variance_hat"], mst["variance"])

        phi = self.__phi(t=self._astate["idx"])
        psi = self.__psi(t=self._astate["idx"])
        ## =====================================================

        ## Part 2: take a step in the gradient's direction =====
        # In-place implementation of -----------------
        #   x = x - a * (phi / psi)
        phi /= psi
        phi *= a
        x -= phi
        ## =====================================================

        ## Part 3: eval PGD sub-problems =======================
        if self._prox_required:
            # Assume N initial points were provided in fit().
            # We need to solve a PGD sub-problem for each of them.
            #
            # Option 1: solve one large PGD(f, g) sub-problem, with
            #    f = hstack(f1, ..., fN)
            #    g = hstack(g, ..., g)
            # Option 2: solve N small PGD(f, g) sub-problems, with
            #    f = fn
            #    g = g
            #
            # The former (1) is more convenient since all sub-problems are solved in parallel, but:
            # * convergence will be slower since diff-Lipschitz constant of hstack(f1, ..., fN) is
            #   larger than those of fn. (Moreover the diff-Lipschitz constants of the fn may be
            #   spread out.)
            # * in the event `stop_crit_sub` is provided, we cannot always intercept/adapt it to
            #   apply block-wise on `_mstate`.
            #
            # Each sub-problem in the latter (2) converges fast, but sub-problems may execute in
            # sequence depending on the array backend used.
            #
            # In light of the above, we opt for Option 2.
            *x_sh, N = x.shape
            x_sub = []  # sub-problem solutions
            scale = pycrt.coerce(0.5 / a)
            for _x, _psi in zip(x.reshape(-1, N), psi.reshape(-1, N)):
                gamma = pycrt.coerce(a / xp.max(_psi))
                h_half = pycl.DiagonalOp(xp.sqrt(_psi))

                f1 = pycof.SquaredL2Norm().asloss(h_half.apply(_x)) * h_half
                f = f1 * scale

                kwargs = mst["subproblem_init_kwargs"].copy()
                kwargs.update(show_progress=False)
                slvr = pycos.PGD(f=f, g=self._g, **kwargs)

                slvr.fit(
                    x0=_x,
                    tau=gamma,
                    stop_crit=mst["subproblem_stop_crit"],
                )
                x_sub.append(slvr.solution())
            x = xp.stack(x_sub, axis=0).reshape(*x_sh, N)  # (..., N) original shape
        ## =====================================================

        mst["x"] = x

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        from pycsou.opt.stop import RelError

        # Described in [ProxAdam]_ and used in their implementation:
        # https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.py
        rel_error = RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return rel_error

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

    def __phi(self, t: int):
        mst = self._mstate
        var = mst["variant"]
        if var == "adam":
            out = mst["mean"].copy()
            out /= 1 - (mst["b1"] ** t)
        elif var in ["amsgrad", "padam"]:
            out = mst["mean"].copy()  # to allow in-place updates outside __compute_phi()
        return out

    def __psi(self, t: int):
        mst = self._mstate
        xp = pycu.get_array_module(mst["x"])
        var = mst["variant"]
        if var == "adam":
            out = xp.sqrt(mst["variance"])
            out /= xp.sqrt(1 - mst["b2"] ** t)
            out += mst["eps_adam"]
        elif var == "amsgrad":
            out = xp.sqrt(mst["variance_hat"])
        elif var == "padam":
            out = mst["variance_hat"] ** mst["padam_p"]
        return out

    def __parse_variant(self, variant: str) -> str:
        supported_variants = {"adam", "amsgrad", "padam"}
        if (v := variant.lower().strip()) not in supported_variants:
            raise ValueError(f"Unsupported variant '{variant}'.")
        return v
