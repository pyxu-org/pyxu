import math

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.util as pxu

__all__ = [
    "Adam",
]


class Adam(pxa.Solver):
    r"""
    Adam solver [ProxAdam]_.

    Adam minimizes

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    Adam is a suitable alternative to Proximal Gradient Descent (:py:class:`~pyxu.opt.solver.PGD`) when:

    * the cost function is differentiable,
    * computing :math:`\beta` to optimally choose the step size is infeasible,
    * line-search methods to estimate step sizes are too expensive.

    Compared to PGD, Adam auto-tunes gradient updates based on stochastic estimates of :math:`\phi_{t} =
    \mathbb{E}[\nabla\mathcal{F}]` and :math:`\psi_{t} = \mathbb{E}[\nabla\mathcal{F}^{2}]` respectively.

    Adam has many named variants for particular choices of :math:`\phi` and :math:`\psi`:

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
         } + \epsilon,

    * AMSGrad:

      .. math::

         \phi_t = \mathbf{m}_t
         \qquad
         \psi_t = \sqrt{\hat{\mathbf{v}}_t},

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

    Remarks
    -------
    * The convergence is guaranteed for step sizes :math:`\alpha\leq 2/\beta`.

    * The default stopping criterion is the relative norm change of the primal variable.  By default, the algorithm
      stops when the norm of the difference between two consecutive iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}`
      is smaller than 1e-4.  Different stopping criteria can be used.  It is recommended to change the stopping
      criterion when using the PAdam and AMSGrad variants to avoid premature stops.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., N) initial point(s).
    * **variant** ("adam", "amsgrad", "padam")
      --
      Name of the Adam variant to use.  Defaults to "adam".
    * **a** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Max normalized gradient step size.  Defaults to :math:`1 / \beta` if unspecified.
    * **b1** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      1st-order gradient exponential decay :math:`\beta_{1} \in [0, 1)`.
    * **b2** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      2nd-order gradient exponential decay :math:`\beta_{2} \in [0, 1)`.
    * **m0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., N) initial 1st-order gradient estimate corresponding to each initial point.  Defaults to the null vector if
      unspecified.
    * **v0** (:py:attr:`~pyxu.info.ptype.NDArray`, :py:obj:`None`)
      --
      (..., N) initial 2nd-order gradient estimate corresponding to each initial point.  Defaults to the null vector if
      unspecified.
    * **p** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      PAdam power parameter :math:`p \in (0, 0.5]`.  Must be specified for PAdam, unused otherwise.
    * **eps_adam** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      Adam noise parameter :math:`\epsilon`.  This term is used exclusively if `variant="adam"`.  Defaults to 1e-6.
    * **eps_var** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      Avoids division by zero if estimated gradient variance is too small.  Defaults to 1e-6.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.

    Note
    ----
    If provided, `m0` and `v0` must be broadcastable with `x0`.

    Example
    -------
    Consider the following optimization problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N} \Vert{\mathbf{x}-\mathbf{1}}\Vert_2^2

    .. code-block:: python3

       import numpy as np

       from pyxu.operator import SquaredL2Norm
       from pyxu.opt.solver import Adam

       N = 3
       f = SquaredL2Norm(dim=N).asloss(1)

       slvr = Adam(f)
       slvr.fit(
           x0=np.zeros((N,)),
           variant="padam",
           p=0.25,
       )
       x_opt = slvr.solution()
       np.allclose(x_opt, 1, rtol=1e-4)  # True
    """

    def __init__(
        self,
        f: pxa.DiffFunc,
        **kwargs,
    ):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f

    def m_init(  # default values from https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.py
        self,
        x0: pxt.NDArray,
        variant: str = "adam",
        a: pxt.Real = None,
        b1: pxt.Real = 0.9,
        b2: pxt.Real = 0.999,
        m0: pxt.NDArray = None,
        v0: pxt.NDArray = None,
        p: pxt.Real = 0.5,
        eps_adam: pxt.Real = 1e-6,
        eps_var: pxt.Real = 1e-6,
    ):
        mst = self._mstate  # shorthand
        xp = pxu.get_array_module(x0)
        mst["x"] = x0

        mst["variant"] = self.__parse_variant(variant)

        if a is None:
            g = lambda _: math.isclose(self._f.diff_lipschitz, _)
            if g(0) or g(math.inf):
                error_msg = "Cannot auto-infer step size: choose it manually."
                raise pxw.AutoInferenceWarning(error_msg)
            else:
                mst["a"] = 1.0 / self._f.diff_lipschitz
        else:
            assert a > 0, f"Parameter[a] must be positive, got {a}."
            mst["a"] = a

        assert 0 <= b1 < 1, f"Parameter[b1]: expected value in [0, 1), got {b1}."
        mst["b1"] = b1

        assert 0 <= b2 < 1, f"Parameter[b2]: expected value in [0, 1), got {b2}."
        mst["b2"] = b2

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

        assert 0 < p <= 0.5, f"Parameter[p]: expected value in (0, 0.5], got {p}."
        mst["padam_p"] = p

        assert eps_adam > 0, f"Parameter[eps_adam]: expected positive value, got {eps_adam}."
        mst["eps_adam"] = eps_adam

        assert eps_var > 0, f"Parameter[eps_var]: expected positive value, got {eps_var}."
        mst["eps_variance"] = eps_var

    def m_step(self):
        mst = self._mstate  # shorthand

        x, a = mst["x"], mst["a"]
        xp = pxu.get_array_module(x)
        gm = pxu.copy_if_unsafe(self._f.grad(x))
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
        mst["x"] = x - phi

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import RelError

        # Described in [ProxAdam]_ and used in their implementation:
        # https://github.com/pmelchior/proxmin/blob/master/proxmin/algorithms.py
        rel_error = RelError(
            eps=1e-4,
            var="x",
            rank=self._f.dim_rank,
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return rel_error

    def objective_func(self) -> pxt.NDArray:
        func = lambda x: self._f.apply(x)
        y = func(self._mstate["x"])
        return y

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: NDArray
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
        xp = pxu.get_array_module(mst["x"])
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
