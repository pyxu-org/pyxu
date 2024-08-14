import itertools
import math
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator as pxo
import pyxu.util as pxu

__all__ = [
    "PGD",
]


class PGD(pxa.Solver):
    r"""
    Proximal Gradient Descent (PGD) solver.

    PGD solves minimization problems of the form

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^{M_{1} \times\cdots\times M_{D}}} \;
       \mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^{M_{1} \times\cdots\times M_{D}}\rightarrow \mathbb{R}` is *convex* and
      *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^{M_{1} \times\cdots\times M_{D}}\rightarrow \mathbb{R}\cup\{+\infty\}` is a
      *proper*, *lower semicontinuous* and *convex function* with a *simple proximal operator*.

    Remarks
    -------
    * The problem is *feasible* -- i.e. there exists at least one solution.

    * The algorithm is still valid if either :math:`\mathcal{F}` or :math:`\mathcal{G}` is zero.

    * The convergence is guaranteed for step sizes :math:`\tau\leq 1/\beta`.

    * Various acceleration schemes are described in [APGD]_.  PGD achieves the following (optimal) *convergence rate*
      with the implemented acceleration scheme from Chambolle & Dossal:

      .. math::

         \lim\limits_{n\rightarrow \infty} n^2\left\vert \mathcal{J}(\mathbf{x}^\star)- \mathcal{J}(\mathbf{x}_n)\right\vert=0
         \qquad\&\qquad
         \lim\limits_{n\rightarrow \infty} n^2\Vert \mathbf{x}_n-\mathbf{x}_{n-1}\Vert^2_\mathcal{X}=0,

      for *some minimiser* :math:`{\mathbf{x}^\star}\in\arg\min_{\mathbf{x}\in\mathbb{R}^{M_{1} \times\cdots\times
      M_{D}}} \;\left\{\mathcal{J}(\mathbf{x}):=\mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})\right\}`.  In other
      words, both the objective functional and the PGD iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` converge at a
      rate :math:`o(1/n^2)`.  Significant practical *speedup* can be achieved for values of :math:`d` in the range
      :math:`[50,100]` [APGD]_.

    * The relative norm change of the primal variable is used as the default stopping criterion.  By default, the
      algorithm stops when the norm of the difference between two consecutive PGD iterates
      :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` is smaller than 1e-4.  Different stopping criteria can be used.

    Parameters (``__init__()``)
    ---------------------------
    * **f** (:py:class:`~pyxu.abc.DiffFunc`, :py:obj:`None`)
      --
      Differentiable function :math:`\mathcal{F}`.
    * **g** (:py:class:`~pyxu.abc.ProxFunc`, :py:obj:`None`)
      --
      Proximable function :math:`\mathcal{G}`.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.__init__`.

    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
      --
      (..., M1,...,MD) initial point(s).
    * **tau** (:py:attr:`~pyxu.info.ptype.Real`, :py:obj:`None`)
      --
      Gradient step size.  Defaults to :math:`1 / \beta` if unspecified.
    * **acceleration** (:py:obj:`bool`)
      --
      If True (default), then use Chambolle & Dossal acceleration scheme.
    * **d** (:py:attr:`~pyxu.info.ptype.Real`)
      --
      Chambolle & Dossal acceleration parameter :math:`d`.  Should be greater than 2.  Only meaningful if `acceleration`
      is True.  Defaults to 75 in unspecified.
    * **\*\*kwargs** (:py:class:`~collections.abc.Mapping`)
      --
      Other keyword parameters passed on to :py:meth:`pyxu.abc.Solver.fit`.
    """

    def __init__(
        self,
        f: pxa.DiffFunc = None,
        g: pxa.ProxFunc = None,
        **kwargs,
    ):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        if (f is None) and (g is None):
            msg = " ".join(
                [
                    "Cannot minimize always-0 functional.",
                    "At least one of Parameter[f, g] must be specified.",
                ]
            )
            raise NotImplementedError(msg)
        elif f is None:
            self._f = pxo.NullFunc(dim_shape=g.dim_shape)
            self._g = g
        elif g is None:
            self._f = f
            self._g = pxo.NullFunc(dim_shape=f.dim_shape)
        else:
            self._f = f
            self._g = g

    def m_init(
        self,
        x0: pxt.NDArray,
        tau: pxt.Real = None,
        acceleration: bool = True,
        d: pxt.Real = 75,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0

        if tau is None:
            mst["tau"] = 1 / np.array(self._f.diff_lipschitz)
            if math.isclose(mst["tau"], 0):
                # _f does not provide any "useful" diff_lipschitz constant.
                msg = "\n".join(
                    [
                        "No useful step size could be auto-determined from Parameter[f].",
                        "Consider initializing Parameter[tau] directly, or set (an estimate of) the diff-Lipschitz constant of Parameter[f] before calling fit().",
                        "Solver iterations as-is may stagnate.",
                    ]
                )
                warnings.warn(msg, pxw.AutoInferenceWarning)
            if math.isinf(mst["tau"]):
                # _f is constant-valued: \tau is a free parameter.
                mst["tau"] = 1
                msg = "\n".join(
                    [
                        rf"The gradient/proximal step size \tau is auto-set to {mst['tau']}.",
                        r"Choosing \tau manually may lead to faster convergence.",
                    ]
                )
                warnings.warn(msg, pxw.AutoInferenceWarning)
        else:
            try:
                assert tau > 0
                mst["tau"] = tau
            except Exception:
                raise ValueError(f"tau must be positive, got {tau}.")

        if acceleration:
            try:
                assert d > 2
                mst["a"] = (k / (k + 1 + d) for k in itertools.count(start=0))
            except Exception:
                raise ValueError(f"Expected d > 2, got {d}.")
        else:
            mst["a"] = itertools.repeat(0.0)

    def m_step(self):
        mst = self._mstate  # shorthand
        a = next(mst["a"])

        # In-place implementation of -----------------
        #   y = (1 + a) * mst["x"] - a * mst["x_prev"]
        y = mst["x"] - mst["x_prev"]
        y *= a
        y += mst["x"]
        # --------------------------------------------

        # In-place implementation of -----------------
        #   z = y - mst["tau"] * self._f.grad(y)
        z = pxu.copy_if_unsafe(self._f.grad(y))
        z *= -mst["tau"]
        z += y
        # --------------------------------------------

        mst["x_prev"], mst["x"] = mst["x"], self._g.prox(z, mst["tau"])

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import RelError

        stop_crit = RelError(
            eps=1e-4,
            var="x",
            rank=self._f.dim_rank,
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pxt.NDArray:
        func = lambda x: self._f.apply(x) + self._g.apply(x)

        y = func(self._mstate["x"])
        return y

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., M1,...,MD) solution.
        """
        data, _ = self.stats()
        return data.get("x")
