import numpy as np

import pycsou.abc as pyca
import pycsou.info.ptype as pyct
import pycsou.math.linalg as pylinalg
import pycsou.math.linesearch as ls
import pycsou.runtime as pycrt

__all__ = [
    "NLCG",
]


class NLCG(pyca.Solver):
    r"""
    Nonlinear Conjugate Gradient Method (NLCG).

    The Nonlinear Conjugate Gradient method finds a local minimum of the problem

    .. math::

       \min_{x\in\mathbb{R}^{N}} f(x),

    where :math:`f: \mathbb{R}^{N} \to \mathbb{R}` is a *differentiable* functional.
    When :math:`f` is quadratic, NLCG is equivalent to the Conjugate Gradient (CG) method.
    NLCG hence has similar convergence behaviour to CG if :math:`f` is locally-quadratic.
    The converge speed may be slower however due to its line-search overhead [NumOpt_NocWri]_.

    The norm of the `gradient <https://www.wikiwand.com/en/Nonlinear_conjugate_gradient_method>`_
    :math:`\nabla f_k = \nabla f(x_k)` is used as the default stopping criterion.
    By default, the iterations stop when the norm of the gradient is smaller than 1e-4.

    Multiple variants of NLCG exist.
    They differ mainly in how the weights applied to conjugate directions are updated.
    Two popular variants are implemented:

    * The Fletcher-Reeves variant:

    .. math::

       \beta_k^\text{FR}
       =
       \frac{
         \Vert{\nabla f_{k+1}}\Vert_2^2
       }{
         \Vert{\nabla f_{k}}\Vert_2^2
       }

    * The Polak-Ribière+ method:

    .. math::

       \beta_k^\text{PR}
       =
       \frac{
         \nabla f_{k+1}^T\left(\nabla f_{k+1} - \nabla f_k\right)
       }{
         \Vert{\nabla f_{k}}\Vert_2^2
       } \\
       \beta_k^\text{PR+}
       =
       \max\left(0, \beta_k^\text{PR}\right)

    ``NLCG.fit()`` **Parameterization**

    x0: pyct.NDArray
        (..., N) initial point(s).
    variant: str
        Name of the NLCG variant to use:

        * "PR" for the Polak-Ribière+ variant (default).
        * "FR" for the Fletcher-Reeves variant.
    restart_rate: pyct.Integer
        Number of iterations after which restart is applied.

        By default, restart is done after :math:`N` iterations.
    **kwargs
        Optional parameters forwarded to :py:func:`~pycsou.math.linesearch.backtracking_linesearch`.
        (See: :py:mod:`~pycsou.math.linesearch`.)

        If `a0` is unspecified and :math:`\nabla f` is :math:`\beta`-Lipschitz continuous, then `a0`
        is auto-chosen as :math:`\beta^{-1}`.
        Users are expected to set `a0` if its value cannot be auto-inferred.

    Example
    --------
    Consider the following quadratic optimization problem:

    .. math:

       \min_{\mathbf{x}} \Vert{A\mathbf{x}-\mathbf{b}}\Vert_2^2


    This problem is strictly convex, hence NLCG will converge to the optimal solution:

    .. code-block:: python3

       import numpy as np

       import pycsou.operator as pyco
       import pycsou.opt.solver as pycs

       N, a, b = 5, 3, 1
       f = pyco.SquaredL2Norm(N).asloss(b).argscale(a)  # \norm(Ax - b)**2

       nlcg = pycs.NLCG(f)
       nlcg.fit(x0=np.zeros((N,)), variant="FR")
       x_opt = nlcg.solution()
       np.allclose(x_opt, 1/a)  # True

    Note however that the CG method is preferable in this context since it omits the linesearch
    overhead. The former depends on the cost of applying :math:`A`, and may be significant.
    """

    def __init__(self, f: pyca.DiffFunc, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f

    @pycrt.enforce_precision(i="x0")
    def m_init(
        self,
        x0: pyct.NDArray,
        variant: str = "PR",
        restart_rate: pyct.Integer = None,
        **kwargs,
    ):
        mst = self._mstate  # shorthand

        if (a0 := kwargs.get("a0")) is None:
            d_l = self._f.diff_lipschitz()
            if np.isclose(d_l, np.inf) or np.isclose(d_l, 0):
                msg = "[NLCG] cannot auto-infer initial step size: specify `a0` manually in NLCG.fit()"
                raise ValueError(msg)
            else:
                a0 = pycrt.coerce(1 / d_l)

        if restart_rate is not None:
            assert restart_rate >= 1
            mst["restart_rate"] = int(restart_rate)
        else:
            mst["restart_rate"] = x0.shape[-1]

        mst["x"] = x0
        mst["gradient"] = self._f.grad(x0)
        mst["conjugate_dir"] = -mst["gradient"].copy()
        mst["variant"] = self.__parse_variant(variant)
        mst["ls_a0"] = a0
        mst["ls_r"] = kwargs.get("r", ls.LINESEARCH_DEFAULT_R)
        mst["ls_c"] = kwargs.get("c", ls.LINESEARCH_DEFAULT_C)
        mst["ls_a_k"] = mst["ls_a0"]

    def m_step(self):
        mst = self._mstate  # shorthand
        x_k, g_k, p_k = mst["x"], mst["gradient"], mst["conjugate_dir"]

        a_k = ls.backtracking_linesearch(
            f=self._f,
            x=x_k,
            gradient=g_k,
            direction=p_k,
            a0=mst["ls_a0"],
            r=mst["ls_r"],
            c=mst["ls_c"],
        )
        # In-place implementation of -----------------
        #   x_kp1 = x_k + p_k * a_k
        x_kp1 = p_k.copy()
        x_kp1 *= a_k
        x_kp1 += x_k
        # --------------------------------------------
        g_kp1 = self._f.grad(x_kp1)

        # Because NLCG can only generate n conjugate vectors in an n-dimensional space, it makes sense
        # to restart NLCG every n iterations.
        if self._astate["idx"] % mst["restart_rate"] == 0:
            beta_kp1 = pycrt.coerce(0)
        else:
            beta_kp1 = self.__compute_beta(g_k, g_kp1)

        # In-place implementation of -----------------
        #   p_kp1 = -g_kp1 + beta_kp1 * p_k
        p_kp1 = p_k.copy()
        p_kp1 *= beta_kp1
        p_kp1 -= g_kp1
        # --------------------------------------------

        mst["x"], mst["gradient"], mst["conjugate_dir"], mst["ls_a_k"] = x_kp1, g_kp1, p_kp1, a_k

    def default_stop_crit(self) -> pyca.StoppingCriterion:
        from pycsou.opt.stop import AbsError

        stop_crit = AbsError(
            eps=1e-4,
            var="gradient",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pyct.NDArray:
        return self._f(self._mstate["x"])

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: pyct.NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")

    def __compute_beta(self, g_k: pyct.NDArray, g_kp1: pyct.NDArray) -> pyct.NDArray:
        v = self._mstate["variant"]
        if v == "fr":  # Fletcher-Reeves
            gn_k = pylinalg.norm(g_k, axis=-1, keepdims=True)
            gn_kp1 = pylinalg.norm(g_kp1, axis=-1, keepdims=True)
            beta = (gn_kp1 / gn_k) ** 2
        elif v == "pr":  # Poliak-Ribière+
            gn_k = pylinalg.norm(g_k, axis=-1, keepdims=True)
            numerator = (g_kp1 * (g_kp1 - g_k)).sum(axis=-1, keepdims=True)
            beta = numerator / (gn_k**2)
            beta = beta.clip(min=0)
        return beta  # (..., 1)

    def __parse_variant(self, variant: str) -> str:
        supported_variants = {"fr", "pr"}
        if (v := variant.lower().strip()) not in supported_variants:
            raise ValueError(f"Unsupported variant '{variant}'.")
        return v
