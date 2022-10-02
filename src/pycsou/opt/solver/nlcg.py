import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.math.linesearch as ls
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct

__all__ = [
    "NLCG",
]


class NLCG(pyca.Solver):
    r"""
    Nonlinear Conjugate Gradient Method.

    The Nonlinear Conjugate Gradient method finds local minima of the problem

    .. math::

       \min_{x\in\mathbb{R}^{N}} f(x),

    where :math:`f: \mathbb{R}^{N} \to \mathbb{R}` is a *differentiable* functional.

    The norm of the `gradient <https://www.wikiwand.com/en/Nonlinear_conjugate_gradient_method>`_
    :math:`\nabla f_k = \nabla f(x_k)` is used as the default stopping criterion.
    This provides a guaranteed level of accuracy both in exact arithmetic and in the presence of
    round-off errors.
    By default, the iterations stop when the norm of the explicit residual is smaller than 1e-4.

    Multiple variants of NLCG exist, that essentially only differ in the weighing of conjugate
    directions. Two of the most popular variants are offered:

    * The Fletcher-Reeves variant:

    .. math::

       \beta_k^\text{FR} = \frac{\Vert{\nabla f_{k+1}}\Vert_2^2}{\Vert{\nabla f_{k}}\Vert_2^2}

    * The Polak-Ribière+ method:

    .. math::

       \beta_k^\text{PR} = \frac{\nabla f_{k+1}^T\left(\nabla f_{k+1} - \nabla f_k\right)}{\Vert{\nabla f_{k}}\Vert_2^2}
       \beta_k^\text{PR+} = \max\left(0, \beta_k^\text{PR}\right)

    ``NLCG.fit()`` **Parameterization**

    x0: pyct.NDArray
       (..., N) initial point(s).
    variant: str
       Name of the used variant for NLCG. Use "PR" for the Polak-Ribière+ variant, and "FR" for the
       Fletcher-Reeves variant.
    restart_rate: pyct.Integer
       Number of iterations after which restart is applied.
       By default, restart is done after 'n' iterations, where 'n' corresponds to the dimension of
       the inputs of :math:`f`.
    a_bar: pyct.Real
       Line search optional argument, see: :py:`~pycsou.math.linesearch`.
    r: pyct.Real
       Line search optional argument, see: :py:`~pycsou.math.linesearch`.
    c: pyct.Real
       Line search optional argument, see: :py:`~pycsou.math.linesearch`.

    """

    def __init__(self, f: pyca.DiffFunc, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f
        self._ls_a_bar = None
        self._ls_r = None
        self._ls_c = None
        self._variant = "xx"

    @pycrt.enforce_precision(i=("x0", "a_bar", "r", "c"))
    def m_init(
        self,
        x0: pyct.NDArray,
        variant: str,
        restart_rate: pyct.Integer = None,
        a_bar: pyct.Real = ls.LINESEARCH_DEFAULT_A_BAR,
        r: pyct.Real = ls.LINESEARCH_DEFAULT_R,
        c: pyct.Real = ls.LINESEARCH_DEFAULT_C,
    ):
        mst = self._mstate  # shorthand
        self._variant = self.__parse_variant(variant)
        self._ls_a_bar = a_bar
        self._ls_r = r
        self._ls_c = c

        if restart_rate is not None:
            assert restart_rate >= 1
            mst["restart_rate"] = int(restart_rate)
        else:
            mst["restart_rate"] = x0.shape[0]

        mst["x"] = x0
        mst["gradient"] = self._f.grad(x0)
        mst["conjugate_dir"] = -mst["gradient"].copy()
        mst["linesearch_a_bar"] = a_bar
        mst["linesearch_r"] = r
        mst["linesearch_c"] = c
        mst["linesearch_a_k"] = a_bar

    def m_step(self):

        mst = self._mstate  # shorthand
        x_k, g_f_k, p_k = mst["x"], mst["gradient"], mst["conjugate_dir"]

        a_k = ls.backtracking_linesearch(
            f=self._f, x=x_k, g_f_x=g_f_k, p=p_k, a_bar=self._ls_a_bar, r=self._ls_r, c=self._ls_c
        )

        x_kp1 = x_k + a_k * p_k

        g_f_kp1 = self._f.grad(x_kp1)

        # Because NLCG can only generate n conjugate vectors in an n-dimensional space, it makes sense
        # to restart NLCG every n iterations.
        if self._astate["idx"] % mst["restart_rate"] == 0:  # explicit eval
            beta_kp1 = 0.0
        else:
            beta_kp1 = self.__compute_beta(g_f_k, g_f_kp1)
        p_kp1 = -g_f_kp1 + beta_kp1 * p_k

        mst["x"], mst["gradient"], mst["conjugate_dir"], mst["linesearch_a_k"] = x_kp1, g_f_kp1, p_kp1, a_k

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

    def __compute_beta(self, g_f_k: pyct.NDArray, g_f_kp1: pyct.NDArray) -> pyct.Real:
        if self._variant == 0:
            return (pylinalg.norm(g_f_kp1) / pylinalg.norm(g_f_k)) ** 2
        return max((g_f_kp1 @ (g_f_kp1 - g_f_k)) / (pylinalg.norm(g_f_k) ** 2), 0.0)

    def __parse_variant(self, variant: str):
        variant = variant.lower()
        FR_indicator = variant == "FR" or "fletcher" in variant or "reeves" in variant
        PR_indicator = variant == "PR" or "polak" in variant or "ribi" in variant

        if FR_indicator and PR_indicator:
            raise ValueError("The variant was ambiguously specified.")
        elif FR_indicator:
            self._variant = 0
        elif PR_indicator:
            self._variant = 1
        else:
            raise ValueError("The NLCG variant was incorrectly specified.")
