import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.math.linesearch as ls
import pycsou.runtime as pycrt
import pycsou.util as pycu
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
        Name of the NLCG variant to use.
        Use "PR" for the Polak-Ribière+ variant.
        Use "FR" for the Fletcher-Reeves variant.
    restart_rate: pyct.Integer
        Number of iterations after which restart is applied.
        By default, restart is done after 'n' iterations, where 'n' corresponds to the dimension of
        the inputs to :math:`f`.
    a_bar, r, c: pyct.Real
        Optional line search parameters. (See: :py:mod:`~pycsou.math.linesearch`.)
    """

    def __init__(self, f: pyca.DiffFunc, **kwargs):
        kwargs.update(
            log_var=kwargs.get("log_var", ("x",)),
        )
        super().__init__(**kwargs)

        self._f = f

    @pycrt.enforce_precision(i=("x0", "a_bar", "r", "c"))
    def m_init(
        self,
        x0: pyct.NDArray,
        variant: str,
        restart_rate: pyct.Integer = None,
        a_bar: pyct.Real = None,
        r: pyct.Real = None,
        c: pyct.Real = None,
    ):
        mst = self._mstate  # shorthand

        if restart_rate is not None:
            assert restart_rate >= 1
            mst["restart_rate"] = int(restart_rate)
        else:
            mst["restart_rate"] = x0.shape[-1]

        sanitize = lambda _, default: _ if (_ is not None) else default

        mst["x"] = x0 if len(x0.shape) > 1 else x0.reshape(1, x0.shape[0])
        mst["gradient"] = self._f.grad(x0)
        mst["conjugate_dir"] = -mst["gradient"].copy()
        mst["variant"] = self.__parse_variant(variant)
        mst["ls_a_bar"] = sanitize(a_bar, ls.LINESEARCH_DEFAULT_A_BAR)
        mst["ls_r"] = sanitize(r, ls.LINESEARCH_DEFAULT_R)
        mst["ls_c"] = sanitize(c, ls.LINESEARCH_DEFAULT_C)
        mst["ls_a_k"] = mst["ls_a_bar"]

    def m_step(self):

        mst = self._mstate  # shorthand
        x_k, g_f_k, p_k = mst["x"], mst["gradient"], mst["conjugate_dir"]

        a_k = ls.backtracking_linesearch(
            f=self._f, x=x_k, g_f_x=g_f_k, p=p_k, a_bar=mst["ls_a_bar"], r=mst["ls_r"], c=mst["ls_c"]
        )

        x_kp1 = x_k + p_k * a_k[:, None]
        g_f_kp1 = self._f.grad(x_kp1)

        # Because NLCG can only generate n conjugate vectors in an n-dimensional space, it makes sense
        # to restart NLCG every n iterations.
        if self._astate["idx"] % mst["restart_rate"] == 0:
            xp = pycu.get_array_module(x_k)
            beta_kp1 = xp.full((1, x_k.shape[-1]), 0.0, dtype=x_k.dtype)
        else:
            beta_kp1 = self.__compute_beta(g_f_k, g_f_kp1)
        p_kp1 = -g_f_kp1 + beta_kp1 * p_k

        mst["x"], mst["gradient"], mst["conjugate_dir"], mst["ls_a_k"] = x_kp1, g_f_kp1, p_kp1, a_k

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
        pycu.compute()
        data, _ = self.stats()
        return data.get("x")

    def __compute_beta(self, g_f_k: pyct.NDArray, g_f_kp1: pyct.NDArray) -> pyct.Real:
        if self._variant == 0:
            return (pylinalg.norm(g_f_kp1, keepdims=True) / pylinalg.norm(g_f_k, keepdims=True)) ** 2
        temp = (g_f_kp1 * (g_f_kp1 - g_f_k)).sum(axis=-1) / (pylinalg.norm(g_f_k, keepdims=True) ** 2)
        xp = pycu.get_array_module(temp)
        return xp.where(temp > 0, temp, 0).T

    def __parse_variant(self, variant: str):
        variant = variant.lower()
        FR_indicator = variant == "fr"
        PR_indicator = variant == "pr"

        if FR_indicator:
            return 0
        elif PR_indicator:
            return 1
        else:
            raise ValueError("The NLCG variant was incorrectly specified.")
