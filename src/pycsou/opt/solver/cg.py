import types
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.linop.base as pyclo
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class CG(pycs.Solver):
    r"""
    Conjugate Gradient Method.
    The Conjugate Gradient method solves the system of linear equations of the form
    .. math::
       {\mathbf{A}\mathbf{x} = \mathbf{b}},
    for the vector :math:`\mathcal{x}: \mathbb{x}^N`.
    where:
    * :math:`\mathcal{A}: \mathbb{R}^M\times^N` is a *symmetric* *positive definite* matrix.
    * :math:`\mathcal{b}: \mathbb{b}^M`.
    """

    def __init__(
        self,
        a: typ.Optional[pyco.PosDefOp] = None,
        b: pyct.NDArray = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

        self._a = pyclo.NullOp() if (a is None) else a
        self._b = b if (b is None) else pycrt.coerce(b)

        if (a is None) or (b is None):
            msg = " ".join(
                [
                    "Both the linear operator `a` and the data array `b` must be specified.",
                ]
            )
            raise ValueError(msg)

    def fit(
        self,
        x0: pyct.NDArray = None,
        stop_crit: pycs.StoppingCriterion = None,
        mode: pycs.Mode = pycs.Mode.BLOCK,
        tol: float = 1e-5,
    ):
        r"""
        Solve the minimization problem defined in :py:meth:`CG.__init__`, with the provided
        run-specifc parameters.
        Parameters
        ----------
        x0: NDArray
           (..., N) Starting guess(es) for the solution(s). Defaults to a zero NDArray if unspecified.
        stop_crit: StoppingCriterion
            Stopping criterion to end solver iterations. Defaults to a stopping criterion based on the convergence of
            the norm of the residual, up to the specified tolerance `tol`.
        mode: Mode
           Execution mode. See :py:class:`Solver` for usage examples.
           Useful method pairs depending on the execution mode:
           * BLOCK: fit()
           * ASYNC: fit() + busy() + stop()
           * MANUAL: fit() + steps()
        tol: Real
           Tolerance for convergence, norm(residual) <= tol.  Defaults to :math:`1e-5` if unspecified.
        """
        xp = pycu.get_array_module(self._b)
        x0 = x0 if (x0 is None) else pycrt.coerce(x0)

        try:
            assert tol > 0
        except:
            raise ValueError(f"tol must be positive, got {tol}.")
        if x0 is not None:
            try:
                assert x0.shape == self._b.shape
            except:
                raise ValueError(
                    f"Input initial guess has a mismatch in its shape dimension with data array `b` "
                    f"(shape {x0.shape} is different from {self._b.shape})."
                )
            try:
                assert pycu.get_array_module(x0) == xp
            except:
                raise ValueError(
                    f"Input initial guess has a mismatch in its shape array module with data array `b` "
                    f"(array module {pycu.get_array_module(x0)} is different from {xp}."
                )
        else:
            x0 = xp.zeros(self._b.shape, dtype=pycrt.getPrecision().value)
        if stop_crit is None:
            stop_crit = pycs.StoppingCriterion()

            def info(obj):
                mean_norms = xp.mean(obj.r_norms)
                try:
                    mean_norms = mean_norms.get()
                except:
                    pass
                return {"mean_rnorms": mean_norms}

            def stop(obj, state):
                obj.r_norms = xp.linalg.norm(state["r"], axis=-1)
                return (obj.r_norms < state["tol"]).all()

            setattr(stop_crit, "info", types.MethodType(info, stop_crit))
            setattr(stop_crit, "stop", types.MethodType(stop, stop_crit))
        self._fit_init(mode=mode, stop_crit=stop_crit)
        self._astate["history"] = []  # todo REMOVE line
        self.m_init(x0=x0, tol=tol)
        self._fit_run()

    def m_init(self, x0: pyct.NDArray, tol: float = 1e-5):
        self._mstate["x"] = x0
        self._mstate["r"] = r = self._b - self._a.apply(x0)
        self._mstate["p"] = r.copy()
        self._mstate["tol"] = tol

    def m_step(self):
        mst = self._mstate
        x = mst["x"]
        r = mst["r"]
        p = mst["p"]

        ap = self._a.apply(p)
        rr = (r * r).sum(-1)[..., None]
        alpha = rr / (p * ap).sum(-1)[..., None]
        x = x + alpha * p
        r = r - alpha * ap
        beta = (r * r).sum(-1)[..., None] / rr
        p = r + beta * p
        mst["x"], mst["r"], mst["p"] = x, r, p

    @pycrt.enforce_precision(o=True)
    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., N) The converged solution.
        """
        _, data = self.stats()
        return data.get("x")
