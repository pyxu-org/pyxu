import types
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.linop.base as pyclo
import pycsou.runtime as pycrt
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
        try:
            assert tol > 0
        except:
            raise ValueError(f"tol must be positive, got {tol}.")
        if x0 is not None:
            try:
                assert x0.shape == self._b
            except:
                raise ValueError(
                    f"Input initial guess has a mismatch in its shape dimension with data array `b` "
                    f"(shape {x0.shape} is different from {self._b.shape}."
                )
        else:
            x0 = np.zeros(self._b.shape)
        if stop_crit is None:
            stop_crit = pycs.StoppingCriterion()

            def info(obj):
                return {"mean_rnorms": np.mean(obj.r_norms)}

            def stop(obj, state):
                obj.r_norms = np.linalg.norm(state["r"], axis=-1)
                return (obj.r_norms < state["tol"]).all()

            setattr(stop_crit, "info", types.MethodType(info, stop_crit))
            setattr(stop_crit, "stop", types.MethodType(stop, stop_crit))
        self._fit_init(mode=mode, stop_crit=stop_crit)
        self._astate["history"] = []  # todo REMOVE line
        self.m_init(x0=x0, tol=tol)
        self._fit_run()

    def m_init(self, x0: pyct.NDArray, tol: float = 1e-5):
        self._mstate["x"] = pycrt.coerce(x0)
        self._mstate["r"] = r = self._b - self._a.apply(x0)
        self._mstate["p"] = np.copy(r)
        self._mstate["tol"] = tol

    def m_step(self):
        mst = self._mstate
        x = mst["x"]
        r = mst["r"]
        p = mst["p"]

        ap = self._a.apply(p)
        rr = np.einsum("...j,...j->...", r, r).reshape(*r.shape[:-1], 1)
        alpha = rr / np.einsum("...j,...j->...", p, ap).reshape(*r.shape[:-1], 1)
        x = x + alpha * p
        r = r - alpha * ap
        beta = np.einsum("...j,...j->...", r, r).reshape(*r.shape[:-1], 1) / rr
        p = r + beta * p
        mst["x"], mst["r"], mst["p"] = x, r, p

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., N) The converged solution.
        """
        _, data = self.stats()
        return data.get("x")
