import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


class LSQR(pycs.Solver):
    r"""
     LSQR Method.

     The LSQR method solves the system of linear equations :math:`\mathbf{A}\mathbf{x}=\mathbf{b}` iteratively.
     :math:`\mathbf{A}` is a rectangular matrix of dimension m-by-n, where all cases are allowed: m=n, m>n or m<n.
     :math:`\mathbf{b}` is a vector of length m. The matrix :math:`\mathbf{A}` may be square or rectangular and may have
     any rank.
     For unsymmetric equations, it solves :math:`\mathbf{A}\mathbf{x}=\mathbf{b}`.
     For linear least squares, it solves :math:`||\mathbf{b}-\mathbf{A}\mathbf{x}||^2`.
     For damped least squares, it solves :math:`||\mathbf{b}-\mathbf{A}\mathbf{x}||^2 + d^2 ||\mathbf{x}-\mathbf{x_0}||
     ^2`.


     ``CG()`` **Parameterization**
     A: pycsou.abc.operator.PosDefOp
        Any positive definite linear operator is accepted.
     damp: float
        Damping coefficient. Default is 0.
     atol, btol: float
        Stopping tolerances. LSQR continues iterations until a certain backward error estimate is smaller than some
        quantity depending on atol and btol. Default is 1e-6 for both.
     conlim: float
        LSQR terminates if an estimate :math:`cond(\mathbf{A})` exceeds conlim. Default is 1e8.
     iter_lim: int, optional
        LSQR terminates if the number of iterations reaches iter_lim. Default is None.

     ``CG.fit()`` **Parameterization**

     b: NDArray
         (..., N) 'b' terms in the LSQR cost function. All problems are solved in parallel.
     x0: NDArray
        (..., N) initial point(s). Defaults to 0 if unspecified.
     restart_rate: int
        Number of iterations after which restart is applied. By default, restart is done after 'n' iterations, where 'n'
        corresponds to the dimension of the linear operator :math:`\mathbf{A}`.

    **Remark:** In pycsou.opt.solver.stop, StopCriterion_LSQR is developed for the stopping criterion of LSQR. For
    computational speed, explicit norms were not computated. Instead, their estimation was used, which is referred from
    [1].

     References
     ----------
        [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
            and Least-Squares Problems", Dissertation,
            http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
     Examples
     --------
     >>> import numpy as np
     >>> from pycsou.abc import LinOp
     >>> # Create a PSD linear operator
     >>> rng = np.random.default_rng(seed=0)
     >>> mat = rng.normal(size=(10, 10))
     >>> A = LinOp.from_array(mat).gram()
     >>> # Create the ground truth 'x_star'
     >>> x_star = rng.normal(size=(2, 2, 10))
     >>> # Generate the corresponding data vector 'b'
     >>> b = A.apply(x_star)
     >>> # Solve 'Ax=b' for 'x' with the conjugate gradient method
     >>> lsqr = LSQR(A, show_progress=False)
     >>> lsqr.fit(b=b)
     >>> x_solution = lsqr.solution()
     >>> assert np.allclose(x_star, x_solution)
    """

    def __init__(
        self,
        A: pyco.PosDefOp,
        *,
        damp: float = 0.0,
        atol: float = 1e-06,
        btol: float = 1e-06,
        conlim: float = 1e08,
        iter_lim: typ.Optional[int] = None,
        eps: float = np.finfo(np.float64).eps,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

        self._A = A
        self._atol = atol
        self._btol = btol
        self._ctol = (1.0 / conlim) if (conlim > 0) else 0.0
        self._iter_lim = (2 * A.shape[1]) if (iter_lim is None) else iter_lim
        self._dampsq, self._damp = damp**2, damp
        self._eps = eps

    @pycrt.enforce_precision(i=["b", "x0"], allow_None=True)
    def m_init(
        self,
        b: pyct.NDArray,
        x0: typ.Optional[pyct.NDArray] = None,
    ):
        xp = pycu.get_array_module(b)
        mst = self._mstate
        mst["anorm"] = mst["acond"] = mst["res2"] = mst["xnorm"] = mst["xxnorm"] = 0
        mst["ddnorm"] = mst["z"] = mst["sn2"] = 0
        mst["cs2"] = -1

        u = b
        bnorm = xp.linalg.norm(b)

        if x0 is None:
            # TODO: make sure shape is correct
            x = xp.zeros((self._A.shape[1],))
            beta = bnorm.copy()
        else:
            x = x0
            u = u - self._A.apply(x)
            beta = xp.linalg.norm(u)

        if beta > 0:
            u = (1 / beta) * u
            v = self._A.T.apply(u)
            alpha = xp.linalg.norm(v)
        else:
            v = x.copy()
            alpha = 0

        if alpha > 0:
            v = (1 / alpha) * v

        mst["x"] = x
        mst["u"], mst["v"] = u, v
        mst["w"] = v.copy()
        mst["rhobar"] = mst["alpha"] = alpha
        mst["phibar"] = mst["rnorm"] = mst["r1norm"] = mst["r2norm"] = beta
        mst["arnorm"], mst["bnorm"] = alpha * beta, bnorm
        mst["itn"], mst["istop"] = 0, None
        mst["test1"], mst["test2"] = 1.0, alpha / beta
        mst["test3"] = mst["t1"] = mst["rtol"] = None

    @staticmethod
    def _sym_ortho(a, b, xp):
        """
        Stable implementation of Givens rotation.
        """
        if b == 0:
            return xp.sign(a), 0, xp.abs(a)
        elif a == 0:
            return 0, xp.sign(b), xp.abs(b)
        elif xp.abs(b) > xp.abs(a):
            tau = a / b
            s = xp.sign(b) / xp.sqrt(1 + tau * tau)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = xp.sign(a) / xp.sqrt(1 + tau * tau)
            s = c * tau
            r = a / c
        return c, s, r

    def m_step(self):

        mst = self._mstate  # shorthand
        xp = pycu.get_array_module(mst["x"])
        x, u, v, w = mst["x"], mst["u"], mst["v"], mst["w"]
        alpha, rhobar, phibar = mst["alpha"], mst["rhobar"], mst["phibar"]
        anorm, ddnorm, xxnorm, bnorm = mst["anorm"], mst["ddnorm"], mst["xxnorm"], mst["bnorm"]
        res2, sn2, cs2, z = mst["res2"], mst["sn2"], mst["cs2"], mst["z"]

        # Bidiagonalizaion to obtain next beta, u, alpha, v
        u = self._A.apply(v) - alpha * u
        beta = xp.linalg.norm(u)

        if beta > 0:
            u = (1 / beta) * u
            anorm = xp.sqrt(anorm**2 + alpha**2 + beta**2 + self._dampsq)
            v = self._A.T.apply(u) - beta * v
            alpha = xp.linalg.norm(v)
            if alpha > 0:
                v = (1 / alpha) * v

        # Plane rotation to eliminate the damping parameter
        if self._damp > 0:
            rhobar1 = xp.sqrt(rhobar**2 + self._dampsq)
            cs1 = rhobar / rhobar1
            sn1 = self._damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0

        # Plane rotation to eliminate the subdiagonal element of lower-bidiagonal matrix
        cs, sn, rho = self._sym_ortho(rhobar1, beta, xp)

        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w
        x = x + t1 * w
        w = v + t2 * w

        mst["x"], mst["u"], mst["v"], mst["w"] = x, u, v, w
        mst["alpha"], mst["rhobar"], mst["phibar"] = alpha, rhobar, phibar

        # Estimation of norms:
        ddnorm = ddnorm + xp.linalg.norm(dk) ** 2
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = xp.sqrt(xxnorm + zbar**2)
        gamma = xp.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        acond = anorm * xp.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = xp.sqrt(res1 + res2)
        arnorm = alpha * abs(tau)

        # Distinguish residual norms
        if self._damp > 0:
            r1sq = rnorm**2 - self._dampsq * xxnorm
            r1norm = xp.sqrt(xp.abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        # Get necessary metrics for convergence test
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + self._eps)
        test3 = 1 / (acond + self._eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = self._btol + self._atol * anorm * xnorm / bnorm

        # Store necessary parameters
        mst["anorm"], mst["ddnorm"], mst["xnorm"], mst["xxnorm"], mst["bnorm"] = anorm, ddnorm, xnorm, xxnorm, bnorm
        mst["res2"], mst["sn2"], mst["cs2"], mst["z"] = res2, sn2, cs2, z
        mst["test1"], mst["test2"], mst["test3"] = test1, test2, test3
        mst["t1"], mst["rtol"] = t1, rtol
        mst["r1norm"], mst["r2norm"], mst["acond"] = r1norm, r2norm, acond

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.StopCriterion_LSQR(
            atol=self._atol,
            ctol=self._ctol,
            itn=0,
            iter_lim=self._iter_lim,
        )

        return stop_crit

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")
