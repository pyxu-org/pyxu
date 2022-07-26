# #############################################################################
# lsqr.py
# ========
# Author : Kaan Okumus [okukaan@gmail.com]
# #############################################################################

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

    * For unsymmetric equations, it solves :math:`\mathbf{A}\mathbf{x}=\mathbf{b}`.

    * For linear least squares, it solves :math:`||\mathbf{b}-\mathbf{A}\mathbf{x}||^2`.

    * For damped least squares, it solves :math:`||\mathbf{b}-\mathbf{A}\mathbf{x}||^2 + d^2 ||\mathbf{x}-\mathbf{x_0}||^2`.


    ``LSQR()`` **Parameterization**

    A: :py:class:`pycsou.abc.operator.PosDefOp`
        Any positive definite linear operator is accepted.
    damp: float
        Damping coefficient. Default is 0.
    atol, btol: float
        Stopping tolerances. LSQR continues iterations until a certain backward error estimate is smaller than some
        quantity depending on atol and btol. Default is 1e-6 for both.
    conlim: float
        LSQR terminates if an estimate :math:`\text{cond}(\mathbf{A})` exceeds conlim. Default is 1e8.
    iter_lim: int, optional
        LSQR terminates if the number of iterations reaches iter_lim. Default is None.

    ``LSQR.fit()`` **Parameterization**

    b: NDArray
         (..., N) 'b' terms in the LSQR cost function. All problems are solved in parallel.
    x0: NDArray
        (..., N) initial point(s). Defaults to 0 if unspecified.

    **Remark:** :py:class:`pycsou.opt.solver.stop.StopCriterion_LSQR` is developed for the stopping criterion of LSQR. For
    computational speed, explicit norms were not computated. Instead, their estimation was used, which is referred from
    [1]_.

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
    True

    **References:**

    .. [1] C. C. Paige and M. A. Saunders "LSQR: An algorithm for sparse linear
        equations and sparse least squares", Dissertation,
        https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
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
        log_var: pyct.VarName = ("x",),
        **kwargs,
    ):
        super().__init__(
            log_var=log_var,
            **kwargs,
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

        b = xp.atleast_1d(b)
        if b.ndim > 1:
            b = b.squeeze()

        mst = self._mstate

        bnorm = xp.linalg.norm(b, axis=-1, keepdims=True)

        # Determine whether there's any trivial solution or not
        mst["trivial"] = bnorm == 0
        if mst["trivial"].all():
            mst["x"] = xp.zeros_like(self._A.T.apply(b))
            return
        elif mst["trivial"].any():
            # If some of them has trivial solution,
            # then remove them for the solving process
            # and add the trivial solution at the end of the process
            mst["b"] = b.copy()  # To remember the original one
            mask = xp.invert(mst["trivial"]).squeeze()
            b = b[mask]

        u = b
        bnorm = xp.linalg.norm(b, axis=-1, keepdims=True)

        if x0 is None:
            x = xp.zeros_like(self._A.T.apply(b))
            beta = bnorm.copy()
        else:
            x = xp.asarray(x0)
            u -= self._A.apply(x)
            beta = xp.linalg.norm(u, axis=-1, keepdims=True)

        beta_cp, beta_gt0, beta_eq0 = beta.copy(), beta > 0, beta == 0
        beta_cp[beta_eq0] = 1
        u /= beta_cp
        v = beta_gt0 * self._A.T.apply(u) + beta_eq0 * x
        alpha = beta_gt0 * xp.linalg.norm(v, axis=-1, keepdims=True)

        alpha_cp = alpha.copy()
        alpha_cp[alpha_cp <= 0] = 1
        v /= alpha_cp

        # Initialize state variables:
        mst["x"], mst["u"], mst["v"], mst["w"] = x, u, v, v.copy()
        mst["normA"] = mst["condA"] = mst["res2"] = mst["xnorm"] = mst["xxnorm"] = xp.zeros_like(beta)
        mst["ddnorm"] = mst["z"] = mst["sn2"] = xp.zeros_like(beta)
        mst["cs2"] = -xp.ones_like(beta)
        mst["rhobar"] = mst["alpha"] = alpha
        mst["phibar"] = mst["rnorm"] = mst["normr1"] = mst["normr2"] = beta
        mst["bnorm"] = bnorm
        mst["test1"], mst["test2"] = xp.ones_like(beta), alpha / beta
        mst["test3"] = mst["t1"] = mst["rtol"] = None

        # Vectorize the function of Givens rotation
        self._sym_ortho_func = xp.vectorize(self._sym_ortho)

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
        normA, ddnorm, xxnorm, bnorm = mst["normA"], mst["ddnorm"], mst["xxnorm"], mst["bnorm"]
        res2, sn2, cs2, z = mst["res2"], mst["sn2"], mst["cs2"], mst["z"]

        # Bidiagonalizaion to obtain next beta, u, alpha, v
        u = self._A.apply(v) - alpha * u
        beta = xp.linalg.norm(u, axis=-1, keepdims=True)

        if (beta > 0).all():
            u *= 1 / beta
            normA = xp.sqrt(normA**2 + alpha**2 + beta**2 + self._dampsq)
            v = self._A.T.apply(u) - beta * v
            alpha = xp.linalg.norm(v, axis=-1, keepdims=True)
            if (alpha > 0).all():
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
            psi = xp.zeros_like(beta)

        # Plane rotation to eliminate the subdiagonal element of lower-bidiagonal matrix
        cs, sn, rho = self._sym_ortho_func(rhobar1, beta, xp)

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

        condA = normA * xp.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = xp.sqrt(res1 + res2)
        arnorm = alpha * abs(tau)

        # Distinguish residual norms
        if self._damp > 0:
            r1sq = rnorm**2 - self._dampsq * xxnorm
            normr1 = xp.sqrt(xp.abs(r1sq))
            normr1[r1sq < 0] *= -1
        else:
            normr1 = rnorm
        normr2 = rnorm

        # Get necessary metrics for convergence test
        test1 = rnorm / bnorm
        test2 = arnorm / (normA * rnorm + self._eps)
        test3 = 1 / (condA + self._eps)
        t1 = test1 / (1 + normA * xnorm / bnorm)
        rtol = self._btol + self._atol * normA * xnorm / bnorm

        # Store necessary parameters
        mst["normA"], mst["ddnorm"], mst["xnorm"], mst["xxnorm"], mst["bnorm"] = normA, ddnorm, xnorm, xxnorm, bnorm
        mst["res2"], mst["sn2"], mst["cs2"], mst["z"] = res2, sn2, cs2, z
        mst["test1"], mst["test2"], mst["test3"] = test1, test2, test3
        mst["t1"], mst["rtol"] = t1, rtol
        mst["normr1"], mst["normr2"], mst["condA"] = normr1, normr2, condA

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.StopCriterion_LSQMR(
            method="lsqr",
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
