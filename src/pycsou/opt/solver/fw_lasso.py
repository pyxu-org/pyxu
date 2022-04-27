import typing as typ

# for development purposes
import numpy as np

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
from pycsou._dev import L1Norm, SquaredL2Norm, SubSampling
from pycsou.opt.solver.pgd import PGD

# [TODO] Need to import these methods from files where there will be properly defined


# [TODO] Integrate a minimum number of iterations

# Question: is there a way to keep track of the objective funciton values ? Or do we need to store it in the mstate ?


class GenericFWforLasso(pycs.Solver):
    r"""
    Base class for Frank-Wolfe algorithms (FW) for the LASSO problem.

    This is an abstract class that cannot be instantiated.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )
        self.lambda_ = pycrt.coerce(lambda_)
        self.forwardOp = forwardOp
        self.data = pycrt.coerce(data)

        # Do we need to/Is there an advantage of specifying the shape/dim of the inputs in the losses? (we have access
        # to them with the forward operator
        self._data_fidelity = 0.5 * SquaredL2Norm().asloss(data=self.data) * self.forwardOp
        self._penalty = self.lambda_ * L1Norm()  # rename as regul ??
        self.objective = self._data_fidelity + self._penalty

        self._bound = 0.5 * SquaredL2Norm()(data)[0] / self.lambda_  # todo : [0] te remove maybe ?

        # self.module = pycu.get_array_module(data)
        self.compute_ofv = True  # default

    # The inputs of m_init should be the type of the iterates to manipulate (module) and the running precision.
    def m_init(self, **kwargs):
        xp = pycu.get_array_module(self.data)
        mst = self._mstate  # shorthand

        mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision())
        mst["dcv"] = np.inf
        self.compute_ofv = kwargs.pop("compute_ofv", True)  # if we don't need the objective function, this can be
        # specified with compute_ofv=False
        # if isinstance(crit := self._astate["stop_crit"], (pycos.RelError, pycos.AbsError)):
        #     self.compute_ofv = (crit._var == "ofv") | kwargs.pop("compute_ofv", False)
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])[
                0
            ]  # todo : [0] te remove maybe ?  # objective function value should not be computed if not required
            # as stopping crit
            self._astate["log_var"] = frozenset(("ofv",)).union(self._astate["log_var"])
        # mst["positions"] = []  # numpy array of indices instead ?
        # should I keep track of previous encountered locations ? Allow for multiple indices = history

        # What happens at iteration 0 ? Before any algorithm iteration, should we update the variables and history ?

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.RelError(
            eps=1e-4,
            var="ofv",
            f=None,
            norm=2,
            satisfy_all=True,
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


class VanillaFWforLasso(GenericFWforLasso):
    r"""
    Vanilla version of the Frank-Wolfe algorithm (FW) specifically design to solve the LASSO problem of the form

    .. math ::

        {\argmin_{\mathbf{x} \in \mathbb{R}^N} \; \frac{1}{2} \lVert \mathbf{y} - \mathbf{G}(\mathbf{x}) \rVert_2^2
        + \lambda \lVert \mathbf{x} \rVert_1 }

    where:

    * :math:`\mathbf{G}:\mathbb{R}^N\rightarrow\mathbb{C}^L` is a *linear* operator, referred to as *forward*
     or *measurement* operator.
    * :math:`\mathbf{y}\in\mathbb{C}^L` is the vector of measurements data.
    * :math:`\lambda>0` is the *regularization* or *penalty* parameter.

    The FW algorithms ensure convergence of the iterates in terms of the value of the objective function with a
    convergence rate of :math:`\mathcal{O}(1/k)`[RevFW]_. Consequently, the default stopping criterion is set as a threshold
    over the relative improvement of the value of the objective function. The algorithm stops if the relative
    improvement is below 1e-4. If this default stopping criterion is used, you must not run the algorithm with argument
    ``compute_ofv=False``.

    **Remark:** The array module used in the algorithm iterations is inferred from the module of the input measurements.

    ``VanillaFWforLasso.fit()`` **Parametrisation**

    compute_ofv: Bool
        Indicator to keep track of the value of the objective function along the iterations.
        This value is optional for running the Vanilla FW iterations, but can be used to determine a stopping criterion.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        step_size: str = "optimal",
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 50,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):
        r"""

        Parameters
        ----------
        data: NDArray
            (..., L) measurements term(s) in the LASSO objective function.
        forwardOp: LinOp
            (N, L) measurement operator in the LASSO objective function.
        lambda_: float
            Regularisation parameter from the LASSO problem.
        step_size: str
            ["regular", "optimal"] If ``step_size="regular"``, the convex combination weight in the weighting step of
            FW is set according to the default strategy, :math:`\gamma_k = 2/k+2`.
            If ``step_size="regular"``, the convex combination weight :math:`\gamma_k` is computed at each step
            as the optimal step size, for which there exists a close form formulation in the specific case of minimising
            the LASSO objective function.
        """
        super().__init__(
            data=data,
            forwardOp=forwardOp,
            lambda_=lambda_,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

        try:
            assert step_size in ["optimal", "regular"]
            self.step_size_strategy = step_size
        except:
            raise ValueError(f"step_size must be in ['regular', 'optimal'], got {step_size}.")

    def m_init(self, **kwargs):
        super(VanillaFWforLasso, self).m_init(**kwargs)
        self._mstate["lift_variable"] = 0.0  # todo do I need to enforce the precision on plain floats ?

    def m_step(self):
        mst = self._mstate  # shorthand
        xp = pycu.get_array_module(mst["x"])
        mgrad = -self._data_fidelity.grad(mst["x"])
        new_ind = xp.argmax(xp.abs(mgrad), axis=-1)  # axis for parallel runs
        dcv = mgrad[new_ind] / self.lambda_
        mst["dcv"] = dcv

        if self.step_size_strategy == "regular":
            gamma = 2 / (2 + self._astate["idx"])
        elif self.step_size_strategy == "optimal":
            gamma = -xp.dot(mgrad, mst["x"]).real
            if abs(dcv) > 1.0:
                gamma += self.lambda_ * (mst["lift_variable"] + (abs(dcv) - 1.0) * self._bound)
                injection = SubSampling(self.forwardOp.shape[1], xp.array(new_ind)).T
                # print(SquaredL2Norm(self.bound * np.sign(dcv) * self.forwardOp(injection(self.module.array(1.))) -
                #                        self.forwardOp(mst["x"])))
                # print(self.bound * np.sign(dcv) * self.forwardOp(injection(self.module.array(1.))))
                gamma /= SquaredL2Norm()(
                    self._bound * np.sign(dcv) * self.forwardOp(injection(xp.array(1.0))) - self.forwardOp(mst["x"])[0]
                )  # we can use numpy (np) as dcv is a float
            else:
                gamma += self.lambda_ * mst["lift_variable"]
                gamma /= SquaredL2Norm()(self.forwardOp(mst["x"]))[0]

        if not 0 < gamma < 1:
            print("Warning, gamma value not valid: {}".format(gamma))
            gamma = xp.clip(gamma, 0.0, 1.0)

        mst["x"] *= 1 - gamma
        mst["lift_variable"] *= 1 - gamma
        if abs(dcv) > 1.0:
            mst["x"][new_ind] += pycrt.coerce(gamma * np.sign(dcv) * self._bound)
            mst["lift_variable"] += gamma * self._bound
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])[0]


# 2 ideas for efficient restricted support computations:
#       * provide a restricted support operator as input for PFW
#       * create a new class the inherits from PFW and that redefines the m_step (or the targeted submethod)
class PolyatomicFWforLasso(GenericFWforLasso):
    r"""
    Polyatomic version of the Frank-Wolfe algorithm (FW) specifically design to solve the LASSO problem of the form

    .. math ::

        {\argmin_{\mathbf{x} \in \mathbb{R}^N} \; \frac{1}{2} \lVert \mathbf{y} - \mathbf{G}(\mathbf{x}) \rVert_2^2
        + \lambda \lVert \mathbf{x} \rVert_1 }

    where:

    * :math:`\mathbf{G}:\mathbb{R}^N\rightarrow\mathbb{C}^L` is a *linear* operator, referred to as *forward*
     or *measurement* operator.
    * :math:`\mathbf{y}\in\mathbb{C}^L` is the vector of measurements data.
    * :math:`\lambda>0` is the *regularization* or *penalty* parameter.

    This algorithm is presented in the paper [PFW]_ as an improvement over the Vanilla FW algorithm in order to
    converge faster to a solution. Compared to other LASS solvers, it is particularly efficient when the solution is
    supposed to be very sparse, in terms of convergence time as well as memory requirements.

    The FW algorithms ensure convergence of the iterates in terms of the value of the objective function with a
    convergence rate of :math:`\mathcal{O}(1/k)` [RevFW]_. Consequently, the default stopping criterion is set as a threshold
    over the relative improvement of the value of the objective function. The algorithm stops if the relative
    improvement is below 1e-4. If this default stopping criterion is used, you must not run the algorithm with argument
    ``compute_ofv=False``.

    **Remark:** The array module used in the algorithm iterations is inferred from the module of the input measurements.

    ``PolyatomicFWforLasso.fit()`` **Parametrisation**

    compute_ofv: Bool
        Indicator to keep track of the value of the objective function along the iteration.
        This value is optional for Vanilla FW, but can be used to determine a stopping criterion.
    """

    def __init__(
        self,
        data: pyct.NDArray,
        forwardOp: pyco.LinOp,
        lambda_: float,
        ms_threshold: float = 0.7,  # multi spikes threshold at init
        init_correction_prec: float = 0.2,
        final_correction_prec: float = 1e-4,
        remove_positions: bool = False,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 10,
        show_progress: bool = True,
        log_var: pyct.VarName = (
            "x",
            "dcv",
        ),
    ):

        r"""

        Parameters
        ----------
        data: NDArray
            (..., L) measurements term(s) in the LASSO objective function.
        forwardOp: LinOp
            (N, L) measurement operator in the LASSO objective function.
        lambda_: float
            Regularisation parameter from the LASSO problem.
        ms_threshold: float
            Initial threshold for identifying the first atoms, given as a rate of the dual certificate value. This
            parameter impacts the number of atomes chosen at the first step, but also at all the following iterations.
            Low `ms_threshold` value implies a lot of flexiility in the choice of the atoms, whereas high value leads
            to restrictive selection and thus very few atomes at each iteration.
        init_correction_prec: float
            Precision of the first stopping criterion for the step of correction of the weights at first iteration.
        final_correction_prec: float
            Lower bound for the precision of the correction steps. When reached, the precision does no longer decrease
            with the iterations.
        remove_positions: bool
            When set to `True`, the atoms that get attributed a null weight after the correction of the weights are
            removed from the current iterate. When the parameter is set to `False` however, these atoms remain in the
            set of active indices and thus can still be attributed weight at later iterations.
        """
        self._ms_threshold = ms_threshold
        self._init_correction_prec = init_correction_prec
        self._final_correction_prec = final_correction_prec
        self._remove_positions = remove_positions
        super().__init__(
            data=data,
            forwardOp=forwardOp,
            lambda_=lambda_,
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

    def m_init(self, **kwargs):
        super(PolyatomicFWforLasso, self).m_init(**kwargs)
        xp = pycu.get_array_module(self._mstate["x"])
        mst = self._mstate
        mst["positions"] = xp.array([], dtype="int32")
        # is it worth storing this as an xp array, given that it should remain small or moderate size ?
        # maybe the set of active indices can be stored as an np array
        mst["delta"] = None  # initial buffer for multi spike thresholding
        mst["correction_prec"] = self._init_correction_prec

    def m_step(self):
        mst = self._mstate  # shorthand
        mgrad = -self._data_fidelity.grad(mst["x"])
        mst["dcv"] = max(mgrad.max(), mgrad.min(), key=abs) / self.lambda_
        maxi = abs(mst["dcv"])
        if self._astate["idx"] == 1:
            mst["delta"] = maxi * (1.0 - self._ms_threshold)
        thresh = maxi - (2 / self._astate["idx"] + 1) * mst["delta"]
        new_indices = (abs(mgrad) > max(thresh, 1.0)).nonzero()[0]

        xp = pycu.get_array_module(mst["x"])
        if new_indices.size > 0:
            if self._astate["idx"] > 1 and self._remove_positions:
                mst["positions"] = xp.unique(xp.hstack([mst["x"].nonzero()[0], new_indices]))
            else:
                mst["positions"] = xp.unique(xp.hstack([mst["positions"], new_indices]))
        elif self._remove_positions:
            mst["positions"] = (mst["x"] > 1e-5).nonzero()[0]
        # else would correspond to empty new_indices, in this case the set of active indices does not change

        mst["correction_prec"] = max(self._init_correction_prec / self._astate["idx"], self._final_correction_prec)
        if mst["positions"].size > 1:
            mst["x"] = self.rs_correction(mst["positions"])
        elif mst["positions"].size == 1:
            tmp = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision())
            tmp[mst["positions"]] = 1.0
            column = self.forwardOp(tmp)
            corr = xp.dot(self.data, column).real
            if abs(corr) <= self.lambda_:
                mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision())
            elif corr > self.lambda_:
                mst["x"] = ((corr - self.lambda_) / SquaredL2Norm()(column)[0]) * tmp
            else:
                mst["x"] = ((corr + self.lambda_) / SquaredL2Norm()(column)[0]) * tmp
        else:
            mst["x"] = xp.zeros(self.forwardOp.shape[1], dtype=pycrt.getPrecision())
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])[0]

    def rs_correction(self, support_indices: pyct.NDArray) -> pyct.NDArray:
        def correction_stop_crit(eps) -> pycs.StoppingCriterion:
            stop_crit = pycos.RelError(
                eps=eps,
                var="x",
                f=None,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        injection = SubSampling(size=self.forwardOp.shape[1], sampling_indices=support_indices).T
        # rsOp = self.forwardOp * injection
        # # rsOp._lipschitz = self.forwardOp.lipschitz()
        rs_data_fid = self._data_fidelity * injection
        # rs_penalty = self.penalty * injection
        x0 = injection.T(self._mstate["x"])
        apgd = PGD(rs_data_fid, self._penalty, show_progress=False)
        apgd.fit(x0=x0, stop_crit=correction_stop_crit(self._mstate["correction_prec"]))
        sol, _ = apgd.stats()
        return injection(sol["x"])


def dcvStoppingCrit(eps: float = 1e-2) -> pycs.StoppingCriterion:
    r"""
    Instantiate a ``StoppingCriterion`` class based on the maximum value of the dual certificate, that is automatically
    computed along the iterations of any Frank-Wolfe algorithm. At convergence, the maximum absolute value component of
    the dual certificate is 1, and thus tracking the difference to 1 provides us with a natural stopping criterion.

    Parameters
    ----------
    eps: float
        Precision on the distance to 1 for stopping.

    Returns
    -------
    stop_crit: pycs.StoppingCriterion
        An instance of the ``StoppingCriterion`` class that can be fed to the ``.fit()`` method of any
        Frank-Wolfe solver.
    """

    def abs_diff_to_one(x):
        return abs(x) - 1.0

    stop_crit = pycos.AbsError(
        eps=eps,
        var="dcv",
        f=abs_diff_to_one,
        norm=2,
        satisfy_all=True,
    )
    return stop_crit


# todo :
#   * test pfw first remove=False then True
#   * consistency of the docstrings: s or z in -zation -ze words
#   * define private and public variables
