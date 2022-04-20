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
        self.lambda_ = lambda_
        self.forwardOp = forwardOp
        self.data = data

        # Do we need to/Is there an advantage of specifying the shape/dim of the inputs in the losses? (we have access
        # to them with the forward operator
        self.data_fidelity = 0.5 * SquaredL2Norm().asloss(data=data) * self.forwardOp
        self.penalty = self.lambda_ * L1Norm()  # rename as regul ??
        self.objective = self.data_fidelity + self.penalty

        self.bound = 0.5 * SquaredL2Norm()(data) / self.lambda_

        self.module = pycu.get_array_module(data)
        self.compute_ofv = True

    # The inputs of m_init should be the type of the iterates to manipulate (module) and the running precision.
    def m_init(self, **kwargs):
        # xp = pycu.get_array_module
        mst = self._mstate  # shorthand

        mst["x"] = np.zeros(self.forwardOp.shape[1])
        mst["dcv"] = np.inf
        self.compute_ofv = kwargs.pop("compute_ofv", True)  # if we don't need the objective function, this can be
        # specified with compute_ofv=False
        # if isinstance(crit := self._astate["stop_crit"], (pycos.RelError, pycos.AbsError)):
        #     self.compute_ofv = (crit._var == "ofv") | kwargs.pop("compute_ofv", False)
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])  # objective function value should not be computed if not required
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


class VanillaFWforLasso(GenericFWforLasso):
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
            raise ValueError(f"step_size must be in ['optimal', 'regular'], got {step_size}.")

    def m_init(self, **kwargs):
        super(VanillaFWforLasso, self).m_init(**kwargs)
        self._mstate["lift_variable"] = 0.0

    def m_step(self):
        mst = self._mstate  # shorthand
        mgrad = -self.data_fidelity.grad(mst["x"])
        new_ind = np.argmax(np.abs(mgrad), axis=-1)  # axis for parallel runs
        dcv = mgrad[new_ind] / self.lambda_
        mst["dcv"] = dcv

        if self.step_size_strategy == "regular":
            gamma = 2 / (2 + self._astate["idx"])
        elif self.step_size_strategy == "optimal":
            gamma = -np.dot(mgrad, mst["x"])
            if abs(dcv) > 1.0:
                gamma += self.lambda_ * (mst["lift_variable"] + (abs(dcv) - 1.0) * self.bound)
                injection = SubSampling(self.forwardOp.shape[1], self.module.array(new_ind)).T
                # print(SquaredL2Norm(self.bound * np.sign(dcv) * self.forwardOp(injection(self.module.array(1.))) -
                #                        self.forwardOp(mst["x"])))
                # print(self.bound * np.sign(dcv) * self.forwardOp(injection(self.module.array(1.))))
                gamma /= SquaredL2Norm()(
                    self.bound * np.sign(dcv) * self.forwardOp(injection(self.module.array(1.0)))
                    - self.forwardOp(mst["x"])
                )
            else:
                gamma += self.lambda_ * mst["lift_variable"]
                gamma /= SquaredL2Norm()(self.forwardOp(mst["x"]))

        if not 0 < gamma < 1:
            print("Warning, gamma value not valid: {}".format(gamma))
            gamma = np.clip(gamma, 0.0, 1.0)

        mst["x"] *= 1 - gamma
        mst["lift_variable"] *= 1 - gamma
        if abs(dcv) > 1.0:
            mst["x"][new_ind] += gamma * np.sign(dcv) * self.bound
            mst["lift_variable"] += gamma * self.bound
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])


# todo vendredi:
# todo      * make sure module agnosticity is working fine : use xp.get_module_... and not self.array_module
#           * enforce data precision
#           * define private and public variables

# 2 ideas for efficient restricted support computations:
#       * provide a restricted support operator as input for PFW
#       * create a new class the inherits from PFW and that redefines the m_step (or the targeted submethod)
class PolyatomicFWforLasso(GenericFWforLasso):
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
        self._ms_threshold = ms_threshold
        self._init_correction_prec = init_correction_prec
        self._final_correction_prec = final_correction_prec
        self._correction_prec = init_correction_prec
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
        self._mstate["positions"] = self.module.array([], dtype="int32")
        # is it worth storing this as an xp array, given that it should remain small or moderate size ?

    def m_step(self):
        mst = self._mstate  # shorthand
        mgrad = -self.data_fidelity.grad(mst["x"])
        mst["dcv"] = max(mgrad.max(), mgrad.min(), key=abs) / self.lambda_
        maxi = abs(mst["dcv"])
        if self._astate["idx"] == 1:
            self.delta = maxi * (1.0 - self._ms_threshold)
        thresh = maxi - (2 / self._astate["idx"] + 1) * self.delta
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

        self._correction_prec = max(self._init_correction_prec / self._astate["idx"], self._final_correction_prec)
        if mst["positions"].size > 1:
            mst["x"] = self.rs_correction(mst["positions"])
        elif mst["positions"].size == 1:
            tmp = xp.zeros(self.forwardOp.shape[1])
            tmp[mst["positions"]] = 1.0
            column = self.forwardOp(tmp)
            corr = xp.dot(self.data, column).real
            if abs(corr) <= self.lambda_:
                mst["x"] = xp.zeros(self.forwardOp.shape[1])
            elif corr > self.lambda_:
                mst["x"] = ((corr - self.lambda_) / SquaredL2Norm()(column)) * tmp
            else:
                mst["x"] = ((corr + self.lambda_) / SquaredL2Norm()(column)) * tmp
        else:
            mst["x"] = xp.zeros(self.forwardOp.shape[1])

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
        rs_data_fid = self.data_fidelity * injection
        # rs_penalty = self.penalty * injection
        x0 = injection.T(self._mstate["x"])
        apgd = PGD(rs_data_fid, self.penalty, verbosity=10000)
        apgd.fit(x0=x0, stop_crit=correction_stop_crit(self._correction_prec))
        sol, _ = apgd.stats()
        return injection(sol)


def DCVStoppingCrit(eps: float = 1e-2) -> pycs.StoppingCriterion:
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
#   * then check module agnosticity and enforce precision
#   * verbosity = None => infinity
