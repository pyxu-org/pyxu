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

# [TODO] Need to import these methods from files where there will be properly defined


# [TODO] Integrate a minimum number of iterations

# Question: is there a way to keep track of the objective funciton values ? Or do we need to store it in the mstate ?


class VanillaFWforLasso(pycs.Solver):
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
        try:
            assert step_size in ["optimal", "regular"]
            self.step_size_strategy = step_size
        except:
            raise ValueError(f"step_size must be in ['optimal', 'regular'], got {step_size}.")

        # Do we need to/Is there an advantage of specifying the shape/dim of the inputs in the losses? (we have access
        # to them with the forward operator
        self.data_fidelity = 0.5 * SquaredL2Norm().asloss(data=data) * self.forwardOp
        self.penalty = self.lambda_ * L1Norm()
        self.objective = self.data_fidelity + self.penalty

        self.bound = 0.5 * SquaredL2Norm()(data) / self.lambda_

        self.module = pycu.get_array_module(data)

    # The inputs of m_init should be the type of the iterates to manipulate (module) and the running precision.
    def m_init(self, **kwargs):
        mst = self._mstate  # shorthand

        mst["x"] = np.zeros(self.forwardOp.shape[1])
        mst["lift_variable"] = 0.0
        mst["dcv"] = None
        if isinstance(crit := self._astate["stop_crit"], (pycos.RelError, pycos.AbsError)):
            self.compute_ofv = (crit._var == "ofv") | kwargs.pop("compute_ofv", False)
        if self.compute_ofv:
            mst["ofv"] = self.objective(mst["x"])  # objective funciton value should not be computed if not required
            # as stopping crit
            self._astate["log_var"] = frozenset(("ofv",)).union(self._astate["log_var"])
        # mst["positions"] = []  # numpy array of indices instead ?
        # should I keep track of previous encountered locations ? Allow for multiple indices = history

        # What happens at iteration 0 ? Before any algorithm iteration, should we update the variables and history ?
        # [question] Should I define self.bound inside m_init ?

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

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.RelError(
            eps=1e-5,
            var="ofv",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit


# todo vendredi:
# todo      * make sure module agnosticity is working fine
#           * enforce data precision


class PolyatomicFWforLasso(pycs.Solver):
    def __init__(self):
        pass

    def m_init(self, **kwargs):
        pass

    def m_step(self):
        pass
