import itertools

import numpy as np
import pytest

import pycsou.opt.solver as pycos
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest


def op(N: int = 5) -> pyct.OpT:
    # implements \norm{Ax - b}{2}^{2}, with
    # * b = [1 ... 1] \in \bR^{N}
    # * A \in \bR^{N \times N}
    # * x \in \bR^{N}
    from pycsou_tests.operator.examples.test_proxdifffunc import SquaredL2Norm
    from pycsou_tests.operator.examples.test_unitop import Permutation

    A = Permutation(N=N)
    b = 1
    L2 = SquaredL2Norm()
    op_ = L2.asloss(b) * A
    return op_


# We disable AutoInferenceWarning since we do not test convergence speed.
@pytest.mark.filterwarnings("ignore::pycsou.util.warning.AutoInferenceWarning")
class TestPGD(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        klass = [
            pycos.PGD,
        ]
        kwargs_init = [
            dict(f=op(N), g=None),
            dict(f=None, g=op(N)),
            dict(f=op(N), g=op(N)),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.zeros(N),
                np.zeros((2, N)),  # multiple initial points
            ],
            tau=[
                None,
                0.5 / op(N).diff_lipschitz(),  # below convergence limit -> ok
            ],
            acceleration=[True, False],
            d=[50, 75],
        )
        for config in itertools.product(*param_sweep.values()):
            d = dict(zip(param_sweep.keys(), config))
            kwargs_fit.append(d)

        data = itertools.product(klass, kwargs_init, kwargs_fit)
        return list(data)

    @pytest.fixture(params=spec_data(N=7))
    def spec(self, request) -> tuple[pyct.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit
