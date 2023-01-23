import itertools

import numpy as np
import pytest

import pycsou.opt.solver as pycos
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest
from pycsou.operator import shift_loss


class TestProxAdam(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou.operator.func import L1Norm, SquaredL2Norm

        klass = [
            pycos.ProxAdam,
        ]
        kwargs_init = [
            dict(f=SquaredL2Norm(), g=L1Norm()),
            dict(f=SquaredL2Norm()),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.zeros((N,)),
                np.zeros((3, 4, 2, N)),  # multiple initial points
            ],
            variant=["adam", "amsgrad", "padam"],
            p=[0.25],
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
