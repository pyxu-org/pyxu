import itertools

import numpy as np
import pytest

import pycsou.opt.solver as pycos
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest


class TestCG(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.CG,
        ]
        kwargs_init = [
            dict(A=PSDConvolution(N=N)),
        ]

        kwargs_fit = []
        param_sweep = dict(
            b=[
                np.ones((N,)),
                np.full((2, N), -2),  # multiple problems in parallel
            ],
            x0=[
                None,  # let algorithm choose
                np.full((N,), 3),
                np.vstack([np.full((1, N), 1), (np.full((1, N), 15))]),  # multiple initial points
            ],
            restart_rate=[None, N, 2 * N],
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
