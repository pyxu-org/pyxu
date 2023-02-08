import itertools

import numpy as np
import pytest

import pycsou.opt.solver as pycos
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest


class TestNLCG(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        klass = [
            pycos.NLCG,
        ]

        funcs = conftest.funcs(N, seed=4)
        stream = conftest.generate_funcs(funcs, N_term=1)
        kwargs_init = [dict(f=f) for (f, *_) in stream]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
            ],
            variant=[
                "PR",
                "FR",
            ],
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

    @pytest.fixture
    def cost_function(self, kwargs_init) -> dict[str, pyct.OpT]:
        func = kwargs_init["f"]
        return dict(x=func)
