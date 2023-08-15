import itertools

import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.opt.solver as pxs
import pyxu_tests.opt.solver.conftest as conftest


class TestNLCG(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pxt.SolverC, dict, dict]]:
        klass = [
            pxs.NLCG,
        ]

        funcs = conftest.funcs(N, seed=4)
        stream = conftest.generate_funcs(funcs, N_term=1)
        kwargs_init = [dict(f=f) for (f, *_) in stream]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 50),
                np.full((2, N), 50),  # multiple initial points
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
    def spec(self, request) -> tuple[pxt.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit

    @pytest.fixture
    def cost_function(self, kwargs_init) -> dict[str, pxt.OpT]:
        func = kwargs_init["f"]
        return dict(x=func)
