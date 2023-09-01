import itertools

import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.opt.solver as pxs
import pyxu_tests.opt.solver.conftest as conftest


class TestAdam(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pxt.SolverC, dict, dict]]:
        klass = [
            pxs.Adam,
        ]

        funcs = conftest.funcs(N, seed=5)
        stream1 = conftest.generate_funcs(funcs, N_term=1)
        kwargs_init = [
            *[dict(f=f) for (f, *_) in stream1],
            # We do not test f=None case since unsupported by Adam().
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 50),
                np.full((2, 1, 3, N), 50),  # multiple initial points
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
    def spec(self, request) -> tuple[pxt.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit

    @pytest.fixture
    def cost_function(self, kwargs_init) -> dict[str, pxt.OpT]:
        return dict(x=kwargs_init["f"])
