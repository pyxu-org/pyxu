import itertools

import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.opt.solver as pxs
import pyxu_tests.opt.solver.conftest as conftest


# We disable AutoInferenceWarning since we do not test convergence speed.
@pytest.mark.filterwarnings("ignore::pyxu.info.warning.AutoInferenceWarning")
class TestPGD(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pxt.SolverC, dict, dict]]:
        klass = [
            pxs.PGD,
        ]

        funcs = conftest.funcs(N, seed=3)
        stream1 = conftest.generate_funcs(funcs, N_term=1)
        stream2 = conftest.generate_funcs(funcs, N_term=2)
        kwargs_init = [
            *[dict(f=f, g=None) for (f, *_) in stream1],
            *[dict(f=None, g=g) for (g, *_) in stream1],
            *[dict(f=f, g=g) for (f, g) in stream2],
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full(N, 50),
                np.full((2, N), 50),  # multiple initial points
            ],
            tau=[None],
            acceleration=[True, False],
            d=[50, 75],
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
        func = [kwargs_init[k] for k in ("f", "g")]
        func = [f for f in func if f is not None]
        if len(func) == 1:  # f or g
            func = func[0]
        else:  # f and g
            func = func[0] + func[1]
        return dict(x=func)
