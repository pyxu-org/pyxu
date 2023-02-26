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
                np.stack(
                    [
                        np.full((N,), 1),
                        np.full((N,), 15),
                    ],
                    axis=0,
                ),  # multiple initial points
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

    # Note the different signature compared to SolverT.cost_function:
    # we need access to `b` from `kwargs_fit` to compute the cost function.
    @pytest.fixture
    def cost_function(self, kwargs_init, kwargs_fit) -> dict[str, pyct.OpT]:
        import pycsou.abc as pyca
        import pycsou.operator.blocks as pycb
        import pycsou.operator.func as pycf

        # The value of `b` determines the cost function.
        # Moreover several `b` may be provided.
        # A unique cost function is obtained by flattening `b` and block-expanding the quadratic
        # cost accordingly.

        A = kwargs_init["A"]
        b = kwargs_fit["b"]
        N_b = int(np.prod(b.shape[:-1]))
        func = pycf.QuadraticFunc(
            Q=pycb.block_diag((A,) * N_b),
            c=pyca.LinFunc.from_array(-b.reshape(-1), enable_warnings=False),
            t=0,
            init_lipschitz=False,  # not needed for [apply|grad]() calls.
        )
        return dict(x=func)

    @pytest.mark.parametrize("obj_threshold", [5e-2])
    # @pytest.mark.parametrize("time_threshold", [dt.timedelta(seconds=5)])
    def test_value_fit(
        self,
        solver,
        _kwargs_fit_xp,
        cost_function,
        ground_truth,
        obj_threshold,  # rel-threshold
        # time_threshold,  # max runtime; unused in this override
    ):
        solver.fit(**_kwargs_fit_xp.copy())
        data, _ = solver.stats()

        # cost_function() expects a flattened `b`: we must reshape the solver's output.
        b_dim = _kwargs_fit_xp["b"].ndim
        x = data["x"]
        x = x.reshape(
            *x.shape[:-b_dim],
            np.prod(x.shape[-b_dim:]),
        )
        data.update(x=x)

        self._check_value_fit(data, cost_function, ground_truth, obj_threshold)
