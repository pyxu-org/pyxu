import itertools

import numpy as np
import pytest

import pycsou.opt.solver as pycos
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest
from pycsou_tests.operator.conftest import allclose


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
                np.full((N,), 30),
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

    # Note the different signature compared to SolverT.cost_function():
    # we need access to `b` from `kwargs_fit` to compute the cost function.
    # Moreover, this cost_function is NOT backend-agnostic.
    @pytest.fixture
    def cost_function(self, kwargs_init, kwargs_fit) -> dict[str, pyct.OpT]:
        import pycsou.abc as pyca
        import pycsou.operator.blocks as pycb

        # The value of `b` determines the cost function.
        # Moreover several `b` may be provided.
        # A unique cost function is obtained by flattening `b` and block-expanding the quadratic
        # cost accordingly.

        A = kwargs_init["A"]
        b = kwargs_fit["b"]
        N_b = int(np.prod(b.shape[:-1]))
        func = pyca.QuadraticFunc(
            shape=(1, N_b * A.dim),
            Q=pycb.block_diag((A,) * N_b),
            c=pyca.LinFunc.from_array(-b.reshape(-1), enable_warnings=False),
        )
        return dict(x=func)

    # Note the different signature compared to SolverT.test_value_fit().
    def test_value_fit(
        self,
        solver,
        _kwargs_fit_xp,
        width,
        cost_function,
        ground_truth,
    ):
        # Due to different b/x0 domain combinations, it is difficult to use SolverT.test_value_fit().
        # To test convergence, we just check that the (strongly-convex) cost function is minimized.
        with pycrt.Precision(width):
            solver.fit(**_kwargs_fit_xp.copy())
        data, _ = solver.stats()

        cost_value = dict()
        for var, c_func in cost_function.items():
            x_gt = ground_truth[var]
            c_gt = c_func.apply(pycu.to_NUMPY(x_gt))
            # cost_function defined for NUMPY inputs only. [CG-peculiarity.]

            x_opt = data[var].reshape(-1)
            c_opt = c_func.apply(pycu.to_NUMPY(x_opt))
            # cost_function defined for NUMPY inputs only. [CG-peculiarity.]

            cost_value[var] = allclose(c_gt, c_opt, as_dtype=width.value)
        assert all(cost_value.values())
