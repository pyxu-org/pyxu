import itertools

import numpy as np
import pytest

import pyxu.info.ptype as pxt
import pyxu.opt.solver as pxs
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.opt.solver.conftest as conftest


class TestCG(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pxt.SolverC, dict, dict]]:
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pxs.CG,
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
    def spec(self, request) -> tuple[pxt.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit

    # Note the different signature compared to SolverT.cost_function():
    # we need access to `b` from `kwargs_fit` to compute the cost function.
    # Moreover, this cost_function is NOT backend-agnostic.
    @pytest.fixture
    def cost_function(self, kwargs_init, kwargs_fit) -> dict[str, pxt.OpT]:
        import pyxu.abc as pxa
        import pyxu.operator.blocks as pxb

        # The value of `b` determines the cost function.
        # Moreover several `b` may be provided.
        # A unique cost function is obtained by flattening `b` and block-expanding the quadratic
        # cost accordingly.

        A = kwargs_init["A"]
        b = kwargs_fit["b"]
        N_b = int(np.prod(b.shape[:-1]))
        func = pxa.QuadraticFunc(
            shape=(1, N_b * A.dim),
            Q=pxb.block_diag((A,) * N_b),
            c=pxa.LinFunc.from_array(-b.reshape(-1), enable_warnings=False),
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
        self._skip_if_disabled()

        # Due to different b/x0 domain combinations, it is difficult to use SolverT.test_value_fit().
        # To test convergence, we just check that the (strongly-convex) cost function is minimized.
        with pxrt.Precision(width):
            solver.fit(**_kwargs_fit_xp.copy())
        data, _ = solver.stats()

        cost_value = dict()
        for var, c_func in cost_function.items():
            x_gt = ground_truth[var]
            c_gt = c_func.apply(pxu.to_NUMPY(x_gt))
            # cost_function defined for NUMPY inputs only. [CG-peculiarity.]

            x_opt = data[var].reshape(-1)
            c_opt = c_func.apply(pxu.to_NUMPY(x_opt))
            # cost_function defined for NUMPY inputs only. [CG-peculiarity.]

            cost_value[var] = ct.allclose(c_gt, c_opt, as_dtype=width.value)
        assert all(cost_value.values())
