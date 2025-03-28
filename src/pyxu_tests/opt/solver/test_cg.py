import itertools

import numpy as np
import pytest
import scipy.linalg as splinalg

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.opt.solver as pxsl
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.opt.solver.conftest as conftest


class TestCG(conftest.SolverT):
    @staticmethod
    def spec_data(dim_shape: pxt.NDArrayShape) -> list[tuple[pxt.SolverC, dict, dict]]:
        from pyxu_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pxsl.CG,
        ]
        kwargs_init = [
            dict(A=PSDConvolution(dim_shape=dim_shape)),
        ]

        kwargs_fit = []
        param_sweep = dict(
            b=[
                np.ones(dim_shape, dtype="float64"),
                np.full((2, *dim_shape), -2, dtype="float64"),  # multiple problems in parallel
            ],
            x0=[
                None,  # let algorithm choose
                np.full(dim_shape, 30, dtype="float64"),
            ],
            restart_rate=[
                None,
                np.prod(dim_shape),
                2 * np.prod(dim_shape),
            ],
        )
        for config in itertools.product(*param_sweep.values()):
            d = dict(zip(param_sweep.keys(), config))
            kwargs_fit.append(d)

        data = itertools.product(klass, kwargs_init, kwargs_fit)
        return list(data)

    @pytest.fixture(
        params=[
            *spec_data((7,)),
            *spec_data((5, 3, 7)),
        ]
    )
    def spec(self, request) -> tuple[pxt.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit

    # Note the different signature compared to SolverT.cost_function():
    # we need access to `b` from `kwargs_fit` to compute the cost function.
    # Moreover, this cost_function is NOT backend-agnostic.
    @pytest.fixture
    def cost_function(self, kwargs_init, kwargs_fit) -> dict[str, pxt.OpT]:
        # The value of `b` determines the cost function.
        # Moreover several `b` may be provided.
        # A unique cost function is obtained by flattening `b` and block-expanding the quadratic
        # cost accordingly.

        A = kwargs_init["A"]
        A_f = A.asarray().reshape(A.codim_size, A.dim_size)

        b = kwargs_fit["b"]
        b_f = b.reshape(1, -1)

        N_b = int(np.prod(b.shape[: -A.dim_rank]))
        func = pxa.QuadraticFunc(
            dim_shape=A.dim_size * N_b,
            codim_shape=1,
            Q=pxa.LinOp.from_array(
                splinalg.block_diag(*((A_f,) * N_b)),
                enable_warnings=False,
            ),
            c=pxa.LinFunc.from_array(
                -b_f,
                enable_warnings=False,
            ),
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
