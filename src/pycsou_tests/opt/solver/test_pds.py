import itertools

import numpy as np
import pytest

import pycsou.operator.func as pycf
import pycsou.opt.solver as pycos
import pycsou.opt.stop as pycstop
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest


class TestCV(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.CV,
        ]
        K = PSDConvolution(N=N)
        K.lipschitz()
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                f=pycf.SquaredL2Norm(dim=N), g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestPD3O(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.PD3O,
        ]
        K = PSDConvolution(N=N)
        K.lipschitz()
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                f=pycf.SquaredL2Norm(dim=N), g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestCP(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.CP,
        ]
        K = PSDConvolution(N=N)
        K.lipschitz()
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestLV(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.LV,
        ]
        K = PSDConvolution(N=N)
        K.lipschitz()
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                f=pycf.SquaredL2Norm(dim=N), h=pycf.shift_loss(pycf.L1Norm(dim=N), 1), K=K
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestDY(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.DY,
        ]
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                f=pycf.SquaredL2Norm(dim=N),
                g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
                h=pycf.L1Norm(dim=N),
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestDR(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.DR,
        ]
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
                h=pycf.L1Norm(dim=N),
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestADMM(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.ADMM,
        ]
        f = pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1)
        f.diff_lipschitz()
        K = PSDConvolution(N=N)
        kwargs_init = [
            dict(f=f, h=pycf.L1Norm(dim=N), K=None),  # Classical ADMM (with prox)
            dict(  # Inner-loop CG ADMM
                f=pycf.QuadraticFunc(Q=K, c=pycf.NullFunc(dim=N)), h=pycf.shift_loss(pycf.L1Norm(dim=N), 1), K=K
            ),
            dict(f=f, h=pycf.L1Norm(dim=N), K=K),  # Inner-loop NLCG (with prox)
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
            ],
            solver_kwargs=[
                # MaxIter stopping criterion necessary for NLGC case with single precision (the default stopping
                # criterion is never satisfied)
                dict(stop_crit=pycstop.MaxIter(10) | pycstop.AbsError(1e-4)),
            ],
        )
        for config in itertools.product(*param_sweep.values()):
            d = dict(zip(param_sweep.keys(), config))
            kwargs_fit.append(d)

        data = itertools.product(klass, kwargs_init, kwargs_fit)
        return list(data)[:-1]  # Remove unsupported configuration NLCG with multiple initial points

    @pytest.fixture(params=spec_data(N=7))
    def spec(self, request) -> tuple[pyct.SolverC, dict, dict]:
        klass, kwargs_init, kwargs_fit = request.param
        return klass, kwargs_init, kwargs_fit


class TestFB(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.FB,
        ]
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                f=pycf.SquaredL2Norm(dim=N),
                g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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


class TestPP(conftest.SolverT):
    @staticmethod
    def spec_data(N: int) -> list[tuple[pyct.SolverC, dict, dict]]:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        klass = [
            pycos.PP,
        ]
        kwargs_init = [
            dict(  # Inner-loop NLCG (with prox)
                g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
            ),
        ]

        kwargs_fit = []
        param_sweep = dict(
            x0=[
                np.full((N,), 3.0),
                np.full((2, N), 15.0),  # multiple initial points
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
