import itertools

import numpy as np
import pytest

import pycsou.operator.func as pycf
import pycsou.opt.solver as pycos
import pycsou.opt.stop as pycstop
import pycsou.util.ptype as pyct
import pycsou_tests.opt.solver.conftest as conftest
from pycsou_tests.operator.examples.test_posdefop import PSDConvolution


class MixinPDS(conftest.SolverT):
    @pytest.fixture
    def spec(self, klass, init_kwargs, fit_kwargs) -> tuple[pyct.SolverC, dict, dict]:
        return klass, init_kwargs, fit_kwargs

    @pytest.fixture
    def N(self) -> int:
        return 7

    @pytest.fixture(params=["1d", "nd"])
    def x0(self, N, request) -> dict:
        # Multiple initial points
        return {"1d": np.full((N,), 3.0), "nd": np.full((2, N), 15.0)}[request.param]

    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return NotImplementedError

    @pytest.fixture
    def init_kwargs(self) -> dict:
        return NotImplementedError

    @pytest.fixture
    def fit_kwargs(self, x0) -> dict:
        # Overriden only for ADMM
        return dict(
            x0=x0,
        )


class TestPD3O(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.PD3O

    @pytest.fixture
    def init_kwargs(self, N) -> dict:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        K = PSDConvolution(N=N)
        K.lipschitz()
        return dict(
            f=pycf.SquaredL2Norm(dim=N), g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K
        )


class TestCP(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.CP

    @pytest.fixture
    def init_kwargs(self, N) -> dict:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        K = PSDConvolution(N=N)
        K.lipschitz()
        return dict(g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K)


class TestLV(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.LV

    @pytest.fixture
    def init_kwargs(self, N) -> dict:
        from pycsou_tests.operator.examples.test_posdefop import PSDConvolution

        K = PSDConvolution(N=N)
        K.lipschitz()
        return dict(  # Inner-loop NLCG (with prox)
            f=pycf.SquaredL2Norm(dim=N), h=pycf.shift_loss(pycf.L1Norm(dim=N), 1), K=K
        )


class TestCV(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.CV

    @pytest.fixture
    def init_kwargs(self, N: int) -> dict:
        K = PSDConvolution(N=N)
        K.lipschitz()
        return dict(
            f=pycf.SquaredL2Norm(dim=N), g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1), h=pycf.L1Norm(dim=N), K=K
        )


class TestDY(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.DY

    @pytest.fixture
    def init_kwargs(self, N: int) -> dict:
        return dict(
            f=pycf.SquaredL2Norm(dim=N),
            g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
            h=pycf.L1Norm(dim=N),
        )


class TestDR(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.DR

    @pytest.fixture
    def init_kwargs(self, N: int) -> dict:
        return dict(
            g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1),
            h=pycf.L1Norm(dim=N),
        )


class TestADMM(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.ADMM

    @pytest.fixture(params=["fNone", "classical", "cg", "nlcg"])
    def init_kwargs(self, request, N: int) -> dict:
        f = pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1)
        f.diff_lipschitz()
        K = PSDConvolution(N=N)
        f_quad = pycf.QuadraticFunc(Q=K, c=pycf.NullFunc(dim=N))
        return dict(
            fNone=dict(f=None, h=pycf.shift_loss(pycf.L1Norm(dim=N), 1), K=None),  # f None
            classical=dict(f=f, h=pycf.L1Norm(dim=N), K=None),  # Classical ADMM (with prox)
            cg=dict(f=f_quad, h=pycf.shift_loss(pycf.L1Norm(dim=N), 1), K=K),  # Sub-iterative CG
            nlcg=dict(f=f, h=pycf.L1Norm(dim=N), K=K),  # Sub-iterative NLCG
        )[request.param]

    @pytest.fixture
    def fit_kwargs(self, x0) -> dict:
        # Overriden from base class
        return dict(
            x0=x0,
            solver_kwargs=dict(stop_crit=pycstop.MaxIter(10) | pycstop.AbsError(1e-4)),
            # MaxIter stopping criterion necessary for NLGC case with single precision (the default stopping
            # criterion is never satisfied)
        )

    @pytest.fixture
    def spec(self, klass, init_kwargs, fit_kwargs) -> tuple[pyct.SolverC, dict, dict]:
        isNLCG = (init_kwargs["K"] is not None) and (not isinstance(init_kwargs["f"], pycf.QuadraticFunc))
        if (fit_kwargs["x0"].squeeze().ndim > 1) and isNLCG:
            pytest.skip(f"NLCG scenario with multiple initial points not supported.")
        return klass, init_kwargs, fit_kwargs


class TestFB(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.FB

    @pytest.fixture
    def init_kwargs(self, N: int) -> dict:
        return dict(
            f=pycf.SquaredL2Norm(dim=N),
            g=pycf.shift_loss(pycf.L1Norm(dim=N), 1),
        )


class TestPP(MixinPDS):
    @pytest.fixture
    def klass(self) -> pyct.SolverC:
        return pycos.PP

    @pytest.fixture
    def init_kwargs(self, N: int) -> dict:
        return dict(g=pycf.shift_loss(pycf.SquaredL2Norm(dim=N), 1))
