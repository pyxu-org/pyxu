import collections.abc as cabc
import functools
import itertools
import operator

import numpy as np
import pytest

import pyxu.abc.operator as pxa
import pyxu.info.ptype as pxt
import pyxu.operator as pxo
import pyxu.operator.func as pxf
import pyxu.opt.solver as pxs
import pyxu_tests.opt.solver.conftest as conftest


def generate_funcs_K(descr, N_term) -> cabc.Sequence[tuple[pxt.OpT]]:
    # Take description of many functionals, i.e. output of funcs(), and return a stream of
    # length-N_term tuples, where each of the first N_term - 1 terms of each tuple is a functional created by composing
    # and summing a subset of `descr`, and the last term is a tuple (func, op) that is an element of `descr`.
    #
    # Examples
    # --------
    # generate_funcs([(a, b), (c, d), (e, f)], 2)
    # -> [ (c*d + e*f, (a, b)), (a*b + e*f, (c, d)), (a*b + c*d, (e, f)), ]
    # generate_funcs([(a, b), (c, d), (e, f)], 3)
    # -> [ (c*d, e*f, (a, b)), (a*b, e*f, (c, d)),
    #      (e*f, c*d, (a, b)), (a*b, c*d, (e, f)),
    #      (e*f, a*b, (c, d)), (c*d, a*b, (e, f)) ]

    assert 2 <= N_term <= len(descr)  # Must have at least 2 terms because h cannot be a sum

    def chain(x, y):
        comp = x * y
        comp.diff_lipschitz = comp.estimate_diff_lipschitz()
        return comp

    stream = []
    for d in itertools.permutations(descr, N_term - 1):
        to_sum = list(set(descr) - set(d))
        p = functools.reduce(operator.add, [chain(*dd) for dd in to_sum])
        p.diff_lipschitz = p.estimate_diff_lipschitz()
        stream.append([*d, p])  # The last subset is a func while the others are (func, op) tuples

    stream_K = []
    for part in stream:
        for n in range(N_term - 1):
            p = part.copy()
            del p[n]  # The n-th subset is used for the (h, K) pair
            p = [chain(*dd) for dd in p[:-1]]  # Compose all but the last subset (which is already a func)
            stream_K.append((*p, part[-1], part[n]))  # Append the last subset and the (h, K) tuple
    return stream_K


class MixinPDS(conftest.SolverT):
    disable_test = {
        # Several PDS tests fail with default parameters with certain combinations of objective functionals. They all
        # pass when removing the time threshold and adding a minimum number of iterations (to avoid solver stopping too
        # close to its initial point), but they are extremely slow.
        "test_value_fit",
    }

    dim = 5  # Dimension of input vector for tests

    @classmethod
    def generate_init_kwargs(cls, has_f: bool, has_g: bool, has_h: bool, has_K: bool) -> list[dict]:
        # Returns a stream of dictionaries for the init_kwargs fixture of the solver based on whether that solver has
        # arguments f, g, h and K. All possible combinations of the output of `funcs` are tested.

        funcs = conftest.funcs(cls.dim)
        stream1 = conftest.generate_funcs(funcs, N_term=1)
        stream2 = conftest.generate_funcs(funcs, N_term=2)

        kwargs_init = []
        if has_f:
            kwargs_init.extend([dict(f=f) for (f, *_) in stream1])
        if has_g:
            kwargs_init.extend([dict(g=g) for (g, *_) in stream1])
            if has_f:
                kwargs_init.extend([dict(f=f, g=g) for (f, g) in stream2])
        if has_h:
            kwargs_init.extend([dict(h=h) for (h, *_) in stream1])
            if has_f:
                kwargs_init.extend([dict(f=f, h=h) for (f, h) in stream2])
            if has_g:
                kwargs_init.extend([dict(g=g, h=h) for (g, h) in stream2])
                if has_f:
                    stream3 = conftest.generate_funcs(funcs, N_term=3)
                    kwargs_init.extend([dict(f=f, g=g, h=h) for (f, g, h) in stream3])

        if has_K:
            stream2_K = generate_funcs_K(funcs, N_term=2)
            if has_f:
                kwargs_init.extend([dict(f=f, h=h, K=K) for (f, (h, K)) in stream2_K])
            if has_g:
                kwargs_init.extend([dict(g=g, h=h, K=K) for (g, (h, K)) in stream2_K])
                if has_f:
                    stream3_K = generate_funcs_K(funcs, N_term=3)
                    kwargs_init.extend([dict(f=f, g=g, h=h, K=K) for (f, g, (h, K)) in stream3_K])
        return kwargs_init

    @staticmethod
    def fenchel_conjugate(quad_func: pxa.QuadraticFunc):
        # Fenchel conjugate of a quadratic function (up to the constant term)
        Q, c, _ = quad_func._quad_spec()
        Q_fench = Q.dagger()
        c_fench = -c * Q_fench
        return pxa.QuadraticFunc(shape=quad_func.shape, Q=Q_fench, c=c_fench)

    @pytest.fixture
    def spec(self, klass, init_kwargs, fit_kwargs) -> tuple[pxt.SolverC, dict, dict]:
        return klass, init_kwargs, fit_kwargs

    @pytest.fixture(params=[1, 2, 3])
    def tuning_strategy(self, request) -> int:
        return request.param

    @pytest.fixture(params=["CV", "PD3O"])
    def base(self, request) -> pxt.SolverC:
        bases = {"CV": pxs.CV, "PD3O": pxs.PD3O}
        return bases[request.param]

    @pytest.fixture(params=["1d", "nd"])
    def x0(self, request) -> dict:
        # Multiple initial points
        return {"1d": np.full((self.dim,), 3.0), "nd": np.full((2, self.dim), 15.0)}[request.param]

    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return NotImplementedError

    @pytest.fixture
    def init_kwargs(self) -> dict:
        return NotImplementedError

    @pytest.fixture
    def fit_kwargs(self, x0, tuning_strategy) -> dict:
        # Overriden only for ADMM
        return dict(
            x0=x0,
            tuning_strategy=tuning_strategy,
        )

    @pytest.fixture
    def cost_function(self, init_kwargs) -> dict[str, pxt.OpT]:
        kwargs = [init_kwargs.get(k, pxf.NullFunc(dim=self.dim)) for k in ("f", "g")]
        func = kwargs[0] + kwargs[1]
        out = dict()
        if init_kwargs.get("h") is not None:
            if func._name != "NullFunc":
                h = init_kwargs.get("h")
                K = init_kwargs.get("K", pxo.IdentityOp(dim=self.dim))
                if isinstance(h, pxa.QuadraticFunc) and isinstance(func, pxa.QuadraticFunc):
                    Q_h, _, _ = h._quad_spec()
                    Q_f, _, _ = func._quad_spec()
                    if (Q_h._name == "IdentityOp" or Q_h._name == "HomothetyOp") and (
                        Q_f._name == "IdentityOp" or Q_f._name == "HomothetyOp"
                    ):
                        # The dual cost is guaranteed to have finite values for QuadraticFuncs (it is infinite for
                        # LinFuncs). We only compute the dual cost for QuadraticFuncs with explicit inverses
                        out["z"] = self.fenchel_conjugate(h) + self.fenchel_conjugate(func) * (-K.T)
            func += h * K
        out["x"] = func
        return out


class TestPD3O(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.PD3O

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=True, has_g=True, has_h=True, has_K=True))
    def init_kwargs(self, request) -> dict:
        return request.param


class TestCV(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.CV

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=True, has_g=True, has_h=True, has_K=True))
    def init_kwargs(self, request) -> dict:
        return request.param


class TestCP(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.CP

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=False, has_g=True, has_h=True, has_K=True))
    def init_kwargs(self, request, base) -> dict:
        kwargs = request.param
        kwargs.update({"base": base})
        return kwargs


class TestLV(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.LV

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=True, has_g=False, has_h=True, has_K=True))
    def init_kwargs(self, request) -> dict:
        return request.param


class TestDY(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.DY

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=True, has_g=True, has_h=True, has_K=False))
    def init_kwargs(self, request) -> dict:
        return request.param


class TestDR(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.DR

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=False, has_g=True, has_h=True, has_K=False))
    def init_kwargs(self, request, base) -> dict:
        kwargs = request.param
        kwargs.update({"base": base})
        return kwargs


class TestADMM(MixinPDS):
    @staticmethod
    def nlcg_init_kwargs():
        # Returns init_kwargs to test the (otherwise untested) NLCG scenario
        funcs = conftest.funcs(MixinPDS.dim)

        def chain(x, y):
            comp = x * y
            comp.diff_lipschitz = comp.estimate_diff_lipschitz()
            return comp

        f = chain(*funcs.pop(2))  # f is a LinFunc (and not a QuadraticFunc) -> avoids CG scenario
        h = functools.reduce(operator.add, [chain(*ff) for ff in funcs])
        K = pxo.IdentityOp(dim=MixinPDS.dim)  # Hack to force NLCG scenario (if K = None, then classical ADMM)
        return dict(f=f, h=h, K=K)

    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.ADMM

    @pytest.fixture(
        params=[*MixinPDS.generate_init_kwargs(has_f=True, has_g=False, has_h=True, has_K=True), nlcg_init_kwargs()]
    )
    def init_kwargs(self, request) -> dict:
        return request.param

    @pytest.fixture
    def spec(self, klass, init_kwargs, fit_kwargs) -> tuple[pxt.SolverC, dict, dict]:
        # Overriden from base class
        isNLCG = (init_kwargs.get("K") is not None) and (not isinstance(init_kwargs.get("f"), pxa.QuadraticFunc))
        if (fit_kwargs["x0"].squeeze().ndim > 1) and isNLCG:
            pytest.skip("NLCG scenario with multiple initial points not supported.")
        return klass, init_kwargs, fit_kwargs


class TestFB(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.FB

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=True, has_g=True, has_h=False, has_K=False))
    def init_kwargs(self, request) -> dict:
        return request.param


class TestPP(MixinPDS):
    @pytest.fixture
    def klass(self) -> pxt.SolverC:
        return pxs.PP

    @pytest.fixture(params=MixinPDS.generate_init_kwargs(has_f=False, has_g=True, has_h=False, has_K=False))
    def init_kwargs(self, request, base) -> dict:
        kwargs = request.param
        kwargs.update({"base": base})
        return kwargs
