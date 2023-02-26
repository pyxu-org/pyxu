import collections.abc as cabc
import functools
import itertools
import operator
import time

import numpy as np
import pytest
import scipy.optimize as sopt

import pycsou.abc as pyca
import pycsou.opt.stop as pycs
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def funcs(N: int, seed: int = 0) -> cabc.Sequence[cabc.Sequence[pyct.OpT]]:
    # Sequence of functional descriptors. (More terms can be added.)
    #
    # Used to create strongly-convex functionals.
    import pycsou.operator.func as pycf
    import pycsou.operator.linop as pycl
    import pycsou_tests.operator.examples.test_normalop as normalop
    import pycsou_tests.operator.examples.test_unitop as unitop

    rng = np.random.default_rng(seed)
    f1 = (  # f1(x) = \norm{A1 x}{2}^{2}
        pycf.SquaredL2Norm(dim=N),
        pycl.HomothetyOp(cst=rng.uniform(1.1, 1.3), dim=N),
    )
    f2 = (  # f2(x) = \norm{A2 x - y2}
        pycf.SquaredL2Norm(dim=N).asloss(rng.uniform(1, 3)),
        unitop.Permutation(N=N),
    )
    f3 = (  # f3(x) = sum(x)
        pycl.Sum(arg_shape=N),
        pycl.IdentityOp(dim=N),
    )
    return (f1, f2, f3)


def generate_funcs(descr, N_term: int) -> cabc.Sequence[tuple[pyct.OpT]]:
    # Take description of many functionals, i.e. output of funcs(), and return a stream of
    # length-N_term tuples, where each term of the tuple is a functional created by summing a subset
    # of `descr`.
    #
    # Examples
    # --------
    # generate_funcs([(a, b), (c, d), (e, f)], 1)
    # -> [ (a*b + c*d + e*f,), ]
    #
    # generate_funcs([(a, b), (c, d), (e, f)], 2)
    # -> [ (a*b, c*d + e*f), (a*b + c*d, e*f), (c*d + e*f, a*b), ]
    assert 1 <= N_term <= len(descr)
    chain = lambda x, y: x * y

    stream = []
    for d in itertools.permutations(descr):
        # split into N_term subsets
        parts = [chain(*d[i]) for i in range(N_term - 1)]

        # The last subset may contain more than 1 term
        p = functools.reduce(operator.add, [chain(*dd) for dd in d[N_term - 1 :]])
        parts.append(p)

        stream.append(tuple(parts))
    return stream


class SolverT:
    # Helper Functions --------------------------------------------------------
    @staticmethod
    def _check_allclose(nd_1: dict, nd_2: dict) -> bool:
        same_keys = set(nd_1.keys()) == set(nd_2.keys())
        if not same_keys:
            return False

        stats = dict()
        for k in nd_1.keys():
            stats[k] = np.allclose(
                pycu.compute(nd_1[k]),
                pycu.compute(nd_2[k]),
            )
        return all(stats.values())

    @staticmethod
    def _check_prec_match(nd_1: dict, nd_2: dict) -> bool:
        same_keys = set(nd_1.keys()) == set(nd_2.keys())
        if not same_keys:
            return False

        stats = dict()
        for k in nd_1.keys():
            stats[k] = nd_1[k].dtype == nd_2[k].dtype
        return all(stats.values())

    @classmethod
    def _check_value_fit(
        cls,
        data: dict[str, pyct.NDArray],
        cost_func: dict[str, pyct.OpT],
        ground_truth: dict[str, pyct.NDArray],
    ):
        # SciPy ground truth may have converged closer to the solution than `solver` due to
        # different stopping criteria used.
        # Since we assume cost-functions are strongly convex, we sidestep potential discrepancies
        # between ground-truth/solver outputs by comparing the relative difference between their
        # cost-function values.
        # We piggy-back onto RelError to simplify code in cases where there are multiple values
        # being optimized in parallel.
        success = dict()
        for k, c_func in cost_func.items():
            crit = pycs.RelError(
                eps=5e-2,  # 5% relative discrepancy tolerated.
                var=k,
                f=c_func,
                satisfy_all=True,
            )
            crit.stop(ground_truth)  # load the ground truth
            success[k] = crit.stop(data)  # then ensure solver output is within tolerated objective-func range
        assert all(success.values())

    @staticmethod
    def as_early_stop(kwargs: dict) -> dict:
        # Some tests look at state which does not require a solver to have converged
        # (mathematically).
        # This function adds a max-iter constraint to the kwargs_fit() dictionary to drastically
        # curb test time.
        kwargs = kwargs.copy()
        kwargs["stop_crit"] = pycs.MaxIter(n=5)
        return kwargs

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def spec(self) -> tuple[pyct.SolverC, dict, dict]:
        # override in subclass to return:
        # * the operator (class) to test;
        # * the solver-specific kwargs to pass to __init__().
        # * the solver-specific kwargs to pass to fit().
        #
        # The triplet (solver_klass, kwargs_init, kwargs_fit) must be provided since we need to
        # manipulate solver creation/execution.
        #
        # IMPORTANT: it is assumed operators involved in the solver's instantiation are
        # backend-agnostic.
        raise NotImplementedError

    @pytest.fixture
    def solver_klass(self, spec) -> pyct.SolverC:
        return spec[0]

    @pytest.fixture
    def kwargs_init(self, spec) -> dict:
        kwargs = spec[1]
        kwargs.update(show_progress=False)
        return kwargs

    @pytest.fixture
    def kwargs_fit(self, spec) -> dict:
        return spec[2]

    @pytest.fixture
    def _kwargs_fit_xp(self, kwargs_fit, xp) -> dict:
        # Same as `kwargs_fit`, but with all NDArray inputs transformed to backend[xp].
        # For internal use only: do not override in sub-classes.
        data = kwargs_fit.copy()
        for k in data.keys():
            v = data[k]
            try:
                pycu.get_array_module(v, fallback=None)
                data[k] = xp.array(v)
            except:
                # Not an NDArray -> no transformation
                pass
        return data

    @pytest.fixture
    def solver(self, solver_klass, kwargs_init) -> pyct.SolverT:
        # Solver instance used for most fit() tests.
        slvr = solver_klass(**kwargs_init)
        return slvr

    @pytest.fixture(
        # We do not run solvers with DASK inputs due to general slowness.
        # This is not a problem in practice: it is mostly operators which
        # need to be tested for DASK compliance.
        params=[
            pycd.NDArrayInfo.NUMPY.module(),
            pytest.param(
                pycd.NDArrayInfo.CUPY.module(),
                marks=pytest.mark.skipif(
                    not pycd.CUPY_ENABLED,
                    reason="GPU unsupported on this machine.",
                ),
            ),
        ]
    )
    def xp(self, request) -> pyct.ArrayModule:
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request) -> pycrt.Width:
        return request.param

    @pytest.fixture
    def cost_function(self, kwargs_init) -> dict[str, pyct.OpT]:
        # override in subclass to create cost function minimized per logged variable.
        #
        # Inputs are assumed to be some linear combination of conftest.funcs().
        # The cost functions are hence guaranteed to be a strongly convex.
        raise NotImplementedError

    @pytest.fixture
    def ground_truth(self, xp, cost_function) -> dict:
        # Compute the optimal value of the cost functions.
        #
        # Must return the same output as solver.stats()[0], i.e. the data dictionary.
        def fun(x: np.ndarray, cost_f: pyct.OpT) -> (float, np.ndarray):  # f(x), \grad_{f}(x)
            val = cost_f.apply(x)
            grad = cost_f.grad(x)
            return float(val), grad

        rng = np.random.default_rng()
        data = dict()
        for (log_var, cost_f) in cost_function.items():
            res = sopt.minimize(
                fun=fun,
                args=(cost_f,),
                x0=rng.uniform(size=cost_f.dim),
                method="CG",
                jac=True,
            )
            assert res.success  # if raised, means cost function was not strongly-convex
            data[log_var] = xp.array(res.x)
        return data

    # Tests -------------------------------------------------------------------
    def test_backend_fit(self, solver, _kwargs_fit_xp, xp):
        # solver output-backend match inputs
        solver.fit(**self.as_early_stop(_kwargs_fit_xp))
        data, _ = solver.stats()

        stats = {k: pycu.get_array_module(v) == xp for (k, v) in data.items()}
        assert all(stats.values())

    def test_precCM_fit(self, solver, kwargs_fit, width):
        # solver output-precision match context manager
        with pycrt.Precision(width):
            solver.fit(**self.as_early_stop(kwargs_fit))
            data, _ = solver.stats()

            stats = {k: v.dtype == width.value for (k, v) in data.items()}
            assert all(stats.values())

    def test_value_fit(
        self,
        solver,
        _kwargs_fit_xp,
        cost_function,
        ground_truth,
    ):
        # ensure output computed with backend=xp matches ground_truth NumPy result.
        solver.fit(**_kwargs_fit_xp.copy())
        data, _ = solver.stats()
        self._check_value_fit(data, cost_function, ground_truth)

    def test_transparent_fit(self, solver, kwargs_fit):
        # Running solver twice returns same results.
        with pycrt.EnforcePrecision(False):
            kwargs_fit = self.as_early_stop(kwargs_fit)
            solver.fit(**kwargs_fit.copy())
            data1, _ = solver.stats()
            solver.fit(**kwargs_fit.copy())
            data2, _ = solver.stats()

        assert self._check_allclose(data1, data2)

    @pytest.mark.parametrize("track_objective", [True, False])
    def test_objective_func_tracked(
        self,
        solver,
        kwargs_fit,
        track_objective,
    ):
        # Ensure objective_func value present in history.
        kwargs_fit = self.as_early_stop(kwargs_fit)
        kwargs_fit.update(track_objective=track_objective)
        solver.fit(**kwargs_fit)
        _, history = solver.stats()

        tracked = any("objective_func" in f for f in history.dtype.fields)
        if track_objective:
            assert tracked
        else:
            assert not tracked

    @pytest.mark.parametrize("stop_rate", [1, 2, 3])
    def test_stop_rate_limits_history(
        self,
        solver_klass,
        kwargs_init,
        kwargs_fit,
        stop_rate,
    ):
        # stop_rate affects history granularity.
        kwargs_init = kwargs_init.copy()
        kwargs_init.update(stop_rate=stop_rate)
        solver = solver_klass(**kwargs_init)
        solver.fit(**kwargs_fit.copy())
        _, history = solver.stats()
        assert np.all(history["iteration"] % stop_rate == 0)

    def test_data_contains_logvar(self, solver, kwargs_fit):
        # logged data only contains variables from log_var.
        log_var = solver._astate["log_var"]
        solver.fit(**self.as_early_stop(kwargs_fit))
        data, _ = solver.stats()
        assert set(log_var) == set(data.keys())

    def test_halt_implies_disk_storage(self, solver_klass, kwargs_init, kwargs_fit, tmp_path):
        # When solver stops, data+log files exist at specified folder.
        kwargs_init = kwargs_init.copy()
        kwargs_init.update(folder=tmp_path, exist_ok=True)
        solver = solver_klass(**kwargs_init)
        solver.fit(**self.as_early_stop(kwargs_fit))

        assert solver.workdir.resolve() == tmp_path.resolve()
        assert solver.logfile.exists()
        assert solver.datafile.exists()

    def test_disk_value_matches_memory(self, solver, kwargs_fit):
        # Datafile content (values) match in-memory data after halt.
        solver.fit(**self.as_early_stop(kwargs_fit))

        disk = np.load(solver.datafile)
        data_disk = {k: v for (k, v) in disk.items() if k != "history"}
        hist_disk = disk["history"]
        data_mem, hist_mem = solver.stats()

        assert self._check_allclose(data_disk, data_mem)
        assert self._check_allclose(
            # transform structured arrays to dict.
            {k: hist_disk[k] for k in hist_disk.dtype.fields},
            {k: hist_mem[k] for k in hist_mem.dtype.fields},
        )

    def test_disk_prec_matches_memory(self, solver, kwargs_fit, width):
        # Datafile content (dtypes) match in-memory dtypes after halt.
        with pycrt.Precision(width):
            solver.fit(**self.as_early_stop(kwargs_fit))

            disk = np.load(solver.datafile)
            data_disk = {k: v for (k, v) in disk.items() if k != "history"}
            data_mem, _ = solver.stats()

            assert self._check_prec_match(data_disk, data_mem)

    def test_transparent_mode(self, solver, kwargs_fit):
        # All execution modes return same results.
        data = dict()
        for m in [pyca.Mode.BLOCK, pyca.Mode.MANUAL]:
            kwargs_fit = self.as_early_stop(kwargs_fit)
            kwargs_fit.update(mode=m)
            solver.fit(**kwargs_fit)
            if m == pyca.Mode.BLOCK:
                pass
            elif m == pyca.Mode.ASYNC:
                while solver.busy():
                    time.sleep(0.5)
                solver.stop()
            else:  # m == pyca.Mode.MANUAL
                for _ in solver.steps():
                    pass
            d, _ = solver.stats()
            data[m] = d

        stats = dict()
        for k1, k2 in itertools.combinations(data.keys(), r=2):
            stats[(k1, k2)] = self._check_allclose(data[k1], data[k2])
        assert all(stats.values())
