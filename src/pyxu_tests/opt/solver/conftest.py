import collections.abc as cabc
import datetime as dt
import functools
import itertools
import operator
import time

import numpy as np
import pytest
import scipy.optimize as sopt

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.opt.stop as pxs
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct


def funcs(N: int, seed: int = 0) -> cabc.Sequence[tuple[pxt.OpT, pxt.OpT]]:
    # Sequence of functional descriptors. (More terms can be added.)
    #
    # Used to create strongly-convex functionals.
    #
    # These functions MUST be backend-agnostic.
    import pyxu.operator.func as pxf
    import pyxu.operator.linop as pxl
    import pyxu_tests.operator.examples.test_unitop as unitop

    L2 = pxf.SquaredL2Norm(dim=N)
    Id = pxl.IdentityOp(dim=N)

    rng = np.random.default_rng(seed)
    f = [
        (  # f1(x) = \norm{A1 x}{2}^{2}
            L2,
            pxl.HomothetyOp(cst=rng.uniform(1.1, 1.3), dim=N),
        ),
        (  # f2(x) = \norm{A2 x - y2}
            L2.asloss(rng.uniform(1, 3)),
            unitop.Permutation(N=N),
        ),
        (  # f3(x) = sum(x)
            pxl.Sum(arg_shape=N),
            Id,
        ),
        (  # f4(x) = \norm{a x}{2}^{2}
            L2.argscale(rng.uniform(-5, -1.1)),
            Id,
        ),
        (  # f5(x) = cst * \norm{x + y5}{2}^{2}
            L2.argshift(rng.normal()) * rng.uniform(1.1, 3),
            Id,
        ),
    ]
    return f


def generate_funcs(descr, N_term: int) -> cabc.Sequence[tuple[pxt.OpT]]:
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
    # -> [ (a*b, c*d + e*f), (c*d, a*b + e*f), (e*f, a*b + c*d), ]
    assert 1 <= N_term <= len(descr)

    def chain(x, y):
        comp = x * y
        comp.diff_lipschitz = comp.estimate_diff_lipschitz(method="svd")
        return comp

    stream = []
    for d in itertools.permutations(descr, N_term - 1):
        # split into N_term subsets
        parts = [chain(*dd) for dd in d]  # The N_term - 1 first subsets each correspond to a single term
        # The remaining terms are summed in the last subset (no need to permute them)
        to_sum = list(set(descr) - set(d))
        p = functools.reduce(operator.add, [chain(*dd) for dd in to_sum])
        p.diff_lipschitz = p.estimate_diff_lipschitz(method="svd")
        parts.append(p)
        stream.append(tuple(parts))
    return stream


class SolverT(ct.DisableTestMixin):
    # Helper Functions --------------------------------------------------------
    @staticmethod
    def _check_allclose(nd_1: dict, nd_2: dict) -> bool:
        same_keys = set(nd_1.keys()) == set(nd_2.keys())
        if not same_keys:
            return False

        def standardize(x: pxt.NDArray) -> np.ndarray:
            # _check_allclose() compares inputs for equality using np.allclose().
            # Problem: we want NaNs to compare equal, which is not the case with np.allclose().
            # [Whether this makes sense is test-dependant: safeguards are placed where relevant.]
            # We therefore use np.nan_to_num() to bound potential NaNs and make them compare
            # equally.
            y = pxu.to_NUMPY(x)
            z = np.nan_to_num(y)
            return z

        stats = dict()
        for k in nd_1.keys():
            stats[k] = np.allclose(
                standardize(nd_1[k]),
                standardize(nd_2[k]),
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

    @staticmethod
    def as_early_stop(kwargs: dict) -> dict:
        # Some tests look at state which does not require a solver to have converged
        # (mathematically).
        # This function adds a max-iter constraint to the kwargs_fit() dictionary to drastically
        # curb test time.
        kwargs = kwargs.copy()
        c1 = pxs.MaxIter(n=5)
        c2 = pxs.MaxDuration(t=dt.timedelta(seconds=5))
        kwargs["stop_crit"] = c1 | c2
        return kwargs

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def spec(self) -> tuple[pxt.SolverC, dict, dict]:
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
    def solver_klass(self, spec) -> pxt.SolverC:
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
                pxu.get_array_module(v, fallback=None)
                data[k] = xp.array(v)
            except Exception:
                # Not an NDArray -> no transformation
                pass
        return data

    @pytest.fixture
    def solver(self, solver_klass, kwargs_init) -> pxt.SolverT:
        # Solver instance used for most fit() tests.
        slvr = solver_klass(**kwargs_init)
        return slvr

    @pytest.fixture(
        # We do not run solvers with DASK inputs due to general slowness.
        # This is not a problem in practice: it is mostly operators which
        # need to be tested for DASK compliance.
        params=[
            pxd.NDArrayInfo.NUMPY.module(),
            pytest.param(
                pxd.NDArrayInfo.CUPY.module(),
                marks=pytest.mark.skipif(
                    not pxd.CUPY_ENABLED,
                    reason="GPU unsupported on this machine.",
                ),
            ),
        ]
    )
    def xp(self, request) -> pxt.ArrayModule:
        return request.param

    @pytest.fixture(params=pxrt.Width)
    def width(self, request) -> pxrt.Width:
        return request.param

    @pytest.fixture
    def cost_function(self, kwargs_init) -> dict[str, pxt.OpT]:
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
        def fun(x: np.ndarray, cost_f: pxt.OpT) -> tuple[float, np.ndarray]:  # f(x), \grad_{f}(x)
            val = cost_f.apply(x)
            grad = cost_f.grad(x)
            return float(val), grad

        rng = np.random.default_rng()
        data = dict()
        for log_var, cost_f in cost_function.items():
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
        self._skip_if_disabled()

        solver.fit(**self.as_early_stop(_kwargs_fit_xp))
        data, _ = solver.stats()

        stats = {k: pxu.get_array_module(v) == xp for (k, v) in data.items()}
        assert all(stats.values())

    def test_precCM_fit(self, solver, kwargs_fit, width):
        # solver output-precision match context manager
        self._skip_if_disabled()

        with pxrt.Precision(width):
            solver.fit(**self.as_early_stop(kwargs_fit))
            data, _ = solver.stats()

            stats = {k: v.dtype == width.value for (k, v) in data.items()}
            assert all(stats.values())

    @pytest.mark.parametrize("eps_threshold", [5e-3])
    @pytest.mark.parametrize("time_threshold", [dt.timedelta(seconds=5)])
    def test_value_fit(
        self,
        solver,
        _kwargs_fit_xp,
        width,
        cost_function,
        ground_truth,
        eps_threshold,  # rel-threshold
        time_threshold,  # max runtime
    ):
        # Ensure algorithm converges to ground-truth. (All backends/precisions.)
        self._skip_if_disabled()

        # SciPy ground truth may have converged closer to the solution than `solver` due to
        # different stopping criteria used. (Default or user-specified.)
        #
        # Since cost-functions are assumed strongly convex, a solver is assumed to converge to the
        # ground-truth if the following conditions hold:
        #
        #   1. |x_k+1 - x_k| <= eps |x_k|
        #   2. f(x_gt) <= f(x_N) < f(x_0), where `f` denotes the cost function
        #   3. max runtime <= threshold
        #
        # (1) ensures solver iterates form a Cauchy sequence, hence converge to some limit.
        # (2) ensures the cost function decreases, hence the solver is converging to something.
        # (3) just avoids a solver running forever.
        #
        # NOTE: f(x_N) < f(x_0) implies initial_points specified in kwargs_fit() should be chosen
        # `far` from the ground-truth.
        # Test writers are responsible of setting x_0 such that this condition holds.
        var_crit = [
            pxs.RelError(
                eps=eps_threshold,
                var=k,
                f=None,
                satisfy_all=True,
            )
            for k in cost_function.keys()
        ]
        time_crit = pxs.MaxDuration(t=time_threshold)

        kwargs_fit = _kwargs_fit_xp.copy()
        kwargs_fit.update(
            stop_crit=time_crit | functools.reduce(operator.and_, var_crit),
            mode=pxa.Mode.MANUAL,
        )
        with pxrt.Precision(width):
            solver.fit(**kwargs_fit)

        data_0, _ = solver.stats()
        for data in solver.steps():
            pass
        data_opt, history = solver.stats()

        # ensure solver ended because rel-err threshold reached, not timeout
        relerr_decrease = dict()
        for var in cost_function.keys():
            if data_0[var].ndim == 1:
                k = f"RelError[{var}]"
            else:
                k = f"RelError[{var}]_max"
            relerr_decrease[var] = history[k][-1] <= eps_threshold
        assert all(relerr_decrease.values())

        # ensure cost-function decreased from start to end
        cost_decrease = dict()
        for var, c_func in cost_function.items():
            c_0 = c_func.apply(data_0[var])
            c_opt = c_func.apply(data_opt[var])
            c_gt = c_func.apply(ground_truth[var])

            # Numerical errors can make c_gt > c_opt at convergence.
            # We piggy-back on less_equal() from the operator test suite to handle this case.
            bound_1 = ct.less_equal(c_gt, c_opt, as_dtype=width.value)

            # The cost function MUST decrease. (See test-level comment for details.)
            bound_2 = c_opt < c_0

            cost_decrease[var] = np.all(bound_1 & bound_2)
        assert all(cost_decrease.values())

    def test_transparent_fit(self, solver, kwargs_fit):
        # Running solver twice returns same results.
        self._skip_if_disabled()

        # `kwargs_fit` values may be coerced on 1st call to solver.fit(), in which case transparency
        # will not be tested. (May happen if solver takes `kwargs_fit` values as-is, and does not
        # make a copy internally.)
        # Solution: Manually enforce precision of all input arrays prior to calling solver.fit().
        kw_fit = dict()
        for k, v in kwargs_fit.items():
            try:
                v_c = pxrt.coerce(v)
            except Exception:
                v_c = v
            kw_fit[k] = v_c
        kw_fit = self.as_early_stop(kw_fit)

        solver.fit(**kw_fit.copy())
        data1, _ = solver.stats()
        solver.fit(**kw_fit.copy())
        data2, _ = solver.stats()

        assert self._check_allclose(data1, data2)

    # -------------------------------------------------------------------------
    # The Solver API has a complex structure where sub-classes are expected to override a subset of
    # methods, i.e m_init(), m_step(), default_stop_crit(), objective_func(). The tests below ensure
    # Solver sub-classes do not break the Solver API by overriding forbidden methods.

    @pytest.mark.parametrize("track_objective", [True, False])
    def test_objective_func_tracked(
        self,
        solver,
        kwargs_fit,
        track_objective,
    ):
        # Ensure objective_func value present in history.
        self._skip_if_disabled()

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
        self._skip_if_disabled()

        kwargs_init = kwargs_init.copy()
        kwargs_init.update(stop_rate=stop_rate)
        solver = solver_klass(**kwargs_init)

        kwargs_fit = kwargs_fit.copy()
        kwargs_fit["stop_crit"] = pxs.MaxIter(n=10 * stop_rate)  # avoids infinite loop if solver does not converge.
        solver.fit(**kwargs_fit)

        _, history = solver.stats()
        assert np.all(history["iteration"] % stop_rate == 0)

    def test_data_contains_logvar(self, solver, kwargs_fit):
        # logged data only contains variables from log_var.
        self._skip_if_disabled()

        log_var = solver._astate["log_var"]
        solver.fit(**self.as_early_stop(kwargs_fit))
        data, _ = solver.stats()
        assert set(log_var) == set(data.keys())

    def test_halt_implies_disk_storage(self, solver_klass, kwargs_init, kwargs_fit, tmp_path):
        # When solver stops, data+log files exist at specified folder.
        self._skip_if_disabled()

        kwargs_init = kwargs_init.copy()
        kwargs_init.update(folder=tmp_path, exist_ok=True)
        solver = solver_klass(**kwargs_init)
        solver.fit(**self.as_early_stop(kwargs_fit))

        assert solver.workdir.resolve() == tmp_path.resolve()
        assert solver.logfile.exists()
        assert solver.datafile.exists()

    def test_disk_value_matches_memory(self, solver, kwargs_fit):
        # Datafile content (values) match in-memory data after halt.
        self._skip_if_disabled()

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
        self._skip_if_disabled()

        with pxrt.Precision(width):
            solver.fit(**self.as_early_stop(kwargs_fit))

            disk = np.load(solver.datafile)
            data_disk = {k: v for (k, v) in disk.items() if k != "history"}
            data_mem, _ = solver.stats()

            assert self._check_prec_match(data_disk, data_mem)

    def test_transparent_mode(self, solver, kwargs_fit):
        # All execution modes return same results.
        self._skip_if_disabled()

        data = dict()
        for m in [pxa.Mode.BLOCK, pxa.Mode.MANUAL]:
            kwargs_fit = self.as_early_stop(kwargs_fit)
            kwargs_fit.update(mode=m)
            solver.fit(**kwargs_fit)
            if m == pxa.Mode.BLOCK:
                pass
            elif m == pxa.Mode.ASYNC:
                while solver.busy():
                    time.sleep(0.5)
                solver.stop()
            else:  # m == pxa.Mode.MANUAL
                for _ in solver.steps():
                    pass
            d, _ = solver.stats()
            data[m] = d

        stats = dict()
        for k1, k2 in itertools.combinations(data.keys(), r=2):
            stats[(k1, k2)] = self._check_allclose(data[k1], data[k2])
        assert all(stats.values())
