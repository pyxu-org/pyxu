import itertools
import time

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


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

    @pytest.fixture(params=pycd.supported_array_modules())
    def xp(self, request) -> pyct.ArrayModule:
        return request.param

    @pytest.fixture(params=pycrt.Width)
    def width(self, request) -> pycrt.Width:
        return request.param

    @pytest.fixture
    def ground_truth(self, solver, kwargs_fit) -> dict:
        # The output when computing via NUMPY, assumed correct.
        solver.fit(**kwargs_fit)
        data, _ = solver.stats()
        return data

    # Tests -------------------------------------------------------------------
    def test_backend_fit(self, solver, _kwargs_fit_xp, xp):
        # solver output-backend match inputs
        solver.fit(**_kwargs_fit_xp)
        data, _ = solver.stats()

        stats = {k: pycu.get_array_module(v) == xp for (k, v) in data.items()}
        assert all(stats.values())

    def test_precCM_fit(self, solver, kwargs_fit, width):
        # solver output-precision match context manager
        with pycrt.Precision(width):
            solver.fit(**kwargs_fit.copy())
            data, _ = solver.stats()

            stats = {k: v.dtype == width.value for (k, v) in data.items()}
            assert all(stats.values())

    def test_value_fit(self, solver, _kwargs_fit_xp, ground_truth):
        # ensure output computed with backend=xp matches ground_truth NumPy result.
        solver.fit(**_kwargs_fit_xp.copy())
        data, _ = solver.stats()
        assert self._check_allclose(data, ground_truth)

    def test_transparent_fit(self, solver, kwargs_fit):
        # Running solver twice returns same results.
        solver.fit(**kwargs_fit.copy())
        data1, _ = solver.stats()
        solver.fit(**kwargs_fit.copy())
        data2, _ = solver.stats()

        assert self._check_allclose(data1, data2)

    @pytest.mark.parametrize("track_objective", [True, False])
    def test_objective_func_tracked(self, solver, kwargs_fit, track_objective):
        # Ensure objective_func value present in history.
        kwargs_fit = kwargs_fit.copy()
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
        solver.fit(**kwargs_fit.copy())
        data, _ = solver.stats()
        assert set(log_var) == set(data.keys())

    def test_halt_implies_disk_storage(
        self,
        solver_klass,
        kwargs_init,
        kwargs_fit,
        tmp_path,
    ):
        # When solver stops, data+log files exist at specified folder.
        kwargs_init = kwargs_init.copy()
        kwargs_init.update(folder=tmp_path, exist_ok=True)
        solver = solver_klass(**kwargs_init)
        solver.fit(**kwargs_fit.copy())

        assert solver.workdir.resolve() == tmp_path.resolve()
        assert solver.logfile.exists()
        assert solver.datafile.exists()

    def test_disk_value_matches_memory(self, solver, kwargs_fit):
        # Datafile content (values) match in-memory data after halt.
        solver.fit(**kwargs_fit.copy())

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
            solver.fit(**kwargs_fit.copy())

            disk = np.load(solver.datafile)
            data_disk = {k: v for (k, v) in disk.items() if k != "history"}
            data_mem, _ = solver.stats()

            assert self._check_prec_match(data_disk, data_mem)

    def test_transparent_mode(self, solver, kwargs_fit):
        # All execution modes return same results.
        data = dict()
        for m in [pyca.Mode.BLOCK, pyca.Mode.MANUAL]:
            kwargs_fit = kwargs_fit.copy()
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
